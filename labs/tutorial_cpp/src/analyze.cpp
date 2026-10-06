#include <eggc/all.hpp>

#include "tepl_pattern_analyze/generated.h"
namespace tepl = tepl_pattern_analyze;
#include "utils.h"

namespace d = tepl::dialects::my_dialect;
namespace r = tepl::rules::add_dot;
using tepl::analysis::tensor_info;

struct Host {
  // Only axis-0 vector dots allow swapping.
  std::optional<bool> supports_axis(std::uint64_t axis) const {
    return axis == 0;
  }
};

int main() {
  auto graph_def = tepl::graphs::add_dot_sample::graph_add_dot_sample::build();
  // Configure shape and dtype analysis before insertion.
  auto bindings = graph_def.input_bindings();
  auto analysis = tepl::TensorAnalysis(std::move(bindings));
  eggc::EGraph<tepl::OpNode, tepl::TensorAnalysis> graph{std::move(analysis)};
  auto built = graph_def.insert_nodes(graph);

  // Print inferred input and intermediate types.
  for (auto [name, label] : {std::pair{"x", "x"},
                             {"y", "y"},
                             {"z", "z"},
                             {"a", "dot(x, y)"},
                             {"b", "dot(x, z)"}}) {
    auto info = tensor_info(graph, built.get(name).value()).value();
    std::cout << "  " << label << ": " << utils::tensor_type(info) << '\n';
  }

  // Record the original output type and expression.
  auto before_info = tensor_info(graph, built.root).value();
  auto [before_size, before] = eggc::Extractor(graph).find_best(built.root);

  // Build the basic rules and inherited factoring rule.
  std::vector rules{
      r::rule_associate_add::build(),
      r::rule_swap_add::build(),
      r::rule_swap_dot::build(Host{}),
      r::rule_factor_small_dot::build(),
  };
  auto report = eggc::run(graph, rules);
  utils::require(report.reason == eggc::StopReason::Saturated,
                 "Rewrites did not saturate");
  std::cout << "Stop reason: Saturated\n";
  utils::print_egraph(graph, built.root);

  // Extract a minimum-size expression.
  auto after_info = tensor_info(graph, built.root).value();
  auto [after_size, after] = eggc::Extractor(graph).find_best(built.root);

  // Inspect the inferred type of the factored add.
  auto sum = tepl::OpNode::make(
      d::Op::Add, {}, {built.get("y").value(), built.get("z").value()});
  auto sum_id = graph.lookup(sum).value();
  auto info = tensor_info(graph, sum_id).value();
  std::cout << "  Factored add(y, z): " << utils::tensor_type(info)
            << "\nBefore: " << utils::expression(before) << " (AST size "
            << before_size << ')' << "\nAfter:  " << utils::expression(after)
            << " (AST size " << after_size << ')'
            << "\nOutput: " << utils::tensor_type(after_info) << '\n';

  // Verify type preservation and size reduction.
  utils::require(info == tepl::TensorInfo{{3, 4}, tepl::DType::I32},
                 "Expected factored add to infer i32[3, 4]");
  utils::require(before_info == after_info &&
                     after_info == tepl::TensorInfo{{2, 4}, tepl::DType::I32},
                 "Expected output metadata to remain i32[2, 4]");
  utils::require(before_size == 7 && after_size == 5,
                 "Expected AST size to decrease from 7 to 5");
}
