#include <eggc/all.hpp>

#include "tepl_pattern_basic/generated.h"
namespace tepl = tepl_pattern_basic;
#include "utils.h"

namespace r = tepl::rules::add_dot;
using Analysis = eggc::NoAnalysis<tepl::OpNode>;

struct Host {
  // Implements TEPL's $supports_axis.
  std::optional<bool> supports_axis(std::uint64_t axis) const {
    return axis == 0;
  }
};

int main() {
  // Build the starting graph without tensor metadata analysis.
  auto graph_def = tepl::graphs::add_dot_sample::graph_add_dot_sample::build();
  eggc::EGraph<tepl::OpNode, Analysis> graph;
  auto built = graph_def.insert_nodes(graph);

  // Build rules; swap_dot uses the host callback.
  std::vector rules{
      r::rule_associate_add::build<Analysis>(),
      r::rule_swap_add::build<Analysis>(),
      r::rule_swap_dot::build<Analysis>(Host{}),
  };

  // Run rewrites to saturation.
  auto report = eggc::run(graph, rules);
  utils::require(report.reason == eggc::StopReason::Saturated,
                 "Rewrites did not saturate");
  std::cout << "Stop reason: Saturated\n";
  utils::print_egraph(graph, built.root);

  // Extract a minimum-depth expression.
  auto [depth, best] =
      eggc::Extractor(graph, eggc::ast_depth_cost<tepl::OpNode>())
          .find_best(built.root);
  std::cout << "Minimum AST depth: " << depth << "\nBest expression:\n"
            << utils::expression(best) << '\n';
  utils::require(depth == 4, "Expected minimum AST depth 4");
}
