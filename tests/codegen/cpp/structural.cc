#include <cassert>
#include <eggc/all.hpp>

#include "ir/generated.h"
namespace ir = tepl_generated;
namespace d = ir::dialects::structural;
namespace r = ir::rules::structural;
using A = eggc::NoAnalysis<ir::OpNode>;
struct Host {
  std::optional<bool> supports_axis(std::uint64_t axis) const {
    return axis == 0;
  }
  std::optional<bool> allowed(const ir::TensorInfo&) const { return true; }
  std::optional<ir::OpAttrs> update(const ir::TensorInfo&,
                                    const ir::OpAttrs& attrs) const {
    return attrs;
  }
};
auto add(eggc::Id x, eggc::Id y) {
  return ir::OpNode::make(d::Op::Add, {}, {x, y});
}
auto dot(unsigned axis, eggc::Id x, eggc::Id y) {
  return ir::OpNode::make(d::Op::Dot, d::DotAttrs{axis}, {x, y});
}
template <class F>
void requires_analysis(F build) {
  bool rejected = false;
  try {
    build();
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("requires tensor analysis") !=
               std::string::npos;
  }
  assert(rejected);
}
int main() {
  requires_analysis([] { r::rule_shaped::build_rewrite<A>(); });
  requires_analysis([] { r::rule_tensor_host::build_rewrite<A>(Host{}); });
  requires_analysis([] { r::rule_tensor_derive::build_rewrite<A>(Host{}); });
  requires_analysis([] { r::rule_untyped_rhs::build_rewrite<A>(); });
  eggc::EGraph<ir::OpNode, A> graph;
  auto x = graph.add(ir::OpNode::input("x")),
       y = graph.add(ir::OpNode::input("y"));
  auto sum = graph.add(add(x, y)), product = graph.add(dot(0, x, y));
  graph.add(dot(1, x, y));
  eggc::run(graph, std::vector{r::rule_swap_add::build_rewrite<A>(),
                               r::rule_swap_dot::build_rewrite<A>(Host{})});
  auto swapped_sum = graph.lookup(add(y, x)),
       swapped_dot = graph.lookup(dot(0, y, x));
  assert(swapped_sum && graph.find(sum) == graph.find(*swapped_sum));
  assert(swapped_dot && graph.find(product) == graph.find(*swapped_dot));
  assert(!graph.lookup(dot(1, y, x)));
  auto zero_i32 = graph.add(ir::OpNode::literal("0", ir::DType::I32));
  auto with_zero = graph.add(add(x, zero_i32));
  eggc::run(graph, std::vector{r::rule_typed_rhs::build_rewrite<A>()});
  auto replacement = graph.lookup(add(zero_i32, x));
  assert(replacement && graph.find(with_zero) == graph.find(*replacement));
  for (auto dtype : {ir::DType::I32, ir::DType::F32}) {
    auto zero = graph.add(ir::OpNode::literal("0", dtype));
    auto root = graph.add(add(x, zero));
    eggc::run(graph, std::vector{r::rule_remove_zero::build_rewrite<A>()});
    assert(graph.find(root) == graph.find(x));
  }
  using Checked = ir::analysis::TensorAnalysis;
  ir::analysis::TensorBindingTable inputs;
  inputs.register_symbol("x", {{3}, ir::DType::I32});
  inputs.register_symbol("y", {{3}, ir::DType::I32});
  eggc::EGraph<ir::OpNode, Checked> checked{Checked{inputs}};
  x = checked.add(ir::OpNode::input("x"));
  y = checked.add(ir::OpNode::input("y"));
  auto root = checked.add(add(x, y));
  checked.rebuild();
  assert(checked.analysis_data(root).is_unknown());
  auto before = checked.node_count();
  eggc::run(checked, std::vector{r::rule_swap_add::build_rewrite()});
  assert(checked.node_count() == before && !checked.lookup(add(y, x)));
  r::rule_untyped_rhs::build_rewrite();
}
