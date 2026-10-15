#include <cassert>
#include <eggc/all.hpp>
#include <limits>

#include "ir/generated.h"

namespace ir = tepl_generated;
using A = ir::analysis::TensorAnalysis;
using Graph = eggc::EGraph<ir::OpNode, A>;
using Op = ir::dialects::toolkit::Op;
using Rewrite = eggc::Rewrite<ir::OpNode, A>;
Graph graph(std::vector<std::uint64_t> shape) {
  ir::analysis::TensorBindingTable inputs;
  inputs.insert("x", {shape, ir::DType::F32});
  Graph g{A(inputs)};
  auto x = g.add(ir::OpNode::input("x"));
  g.add(ir::OpNode::make(Op::Copy, {}, {x}));
  g.rebuild();
  return g;
}
int main() {
  namespace rules = ir::rules::builtins;
  ir::TensorInfo input{{2, 3}, ir::DType::F32};
  auto inferred = ir::analysis::infer_tensor(
      Op::Copy, std::span<const ir::TensorInfo>(&input, 1), ir::OpAttrs{});
  assert(inferred.kind == decltype(inferred)::Kind::Known);
  assert(inferred.value == input);
  auto g = graph({2, 3});
  auto x = g.find(0), copy = g.find(1);
  auto report = eggc::run(g, std::vector{rules::rule_utilities::build()});
  assert(g.find(x) == g.find(copy));
  assert(report.reason == eggc::StopReason::Saturated);
  auto shorted = graph({2, 3});
  eggc::run(shorted, std::vector{rules::rule_short_circuit::build()});
  assert(shorted.find(0) == shorted.find(1));
  std::vector<Rewrite> failures{rules::rule_failed_gather::build(),
                                rules::rule_failed_slice::build(),
                                rules::rule_failed_replace::build(),
                                rules::rule_failed_divisor::build(),
                                rules::rule_failed_range::build(),
                                rules::rule_failed_broadcast::build()};
  auto failed = graph({2, 3});
  auto before = failed.node_count();
  eggc::run(failed, failures);
  assert(failed.node_count() == before);
  assert(failed.find(0) != failed.find(1));
  auto overflow = graph({std::numeric_limits<std::uint64_t>::max(), 2});
  before = overflow.node_count();
  eggc::run(overflow, std::vector{rules::rule_failed_sum::build(),
                                  rules::rule_failed_product::build()});
  assert(overflow.node_count() == before);
  assert(overflow.find(0) != overflow.find(1));
  namespace b = ir::builtins;
  assert(b::take(b::common::product(std::vector<std::uint64_t>{
             std::numeric_limits<std::uint64_t>::max(), 2, 0})) == 0);
  assert(!b::common::div(std::numeric_limits<std::int64_t>::min(),
                         std::int64_t(-1)));
  assert(!b::common::rem(std::numeric_limits<std::int64_t>::min(),
                         std::int64_t(-1)));
  assert(!b::common::finite(std::numeric_limits<double>::infinity()));
}
