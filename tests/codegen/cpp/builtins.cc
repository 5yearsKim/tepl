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
struct Host {
  std::shared_ptr<unsigned> calls;
  std::optional<std::vector<std::uint64_t>> axes(const ir::TensorInfo&) const {
    ++*calls;
    return std::vector<std::uint64_t>{1, 2};
  }
  std::optional<std::uint64_t> axis(const ir::TensorInfo&) const {
    ++*calls;
    return 1;
  }
  std::optional<bool> legal(std::uint64_t) const {
    ++*calls;
    return true;
  }
  std::optional<ir::OpAttrs> infer_attrs(
      std::span<const std::uint64_t> axes) const {
    ++*calls;
    return ir::OpAttrs(ir::dialects::toolkit::OpAttrs(
        ir::dialects::toolkit::TagAttrs{{axes.begin(), axes.end()}}));
  }
};
int main() {
  namespace rules = ir::rules::builtins;
  auto g = graph({2, 3});
  auto x = g.find(0), copy = g.find(1);
  auto report = eggc::run(g, std::vector{rules::rule_utilities::build()});
  assert(g.find(x) == g.find(copy));
  assert(report.reason == eggc::StopReason::Saturated);
  auto calls = std::make_shared<unsigned>(0);
  Host host{calls};
  auto nested = graph({2, 3});
  eggc::run(nested, std::vector{rules::rule_nested_hosts::build(host)});
  assert(*calls > 0);
  *calls = 0;
  auto shorted = graph({2, 3});
  eggc::run(shorted, std::vector{rules::rule_short_circuit::build(host)});
  assert(*calls == 0);
  auto derived = graph({2, 3});
  eggc::run(derived, std::vector{rules::rule_derive_builtin::build(host)});
  assert(*calls > 0);
  assert(derived.lookup(
      ir::OpNode::make(Op::Tag, ir::dialects::toolkit::TagAttrs{{0, 0}}, {0})));
  std::vector<Rewrite> failures{rules::rule_failed_gather::build(),
                                rules::rule_failed_slice::build(),
                                rules::rule_failed_replace::build(),
                                rules::rule_failed_divisor::build(),
                                rules::rule_failed_range::build(),
                                rules::rule_failed_broadcast::build(),
                                rules::rule_failed_derive::build(host)};
  auto failed = graph({2, 3});
  auto before = failed.node_count();
  *calls = 0;
  eggc::run(failed, failures);
  assert(failed.node_count() == before);
  assert(failed.find(0) != failed.find(1));
  assert(*calls == 0);
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
