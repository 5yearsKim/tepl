#include <cassert>
#include <eggc/all.hpp>
#include <limits>

#include "ir/generated.h"
#include "support.h"

namespace ir = tepl_generated;
using Op = ir::dialects::custom::Op;
using Axes = ir::dialects::custom::Axes;
using LocalAttrs = ir::dialects::custom::OpAttrs;
using A = support::TestAnalysis;
using Graph = eggc::EGraph<ir::OpNode, A>;
using Info = ir::TensorInfo;
auto metadata = [](const Graph&, eggc::Id) -> std::optional<Info> {
  return Info{{1, 2}, ir::DType::F32};
};
auto inference = [](ir::Op, std::span<const Info>,
                    const ir::OpAttrs&) -> std::optional<Info> {
  return Info{{1, 2}, ir::DType::F32};
};
struct Host {
  std::shared_ptr<std::vector<unsigned>> calls;
  bool wrong_schema = false;
  double float_value = 2;
  std::optional<bool> enabled(const Info&, const ir::OpAttrs&) const {
    calls->push_back(0);
    return true;
  }
  std::optional<ir::OpAttrs> update(const ir::OpAttrs& a) const {
    auto p = ir::attrs_schema_0(a);
    calls->push_back(p->axis);
    if (wrong_schema) return LocalAttrs(ir::dialects::custom::AnotherAttrs{1});
    return LocalAttrs(Axes{p->axis + 1, p->tags});
  }
  std::optional<bool> fails(const Info&) const {
    calls->push_back(9);
    return {};
  }
  std::optional<std::uint64_t> twice(std::uint64_t n) const {
    calls->push_back(1);
    return n * 2;
  }
  std::optional<bool> greater(std::uint64_t n, std::uint64_t bound) const {
    calls->push_back(2);
    return n > bound;
  }
  std::optional<std::int64_t> signed_value(const Info&) const { return 1; }
  std::optional<double> floating(const Info&) const { return float_value; }
  std::optional<Info> host_tensor(const Info& i) const {
    calls->push_back(3);
    return i;
  }
  std::optional<std::vector<std::uint64_t>> host_dims(
      std::span<const std::uint64_t> dims) const {
    calls->push_back(4);
    return std::vector<std::uint64_t>(dims.begin(), dims.end());
  }
  std::optional<bool> host_check(const Info&, std::span<const std::uint64_t>,
                                 bool flag, std::int64_t number, double f,
                                 const ir::OpAttrs&) const {
    calls->push_back(5);
    return flag && number == -1 && f == 1;
  }
};
template <class R>
std::optional<eggc::Id> apply_once(Graph& graph, const R& rule) {
  graph.rebuild();
  std::vector<eggc::Application<ir::OpNode, A>> actions;
  rule.custom_search(graph,
                     [&](auto action) {
                       actions.push_back(std::move(action));
                       return true;
                     },
                     {});
  if (actions.empty()) return {};
  auto rhs = actions.front().apply(graph);
  if (rhs) graph.merge(actions.front().target, *rhs);
  graph.rebuild();
  return rhs;
}
int main() {
  namespace p = ir::pattern;
  namespace r = ir::rules::custom;
  Graph graph;
  auto x = graph.add(ir::OpNode::input("x")),
       y = graph.add(ir::OpNode::input("y"));
  auto xy = graph.add(ir::OpNode::make(Op::Add, {}, {x, y}));
  auto commute = support::configure_rule(graph, metadata, inference,
                                         r::rule_commute::build_rewrite<A>());
  apply_once(graph, commute);
  auto yx = graph.lookup(ir::OpNode::make(Op::Add, {}, {y, x}));
  assert(yx && graph.find(xy) == graph.find(*yx));
  assert(ir::Op::from_name("Custom.plus") == ir::Op(Op::Add));
  auto neg = graph.add(ir::OpNode::make(Op::Negate, {}, {x}));
  auto twice = graph.add(ir::OpNode::make(Op::Add, {}, {neg, neg}));
  graph.rebuild();
  assert(p::matches_at(graph, twice, r::rule_bind::pattern()).size() == 1);
  auto root = graph.add(ir::OpNode::make(Op::Transform, Axes{1, {"tag"}}, {x}));
  graph.rebuild();
  auto calls = std::make_shared<std::vector<unsigned>>();
  auto chain = support::configure_rule(
      graph, metadata, inference,
      r::rule_derive_chain::build_rewrite<A>(Host{calls}));
  assert(apply_once(graph, chain));
  assert(*calls == std::vector<unsigned>({0, 1, 2}));
  auto derived =
      graph.lookup(ir::OpNode::make(Op::Transform, Axes{3, {"tag"}}, {x}));
  assert(derived && graph.find(*derived) == graph.find(root));
  auto bad = support::configure_rule(
      graph, metadata, inference,
      r::rule_derive_chain::build_rewrite<A>(Host{calls, true}));
  auto before = graph.node_count();
  calls->clear();
  assert(!apply_once(graph, bad));
  assert(graph.node_count() == before);
  assert(std::find(calls->begin(), calls->end(), 2) == calls->end());
  calls->clear();
  auto shorted = support::configure_rule(
      graph, metadata, inference,
      r::rule_short_circuit::build_rewrite<A>(Host{calls}));
  assert(apply_once(graph, shorted));
  assert(calls->empty());
  // Use a fresh graph so each host-order check has one structural match.
  Graph nested;
  auto input = nested.add(ir::OpNode::input("x"));
  nested.add(ir::OpNode::make(Op::Negate, {}, {input}));
  auto dims = [](const Graph&, eggc::Id) -> std::optional<Info> {
    return Info{{2}, ir::DType::F32};
  };
  calls->clear();
  assert(apply_once(nested,
                    support::configure_rule(
                        nested, dims, inference,
                        r::rule_nested_calls::build_rewrite<A>(Host{calls}))));
  assert(*calls == std::vector<unsigned>({1, 2}));
  Graph pipeline;
  input = pipeline.add(ir::OpNode::input("x"));
  pipeline.add(ir::OpNode::make(Op::Transform, Axes{1, {}}, {input}));
  calls->clear();
  assert(apply_once(
      pipeline, support::configure_rule(
                    pipeline, metadata, inference,
                    r::rule_typed_pipeline::build_rewrite<A>(Host{calls}))));
  assert(*calls == std::vector<unsigned>({3, 4, 5}));
  Graph overflow;
  input = overflow.add(ir::OpNode::input("x"));
  overflow.add(ir::OpNode::make(Op::Negate, {}, {input}));
  auto huge = [](const Graph&, eggc::Id) -> std::optional<Info> {
    return Info{{std::numeric_limits<std::uint64_t>::max()}, ir::DType::F32};
  };
  before = overflow.node_count();
  assert(!apply_once(
      overflow, support::configure_rule(overflow, huge, inference,
                                        r::rule_overflow::build_rewrite<A>())));
  assert(!apply_once(overflow, support::configure_rule(
                                   overflow, huge, inference,
                                   r::rule_divide_zero::build_rewrite<A>())));
  assert(overflow.node_count() == before);
  Graph numeric;
  input = numeric.add(ir::OpNode::input("x"));
  numeric.add(ir::OpNode::make(Op::Negate, {}, {input}));
  before = numeric.node_count();
  assert(!apply_once(
      numeric,
      support::configure_rule(
          numeric, metadata, inference,
          r::rule_signed_float::build_rewrite<A>(
              Host{calls, false, std::numeric_limits<double>::infinity()}))));
  assert(numeric.node_count() == before);
  assert(apply_once(numeric,
                    support::configure_rule(
                        numeric, metadata, inference,
                        r::rule_signed_float::build_rewrite<A>(Host{calls}))));
}
