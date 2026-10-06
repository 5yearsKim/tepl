#include <cassert>
#include <eggc/all.hpp>

#include "ir/generated.h"
#include "support.h"

namespace ir = tepl_generated;
using A = support::TestAnalysis;
using Graph = eggc::EGraph<ir::OpNode, A>;
using Op = ir::dialects::guard::Op;
struct Host {
  std::shared_ptr<unsigned> calls;
  std::optional<bool> allowed(const ir::TensorInfo&) const {
    ++*calls;
    return true;
  }
};
int main() {
  namespace p = ir::rewriting;
  namespace r = ir::rules::early_where;
  Graph graph;
  auto x = graph.add(ir::OpNode::input("x")),
       y = graph.add(ir::OpNode::input("y"));
  auto root = graph.add(ir::OpNode::make(Op::Pair, {}, {x, y}));
  graph.rebuild();
  unsigned reads = 0;
  auto metadata = [&](const Graph& g,
                      eggc::Id id) -> std::optional<ir::TensorInfo> {
    ++reads;
    return ir::TensorInfo{
        {g.find(id) == g.find(x) ? std::uint64_t(5) : std::uint64_t(2)},
        ir::DType::F32};
  };
  auto checks = support::configure_checks(graph, metadata,
                                          r::rule_early::match_checks<A>());
  auto matches = p::matches_at_with_checks(
      graph, root, r::rule_early::pattern(), checks, metadata);
  assert(matches.empty());
  assert(reads == 1);  // N fails before Y metadata is read.
  auto calls = std::make_shared<unsigned>(0);
  auto inference = [](ir::Op, std::span<const ir::TensorInfo>,
                      const ir::OpAttrs&) -> std::optional<ir::TensorInfo> {
    return ir::TensorInfo{{5}, ir::DType::F32};
  };
  auto copy = graph.add(ir::OpNode::make(Op::Copy, {}, {x}));
  graph.rebuild();
  auto boundary = support::configure_rule(
      graph, metadata, inference, r::rule_host_boundary::build<A>(Host{calls}));
  auto before = graph.node_count();
  eggc::run(graph, std::vector{boundary});
  assert(*calls > 0);
  assert(graph.find(copy) != graph.find(x));
  assert(graph.node_count() == before);
  *calls = 0;
  auto nested = support::configure_rule(
      graph, metadata, inference,
      r::rule_nested_host_boundary::build<A>(Host{calls}));
  eggc::run(graph, std::vector{nested});
  assert(*calls == 0);
  assert(graph.find(copy) != graph.find(x));
  // Unavailable metadata in a skipped branch must remain unevaluated.
  reads = 0;
  auto skipped = support::configure_checks(
      graph, metadata, r::rule_skipped_builtin::match_checks<A>());
  assert(p::matches_at_with_checks(
             graph, copy, r::rule_skipped_builtin::pattern(), skipped, metadata)
             .size() == 1);
  assert(reads == 0);
  // Limits count only surviving substitutions, so an invalid first candidate
  // must not consume the single allowed match.
  Graph limited;
  auto bad = limited.add(ir::OpNode::input("bad")),
       good = limited.add(ir::OpNode::input("good"));
  auto bad_root = limited.add(ir::OpNode::make(Op::Copy, {}, {bad}));
  auto good_root = limited.add(ir::OpNode::make(Op::Copy, {}, {good}));
  auto md = [bad](const Graph& g,
                  eggc::Id id) -> std::optional<ir::TensorInfo> {
    return ir::TensorInfo{
        {g.find(id) == g.find(bad) ? std::uint64_t(0) : std::uint64_t(2)},
        ir::DType::F32};
  };
  *calls = 0;
  auto rule = support::configure_rule(
      limited, md, inference, r::rule_host_boundary::build<A>(Host{calls}));
  eggc::RunOptions options;
  options.match_limit = 1;
  options.iteration_limit = 1;
  auto report = eggc::run(limited, std::vector{rule}, options);
  assert(limited.find(good) == limited.find(good_root));
  assert(limited.find(bad) != limited.find(bad_root));
  assert(*calls == 1);
  assert(report.history.front().matches == 1);
}
