#include <cassert>
#include <eggc/all.hpp>

#include "ir/generated.h"

namespace ir = tepl_generated;
using Op = ir::dialects::runtime_test::Op;
using LocalAttrs = ir::dialects::runtime_test::OpAttrs;
using Params = ir::dialects::runtime_test::Params;
using A = ir::analysis::TensorAnalysis;
using Graph = eggc::EGraph<ir::OpNode, A>;
ir::OpAttrs attrs(std::uint64_t axis) { return LocalAttrs(Params{axis}); }
struct Host {
  std::shared_ptr<std::vector<std::uint64_t>> seen;
  std::optional<bool> allow(const ir::OpAttrs& a) const {
    auto p = ir::attrs_schema_0(a);
    seen->push_back(p->axis);
    return p->axis != 1;
  }
  std::optional<ir::OpAttrs> update(const ir::OpAttrs& a) const {
    auto p = ir::attrs_schema_0(a);
    seen->push_back(p->axis + 100);
    return attrs(p->axis + 10);
  }
};
struct OrderedHost {
  std::shared_ptr<std::vector<unsigned>> seen;
  bool fail = false;
  std::optional<std::uint64_t> left(const ir::TensorInfo&) const {
    seen->push_back(1);
    if (fail) return {};
    return 1;
  }
  std::optional<std::uint64_t> right(const ir::TensorInfo&) const {
    seen->push_back(2);
    return 2;
  }
  std::optional<bool> combine(std::uint64_t a, std::uint64_t b) const {
    seen->push_back(3);
    return a == 1 && b == 2;
  }
};
struct FloatHost {
  std::optional<double> floating(const ir::TensorInfo&) const { return 0; }
};
Graph graph(std::vector<std::uint64_t> shape = {2}) {
  ir::analysis::TensorBindingTable inputs;
  inputs.insert("x", {shape, ir::DType::F32});
  Graph g{A(inputs)};
  g.add(ir::OpNode::input("x"));
  return g;
}
void grouped_application_test(unsigned count, bool accept,
                              bool invalidate_metadata = false) {
  namespace p = ir::pattern;
  namespace r = ir::rules::runtime;
  using B = eggc::NoAnalysis<ir::OpNode>;
  using G = eggc::EGraph<ir::OpNode, B>;
  G g;
  std::optional<eggc::Id> root;
  std::vector<eggc::Id> inputs;
  for (unsigned i = 0; i < count; ++i) {
    auto x = g.add(ir::OpNode::input(std::to_string(i)));
    inputs.push_back(x);
    auto copy = g.add(ir::OpNode::make(Op::Copy, {}, {x}));
    if (root)
      g.merge(*root, copy);
    else
      root = copy;
  }
  g.rebuild();
  unsigned metadata_reads = 0, callbacks = 0;
  bool valid = true;
  auto metadata = [&](const G&, eggc::Id) -> std::optional<ir::TensorInfo> {
    ++metadata_reads;
    return ir::TensorInfo{valid ? std::vector<std::uint64_t>{2}
                                : std::vector<std::uint64_t>{2, 2},
                          ir::DType::F32};
  };
  auto rule = p::tensor_rewrite_checked_with_checks<B>(
      "grouped", r::rule_eliminate::pattern(), r::rule_eliminate::expression(),
      r::rule_eliminate::match_checks<B>(metadata), metadata,
      ir::analysis::TensorOutputInference{},
      [&](const auto&, const auto&,
          const auto&) -> std::optional<p::DerivedAttrs> {
        ++callbacks;
        return accept ? std::optional(p::DerivedAttrs{}) : std::nullopt;
      });
  std::vector<eggc::Application<ir::OpNode, B>> actions;
  rule.custom_search(g,
                     [&](auto action) {
                       actions.push_back(std::move(action));
                       return true;
                     },
                     {});
  assert(actions.size() == count);
  for (auto& action : actions) {
    auto replacement = action.apply(g);
    assert(replacement.has_value() == (accept && valid));
    if (replacement) g.merge(*root, *replacement);
    if (invalidate_metadata) valid = false;
  }
  // One search, one grouped rematch, and a bounded number of semantic reads
  // per substitution. The old per-action rematch takes count*count reads.
  assert(metadata_reads <= 6 * count);
  assert(callbacks == (invalidate_metadata ? 1u : count));
  if (accept && !invalidate_metadata)
    for (auto input : inputs) assert(g.find(input) == g.find(*root));
  g.rebuild();
}
int main() {
  namespace p = ir::pattern;
  namespace r = ir::rules::runtime;
  auto g = graph();
  auto x = eggc::Id(0);
  auto one = g.add(ir::OpNode::make(Op::Tag, Params{1}, {x}));
  auto two = g.add(ir::OpNode::make(Op::Tag, Params{2}, {x}));
  auto three = g.add(ir::OpNode::make(Op::Tag, Params{3}, {x}));
  g.merge(one, two);
  g.merge(one, three);
  g.rebuild();
  assert(p::matches_at(g, one, r::rule_witnesses::pattern()).size() == 3);
  auto seen = std::make_shared<std::vector<std::uint64_t>>();
  auto rule = r::rule_witnesses::build_rewrite(Host{seen});
  // Inspect one search/application rather than repeatedly applying the rule
  // which deliberately changes attributes on each iteration.
  std::vector<eggc::Application<ir::OpNode, A>> actions;
  assert(rule.custom_search(g,
                            [&](auto action) {
                              actions.push_back(std::move(action));
                              return true;
                            },
                            {}));
  assert(actions.size() == 1);
  auto replacement = actions.front().apply(g);
  assert(replacement);
  g.merge(one, *replacement);
  g.rebuild();
  for (auto axis : {12u, 13u}) {
    auto node = g.lookup(ir::OpNode::make(Op::Tag, Params{axis}, {x}));
    assert(node && g.find(*node) == g.find(one));
  }
  assert(!g.lookup(ir::OpNode::make(Op::Tag, Params{11}, {x})));
  assert(std::count(seen->begin(), seen->end(), 102) == 1);
  assert(std::count(seen->begin(), seen->end(), 103) == 1);
  // New searches own fresh witness batches, including descriptors inserted by
  // the previous application.
  actions.clear();
  rule.custom_search(g,
                     [&](auto action) {
                       actions.push_back(std::move(action));
                       return true;
                     },
                     {});
  assert(actions.size() == 1);
  replacement = actions.front().apply(g);
  assert(replacement);
  g.merge(one, *replacement);
  g.rebuild();
  for (auto axis : {22u, 23u}) {
    auto node = g.lookup(ir::OpNode::make(Op::Tag, Params{axis}, {x}));
    assert(node && g.find(*node) == g.find(one));
  }
  // Search budget counts one tensor substitution, regardless of witnesses.
  assert(!rule.custom_search(g, [](auto) { return false; }, {}));
  assert(
      !rule.custom_search(g, [](auto) { return true; }, [] { return true; }));
  // Host argument evaluation is left-to-right, and a failed earlier argument
  // prevents later calls in both debug and optimized code.
  for (bool fail : {false, true}) {
    auto ordered = graph();
    auto root = ordered.add(ir::OpNode::make(Op::Copy, {}, {0}));
    auto calls = std::make_shared<std::vector<unsigned>>();
    eggc::run(
        ordered,
        std::vector{r::rule_ordered::build_rewrite(OrderedHost{calls, fail})});
    if (fail) {
      assert(*calls == std::vector<unsigned>{1});
      assert(ordered.find(0) != ordered.find(root));
    } else {
      assert(calls->size() >= 3);
      assert((*calls)[0] == 1 && (*calls)[1] == 2 && (*calls)[2] == 3);
      assert(ordered.find(0) == ordered.find(root));
    }
  }
  // A malformed later RHS child is rejected before inference of any child or
  // insertion of an otherwise valid earlier child.
  auto rejected = graph();
  rejected.rebuild();
  unsigned inference_calls = 0;
  auto inference = [&](ir::Op, std::span<const ir::TensorInfo>,
                       const ir::OpAttrs&) -> std::optional<ir::TensorInfo> {
    ++inference_calls;
    return ir::TensorInfo{{2}, ir::DType::F32};
  };
  auto bad_rhs =
      p::TensorExpr::op(Op::Pair, p::AttrExpr::exact(),
                        {p::TensorExpr::op(Op::Copy, p::AttrExpr::exact(),
                                           {p::TensorExpr::var(0)}),
                         p::TensorExpr::op(Op::Tag, p::AttrExpr::derived(0),
                                           {p::TensorExpr::var(0)})});
  auto bad_rule = p::tensor_rewrite_checked<A>(
      "bad", p::TensorPattern::var(0), bad_rhs, ir::analysis::tensor_info,
      inference,
      [](const auto&, const auto&) -> std::optional<p::DerivedAttrs> {
        return p::DerivedAttrs{{0, ir::OpAttrs{}}};
      });
  auto before = rejected.node_count();
  eggc::run(rejected, std::vector{bad_rule});
  assert(rejected.node_count() == before);
  assert(inference_calls == 0);
  // Branches with conflicting shape bindings do not contaminate siblings.
  using B = eggc::NoAnalysis<ir::OpNode>;
  using G = eggc::EGraph<ir::OpNode, B>;
  G branches;
  auto bad = branches.add(ir::OpNode::input("bad")),
       good = branches.add(ir::OpNode::input("good"));
  auto bad_copy = branches.add(ir::OpNode::make(Op::Copy, {}, {bad}));
  auto good_copy = branches.add(ir::OpNode::make(Op::Copy, {}, {good}));
  branches.merge(bad_copy, good_copy);
  auto root =
      branches.add(ir::OpNode::make(Op::Pair, {}, {bad_copy, good_copy}));
  branches.rebuild();
  auto metadata = [bad](const G& g,
                        eggc::Id id) -> std::optional<ir::TensorInfo> {
    return ir::TensorInfo{
        {g.find(id) == g.find(bad) ? std::uint64_t(5) : std::uint64_t(2)},
        ir::DType::F32};
  };
  auto checks = r::rule_branch::match_checks<B>(metadata);
  auto matches = p::matches_at_with_checks(
      branches, root, r::rule_branch::pattern(), checks, metadata);
  assert(matches.size() == 1);
  assert(matches.front().tensors.at(0) == good);
  // Sequence bindings support empty sequences anywhere in a restriction.
  p::ShapeBindings bindings;
  assert(bindings.check(std::vector<std::uint64_t>{2, 3},
                        {p::ShapePart::dimension(0), p::ShapePart::sequence(1),
                         p::ShapePart::dimension(2)}));
  assert(bindings.sequence(1)->empty());
  assert(bindings.dimension(2) == 3);
  // Contextual literals use the root dtype but must remain scalar.
  auto scalar = graph({});
  auto scalar_root = scalar.add(ir::OpNode::make(Op::Copy, {}, {0}));
  eggc::run(scalar, std::vector{r::rule_scalar::build_rewrite()});
  auto literal = scalar.lookup(ir::OpNode::literal("1", ir::DType::F32));
  assert(literal && scalar.find(*literal) == scalar.find(scalar_root));
  auto nonscalar = graph();
  nonscalar.add(ir::OpNode::make(Op::Copy, {}, {0}));
  before = nonscalar.node_count();
  eggc::run(nonscalar, std::vector{r::rule_scalar::build_rewrite()});
  assert(nonscalar.node_count() == before);
  // Hashing includes metadata even when children and operation agree.
  auto n1 = ir::OpNode::make(Op::Tag, Params{1}, {0}),
       n2 = ir::OpNode::make(Op::Tag, Params{2}, {0});
  assert(n1 != n2);
  assert(n1.hash() != n2.hash());
  // Several queued actions mutate the graph before later rematches. Nested
  // copies canonicalize captures as earlier applications union classes.
  auto queued = graph();
  auto inner = queued.add(ir::OpNode::make(Op::Copy, {}, {0}));
  auto outer = queued.add(ir::OpNode::make(Op::Copy, {}, {inner}));
  auto second = queued.add(ir::OpNode::input("unbound"));
  queued.add(ir::OpNode::make(Op::Copy, {}, {second}));
  auto report =
      eggc::run(queued, std::vector{r::rule_eliminate::build_rewrite()});
  assert(report.reason == eggc::StopReason::Saturated);
  assert(queued.find(outer) == queued.find(0));
  assert(queued.find(inner) == queued.find(0));
  assert(!queued.analysis_data(second).info());
  auto floated = graph();
  auto float_root = floated.add(ir::OpNode::make(Op::Copy, {}, {0}));
  eggc::run(floated,
            std::vector{r::rule_float_literals::build_rewrite(FloatHost{})});
  assert(floated.find(float_root) == floated.find(0));
  grouped_application_test(1000, false);
  grouped_application_test(128, true);
  // Retained structural witnesses must still pass current shape restrictions
  // before a later application can invoke its host callback.
  grouped_application_test(128, true, true);
}
