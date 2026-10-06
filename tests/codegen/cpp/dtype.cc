#include <cassert>
#include <eggc/all.hpp>

#include "ir/generated.h"
#include "support.h"

namespace ir = tepl_generated;
namespace d = ir::dialects::d_types;
namespace r = ir::rules::dtype;
using ir::DType;
using A = ir::analysis::TensorAnalysis;
using Graph = eggc::EGraph<ir::OpNode, A>;
using Inference = ir::analysis::Inference<DType>;

Inference infer(d::Op op, std::initializer_list<DType> inputs,
                ir::OpAttrs attrs = {}) {
  return ir::analysis::infer_dtype(
      op, std::span<const DType>(inputs.begin(), inputs.size()), attrs);
}
ir::TensorInfo info(DType dtype) { return {{4}, dtype}; }
struct Host {
  std::shared_ptr<unsigned> calls;
  std::optional<bool> allow_dtype(DType dtype) const {
    ++*calls;
    return dtype == DType::F32;
  }
};
int main() {
  assert(infer(d::Op::Fixed, {}).value == DType::F32);
  assert(infer(d::Op::Unknown, {DType::F32}).kind == Inference::Kind::Unknown);
  assert(infer(d::Op::Equal, {DType::I32, DType::I32}).value == DType::Bool);
  assert(infer(d::Op::Equal, {DType::Bool, DType::Bool}).kind ==
         Inference::Kind::Invalid);
  assert(infer(d::Op::Equal, {DType::I32, DType::F32}).kind ==
         Inference::Kind::Invalid);
  assert(infer(d::Op::Select, {DType::Bool, DType::BF16, DType::BF16}).value ==
         DType::BF16);
  assert(infer(d::Op::Select, {DType::F32, DType::BF16, DType::BF16}).kind ==
         Inference::Kind::Invalid);
  assert(infer(d::Op::Variadic, {}).kind == Inference::Kind::Invalid);
  assert(infer(d::Op::Variadic, {DType::U8, DType::U8}).value == DType::U8);
  assert(infer(d::Op::Variadic, {DType::U8, DType::I8}).kind ==
         Inference::Kind::Invalid);
  assert(infer(d::Op::Tail, {DType::F16}).value == DType::F16);
  for (bool widen : {false, true}) {
    auto attrs = ir::OpAttrs(d::ChoicesAttrs{DType::F16, {}, widen});
    assert(infer(d::Op::Choices, {DType::BF16}, attrs).value ==
           (widen ? DType::F32 : DType::F16));
  }
  for (DType input :
       {DType::Bool, DType::I8, DType::U8, DType::I32, DType::U32, DType::I64,
        DType::U64, DType::F16, DType::BF16, DType::F32, DType::F64}) {
    for (DType to : {DType::Bool, DType::I64, DType::BF16, DType::F32}) {
      auto result =
          infer(d::Op::Convert, {input}, ir::OpAttrs(d::ConvertAttrs{to}));
      if (input == DType::Bool || to == DType::Bool)
        assert(result.kind == Inference::Kind::Invalid);
      else
        assert(result.value == to);
    }
  }
  assert(infer(d::Op::Convert, {DType::F32}).kind == Inference::Kind::Invalid);
  for (DType input : {DType::BF16, DType::F32, DType::I64}) {
    for (DType to : {DType::BF16, DType::F32, DType::I64}) {
      ir::analysis::TensorBindingTable inputs;
      inputs.insert("x", info(input));
      Graph graph{A(inputs)};
      auto x = graph.add(ir::OpNode::input("x"));
      auto root =
          graph.add(ir::OpNode::make(d::Op::Convert, d::ConvertAttrs{to}, {x}));
      graph.rebuild();
      auto checks = r::rule_remove_identity_convert::match_checks<A>();
      auto matches = ir::pattern::matches_at_with_checks(
          graph, root, r::rule_remove_identity_convert::pattern(), checks,
          ir::analysis::tensor_info);
      assert(!matches.empty() == (input == to));
      auto before = graph.node_count();
      eggc::run(graph,
                std::vector{r::rule_remove_identity_convert::build_rewrite()});
      assert((graph.find(x) == graph.find(root)) == (input == to));
      assert(graph.node_count() == before);
    }
  }
  for (DType input : {DType::F32, DType::BF16}) {
    ir::analysis::TensorBindingTable inputs;
    inputs.insert("x", info(input));
    Graph graph{A(inputs)};
    auto x = graph.add(ir::OpNode::input("x"));
    auto root = graph.add(
        ir::OpNode::make(d::Op::Convert, d::ConvertAttrs{input}, {x}));
    auto calls = std::make_shared<unsigned>(0);
    graph.rebuild();
    auto checks = r::rule_remove_f32_convert::match_checks<A>();
    auto matches = ir::pattern::matches_at_with_checks(
        graph, root, r::rule_remove_f32_convert::pattern(), checks,
        ir::analysis::tensor_info);
    assert(!matches.empty() == (input == DType::F32));
    auto host = r::rule_host_identity::build_rewrite(Host{calls});
    eggc::run(graph, std::vector{host});
    assert((graph.find(x) == graph.find(root)) == (input == DType::F32));
    assert(*calls > 0);
  }
  using NA = support::TestAnalysis;
  using BareGraph = eggc::EGraph<ir::OpNode, NA>;
  for (auto [left, right, accepted] :
       {std::tuple{DType::F32, DType::F32, true},
        std::tuple{DType::BF16, DType::BF16, true},
        std::tuple{DType::F32, DType::BF16, false},
        std::tuple{DType::I32, DType::I32, false}}) {
    BareGraph graph;
    auto x = graph.add(ir::OpNode::input("x"));
    auto y = graph.add(ir::OpNode::input("y"));
    auto root = graph.add(ir::OpNode::make(d::Op::Pair, {}, {x, y}));
    graph.rebuild();
    auto metadata = [=](const BareGraph&,
                        eggc::Id id) -> std::optional<ir::TensorInfo> {
      return info(id == y ? right : left);
    };
    auto checks = support::configure_checks(graph, metadata,
                                            r::rule_shared::match_checks<NA>());
    auto matches = ir::pattern::matches_at_with_checks(
        graph, root, r::rule_shared::pattern(), checks, metadata);
    assert(!matches.empty() == accepted);
  }
  {
    BareGraph alternatives;
    auto f32 = alternatives.add(ir::OpNode::input("f32"));
    auto bf16 = alternatives.add(ir::OpNode::input("bf16"));
    auto bad = alternatives.add(ir::OpNode::make(d::Op::Pair, {}, {f32, bf16}));
    auto good =
        alternatives.add(ir::OpNode::make(d::Op::Pair, {}, {bf16, bf16}));
    alternatives.merge(bad, good);
    alternatives.rebuild();
    auto metadata = [=](const BareGraph&,
                        eggc::Id id) -> std::optional<ir::TensorInfo> {
      return info(id == f32 ? DType::F32 : DType::BF16);
    };
    auto checks = support::configure_checks(alternatives, metadata,
                                            r::rule_shared::match_checks<NA>());
    auto matches = ir::pattern::matches_at_with_checks(
        alternatives, good, r::rule_shared::pattern(), checks, metadata);
    assert(matches.size() == 1);
    for (auto [id, captured] : matches.front().tensors)
      assert(captured == bf16);
  }
  {
    BareGraph pruning;
    auto x = pruning.add(ir::OpNode::input("x"));
    auto y = pruning.add(ir::OpNode::input("y"));
    auto root = pruning.add(ir::OpNode::make(d::Op::Pair, {}, {x, y}));
    pruning.rebuild();
    unsigned reads = 0;
    auto metadata = [&](const BareGraph&,
                        eggc::Id id) -> std::optional<ir::TensorInfo> {
      ++reads;
      assert(id == x);
      return info(DType::I32);
    };
    auto checks = support::configure_checks(pruning, metadata,
                                            r::rule_shared::match_checks<NA>());
    assert(ir::pattern::matches_at_with_checks(
               pruning, root, r::rule_shared::pattern(), checks, metadata)
               .empty());
    assert(reads == 1);
  }
  // Search bindings are reconstructed before application; stale facts cannot
  // authorize a conversion rewrite or mutate the graph.
  BareGraph graph;
  auto x = graph.add(ir::OpNode::input("x"));
  auto root = graph.add(
      ir::OpNode::make(d::Op::Convert, d::ConvertAttrs{DType::F32}, {x}));
  graph.rebuild();
  DType current = DType::F32;
  auto metadata = [&](const BareGraph&,
                      eggc::Id id) -> std::optional<ir::TensorInfo> {
    return info(id == x ? current : DType::F32);
  };
  auto rule = support::configure_rule(
      graph, metadata, ir::analysis::TensorOutputInference{},
      r::rule_remove_identity_convert::build_rewrite<NA>());
  std::vector<eggc::Application<ir::OpNode, NA>> actions;
  rule.custom_search(graph,
                     [&](auto action) {
                       actions.push_back(std::move(action));
                       return true;
                     },
                     {});
  assert(!actions.empty());
  current = DType::BF16;
  auto before = graph.node_count();
  for (auto& action : actions) assert(!action.apply(graph));
  assert(graph.find(x) != graph.find(root));
  assert(graph.node_count() == before);
}
