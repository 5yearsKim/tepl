#include <cassert>
#include <eggc/all.hpp>

#include "ir/generated.h"
int main() {
  namespace ir = tepl_generated;
  static_assert(std::same_as<ir::types::Elements, ir::Elements>);
  static_assert(std::same_as<ir::analysis::TensorInfo, ir::TensorInfo>);
  static_assert(std::same_as<ir::Inference<int>, ir::analysis::Inference<int>>);
  static_assert(std::same_as<ir::TensorAnalysis, ir::analysis::TensorAnalysis>);
  static_assert(
      std::same_as<ir::TensorAnalysisData, ir::analysis::TensorAnalysisData>);
  static_assert(
      std::same_as<ir::TensorBindingTable, ir::analysis::TensorBindingTable>);
  static_assert(std::same_as<ir::GraphDefinition, ir::graphs::GraphDefinition>);
  static_assert(std::same_as<ir::BuiltGraph, ir::graphs::BuiltGraph>);
  static_assert(std::same_as<ir::InputSpec, ir::graphs::InputSpec>);
  static_assert(
      std::same_as<ir::RewriteAnalysis<ir::TensorAnalysis>,
                   ir::rewriting::RewriteAnalysis<ir::TensorAnalysis>>);
  ir::dialects::only::Payload payload{
      2,
      {{{1, 2, 3}}},
      ir::ReplicaGroups{std::vector<std::vector<std::int64_t>>{{0, 1}}},
      ir::Elements{"f32", {2}, {3, 4}}};
  auto a = ir::OpNode::make(ir::dialects::only::Op::Leaf, payload, {});
  auto copy = a;
  assert(a == copy && a.hash() == copy.hash());
  payload.regions[0].bytes.push_back(4);
  auto b = ir::OpNode::make(ir::dialects::only::Op::Leaf, payload, {});
  assert(a != b);
  eggc::EGraph<ir::OpNode> graph;
  auto id = graph.add(a);
  graph.rebuild();
  assert(graph.lookup(copy) == id);
}
