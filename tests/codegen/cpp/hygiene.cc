#include <cassert>
#include <eggc/runner.hpp>

#include "ir/generated.h"
#include "nested_eggc/generated.h"
#include "nested_std/generated.h"

namespace ir = tepl_generated;
using Op = ir::dialects::hygiene::Op;
using Value = ir::dialects::hygiene::Value;
using Arity = ir::dialects::hygiene::Arity;
using Attrs = ir::dialects::hygiene::OpAttrs;

int main() {
  assert(Attrs(Value{1}).schema_id() == 0);
  assert(Attrs(Arity{"one"}).schema_id() == 1);
  assert(Value{1}.hash_value() != Value{2}.hash_value());
  assert(Arity{"one"}.hash_value() != Arity{"two"}.hash_value());
  ir::analysis::TensorBindingTable inputs;
  inputs.insert("x", {{2}, ir::DType::F32});
  eggc::EGraph<ir::OpNode, ir::analysis::TensorAnalysis> graph{
      ir::analysis::TensorAnalysis(inputs)};
  auto x = graph.add(ir::OpNode::input("x"));
  auto one = graph.add(ir::OpNode::make(Op::ValueCopy, Value{1}, {x}));
  auto two = graph.add(ir::OpNode::make(Op::ValueCopy, Value{2}, {x}));
  assert(one != two);
  assert(graph.add(ir::OpNode::make(Op::ValueCopy, Value{1}, {x})) == one);
  auto root = graph.add(ir::OpNode::make(Op::ArityCopy, Arity{"one"}, {x}));
  auto report = eggc::run(
      graph, std::vector{ir::rules::hygiene::rule_identity::build(),
                         ir::rules::hygiene::rule_eliminate::build()});
  assert(report.reason == eggc::StopReason::Saturated);
  assert(graph.find(root) == graph.find(x));
  // Instantiate builders as well as declarations under namespaces whose names
  // overlap the external libraries used by the generated runtime.
  auto eggc_rule = app::eggc::rules::hygiene::rule_eliminate::build();
  auto std_rule = app::std::rules::hygiene::rule_eliminate::build();
  assert(eggc_rule.custom_search && std_rule.custom_search);
  ir::GraphDefinition definition = ir::graphs::hygiene::graph_example::build();
  eggc::EGraph<ir::OpNode, ir::TensorAnalysis> built_graph{
      ir::TensorAnalysis(definition.input_bindings())};
  auto built = definition.insert_nodes(built_graph);
  assert((built_graph.analysis_data(built.root).info() ==
          ir::TensorInfo{{2}, ir::DType::F32}));
  auto eggc_definition = app::eggc::graphs::hygiene::graph_example::build();
  eggc::EGraph<app::eggc::OpNode, app::eggc::TensorAnalysis> eggc_graph{
      app::eggc::TensorAnalysis(eggc_definition.input_bindings())};
  auto eggc_built = eggc_definition.insert_nodes(eggc_graph);
  assert((eggc_graph.analysis_data(eggc_built.root).info() ==
          app::eggc::TensorInfo{{2}, app::eggc::DType::F32}));
  auto std_definition = app::std::graphs::hygiene::graph_example::build();
  eggc::EGraph<app::std::OpNode, app::std::TensorAnalysis> std_graph{
      app::std::TensorAnalysis(std_definition.input_bindings())};
  auto std_built = std_definition.insert_nodes(std_graph);
  assert((std_graph.analysis_data(std_built.root).info() ==
          app::std::TensorInfo{{2}, app::std::DType::F32}));
}
