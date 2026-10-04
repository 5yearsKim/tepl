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
      graph, std::vector{ir::rules::hygiene::rule_identity::build_rewrite(),
                         ir::rules::hygiene::rule_eliminate::build_rewrite()});
  assert(report.reason == eggc::StopReason::Saturated);
  assert(graph.find(root) == graph.find(x));
  // Instantiate builders as well as declarations under namespaces whose names
  // overlap the external libraries used by the generated runtime.
  auto eggc_rule = app::eggc::rules::hygiene::rule_eliminate::build_rewrite();
  auto std_rule = app::std::rules::hygiene::rule_eliminate::build_rewrite();
  assert(eggc_rule.custom_search && std_rule.custom_search);
}
