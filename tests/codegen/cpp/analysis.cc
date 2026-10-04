#include <cassert>
#include <eggc/all.hpp>
#include <limits>

#include "ir/generated.h"

namespace ir = tepl_generated;
using ir::DType;
using ir::TensorInfo;
using Op = ir::dialects::unfamiliar::Op;
using Attrs = ir::dialects::unfamiliar::OpAttrs;
using A = ir::analysis::TensorAnalysis;
using Graph = eggc::EGraph<ir::OpNode, A>;
using Inference = ir::analysis::Inference<TensorInfo>;
Inference tensor(Op op, std::initializer_list<TensorInfo> inputs,
                 Attrs attrs = {}) {
  return ir::analysis::infer_tensor(
      op, std::span<const TensorInfo>(inputs.begin(), inputs.size()),
      ir::OpAttrs(attrs));
}
int main() {
  TensorInfo value{{2, 3}, DType::I32};
  assert(tensor(Op::Fuse, {value, value}).value == value);
  for (auto op : {Op::Add, Op::ShapeOnly, Op::DtypeOnly})
    assert(tensor(op, {value}).kind == Inference::Kind::Unknown);
  assert(tensor(Op::Floats, {value}).kind == Inference::Kind::Invalid);
  assert(
      (tensor(Op::Boolean, {value}).value == TensorInfo{{2, 3}, DType::Bool}));
  assert((tensor(Op::Tail, {value, {{4, 5}, DType::I32}}).value ==
          TensorInfo{{2, 3, 9}, DType::I32}));
  assert(tensor(Op::Variadic, {}).kind == Inference::Kind::Unknown);
  assert((tensor(Op::Variadic, {value, {{4}, DType::I32}}).value ==
          TensorInfo{{5, 4}, DType::I32}));
  ir::dialects::unfamiliar::UtilitiesAttrs utilities{
      {true, true}, {{true, false}}, {{-2, 1}}, 2, true};
  assert(tensor(Op::Utilities, {value}, utilities).value == value);
  utilities.flags = {false};
  assert(tensor(Op::Utilities, {value}, utilities).kind ==
         Inference::Kind::Invalid);
  assert(
      (tensor(Op::Branches, {value}).value == TensorInfo{{2, 4}, DType::I32}));
  assert(tensor(Op::Overflow, {}).kind == Inference::Kind::Invalid);
  assert((tensor(Op::Minimum, {}).value == TensorInfo{{}, DType::F32}));
  assert(tensor(Op::Stopped, {value}).error.find("shape assertion") !=
         std::string::npos);
  assert(
      tensor(Op::Output, {value}, ir::dialects::unfamiliar::OutputAttrs{{-1}})
          .kind == Inference::Kind::Invalid);
  auto facts =
      ir::analysis::TensorAnalysisData::from_inference(Inference::known(value));
  facts.merge(
      ir::analysis::TensorAnalysisData::from_inference(Inference::unknown()));
  assert(!facts.info());
  facts.merge(ir::analysis::TensorAnalysisData::from_inference(
      Inference::known({{3}, DType::I32})));
  assert(facts.is_invalid());
  // Check the evidence join's algebra across known, unknown, and invalid facts.
  std::vector<ir::analysis::TensorAnalysisData> states{
      ir::analysis::TensorAnalysisData::from_inference(Inference::known(value)),
      ir::analysis::TensorAnalysisData::from_inference(
          Inference::known({{3}, DType::I32})),
      ir::analysis::TensorAnalysisData::from_inference(Inference::unknown()),
      ir::analysis::TensorAnalysisData::from_inference(
          Inference::invalid("bad"))};
  for (auto a : states)
    for (auto b : states)
      for (auto c : states) {
        auto ab = a;
        ab.merge(b);
        auto ba = b;
        ba.merge(a);
        assert(ab == ba);
        auto left = ab;
        left.merge(c);
        auto bc = b;
        bc.merge(c);
        auto right = a;
        right.merge(bc);
        assert(left == right);
        auto aa = a;
        aa.merge(a);
        assert(aa == a);
      }
  for (auto shape :
       {std::vector<std::uint64_t>{2}, std::vector<std::uint64_t>{2, 3}}) {
    ir::analysis::TensorBindingTable inputs;
    inputs.insert("x", {shape, DType::F32});
    inputs.insert("y", {shape, DType::F32});
    Graph graph{A(inputs)};
    auto x = graph.add(ir::OpNode::input("x")),
         y = graph.add(ir::OpNode::input("y"));
    auto root = graph.add(ir::OpNode::make(Op::Fuse, {}, {x, y}));
    auto rule = ir::rules::analysis::rule_commute::build_rewrite();
    auto report = eggc::run(graph, std::vector{rule});
    auto swapped = graph.lookup(ir::OpNode::make(Op::Fuse, {}, {y, x}));
    assert(swapped.has_value() == (shape.size() == 1));
    if (swapped) assert(graph.find(root) == graph.find(*swapped));
    assert(report.reason == eggc::StopReason::Saturated);
    assert((eggc::Extractor<ir::OpNode, A>(graph).find_best(root).first == 3));
    auto before = graph.node_count();
    eggc::run(
        graph,
        std::vector{ir::rules::analysis::rule_bad_output::build_rewrite()});
    assert(graph.node_count() == before);
  }
  // Unions retain invalid evidence and propagate it to parents.
  ir::analysis::TensorBindingTable inputs;
  inputs.insert("x", {{2}, DType::F32});
  inputs.insert("y", {{3}, DType::F32});
  Graph graph{A(inputs)};
  auto x = graph.add(ir::OpNode::input("x")),
       y = graph.add(ir::OpNode::input("y"));
  auto parent = graph.add(ir::OpNode::make(Op::Floats, {}, {x}));
  graph.merge(x, y);
  graph.rebuild();
  assert(graph.analysis_data(x).is_invalid());
  assert(graph.analysis_data(parent).is_invalid());
  inputs.register_symbol("x", {{2}, DType::F32});
  bool conflict = false;
  try {
    inputs.register_symbol("x", {{3}, DType::F32});
  } catch (const std::invalid_argument&) {
    conflict = true;
  }
  assert(conflict);
}
