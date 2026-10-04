#include <cassert>
#include <eggc/all.hpp>
#include <map>
#include <numeric>

#include "ir/generated.h"

namespace ir = tepl_generated;
using Op = ir::dialects::tensor_lang::Op;
using Info = ir::TensorInfo;
using Infer = ir::analysis::Inference<Info>;
ir::OpAttrs dot_attrs() {
  return ir::dialects::tensor_lang::OpAttrs(
      ir::dialects::tensor_lang::DotGeneralAttrs{
          {2},
          {1},
          {0},
          {0},
          {ir::Precision::Default, ir::Precision::Default},
          {}});
}
Infer infer(ir::Op op, std::span<const Info> inputs, const ir::OpAttrs& attrs) {
  if (op != Op::DotGeneral)
    return ir::analysis::infer_tensor(op, inputs, attrs);
  std::vector<std::vector<std::uint64_t>> shapes;
  std::vector<ir::DType> dtypes;
  for (const auto& input : inputs) {
    shapes.push_back(input.shape);
    dtypes.push_back(input.dtype);
  }
  auto shape = ir::analysis::infer_shape(op, shapes, attrs);
  auto dtype = ir::builtins::dtype::same_numeric(dtypes);
  if (!shape.value || !dtype.value) return Infer::invalid("invalid dot");
  return Infer::known({*shape.value, *dtype.value});
}
struct LoraAnalysis {
  using Data = ir::analysis::TensorAnalysisData;
  ir::analysis::TensorBindingTable inputs;
  Data make(const eggc::EGraph<ir::OpNode, LoraAnalysis>& graph,
            const ir::OpNode& node) const {
    if (node.op() == ir::Op::Input) {
      auto info =
          inputs.info(std::get<ir::InputAttrs>(node.attrs().value).name);
      return Data::from_inference(info ? Infer::known(*info)
                                       : Infer::unknown());
    }
    std::vector<Info> operands;
    bool unknown = false;
    for (auto id : node.children()) {
      auto facts = graph.analysis_data(id);
      if (facts.is_invalid())
        return Data::from_inference(Infer::invalid("invalid child"));
      if (auto info = facts.info())
        operands.push_back(*info);
      else
        unknown = true;
    }
    return Data::from_inference(
        unknown ? Infer::unknown() : infer(node.op(), operands, node.attrs()));
  }
  eggc::AnalysisMerge merge(Data& into, const Data& from) const {
    return into.merge(from);
  }
};
struct Host {
  bool allow;
  std::optional<bool> is_broadcastable(std::span<const std::uint64_t> a,
                                       std::span<const std::uint64_t> b) const {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  }
  std::optional<bool> is_reassociable(const Info&, const Info&, const Info&,
                                      const ir::OpAttrs& outer,
                                      const ir::OpAttrs& inner) const {
    return allow && outer == dot_attrs() && inner == dot_attrs();
  }
  std::optional<ir::OpAttrs> infer_dot(const Info& a, const Info& b,
                                       const ir::OpAttrs& attrs) const {
    std::vector<Info> inputs{a, b};
    if (!infer(Op::DotGeneral, inputs, attrs).value) return {};
    return attrs;
  }
  std::optional<ir::OpAttrs> infer_lora_out(const Info&, const Info&,
                                            const Info&,
                                            const ir::OpAttrs& outer,
                                            const ir::OpAttrs& inner) const {
    if (outer != dot_attrs() || inner != dot_attrs()) return {};
    return inner;
  }
};
struct Output {
  std::optional<Info> infer_output(ir::Op op, std::span<const Info> operands,
                                   const ir::OpAttrs& attrs) const {
    return infer(op, operands, attrs).into_option();
  }
};
struct Value {
  std::vector<std::uint64_t> shape;
  std::vector<std::int64_t> data;
  bool operator==(const Value&) const = default;
};
std::vector<Value> evaluate(const eggc::RecExpr<ir::OpNode>& expr,
                            const std::map<std::string, Value>& inputs) {
  std::vector<Value> values;
  for (const auto& node : expr.nodes) {
    if (node.op() == ir::Op::Input) {
      values.push_back(
          inputs.at(std::get<ir::InputAttrs>(node.attrs().value).name));
      continue;
    }
    const auto& a = values.at(node.children()[0]);
    const auto& b = values.at(node.children()[1]);
    if (node.op() == Op::Add) {
      auto result = a;
      for (std::size_t i = 0; i < result.data.size(); ++i)
        result.data[i] += b.data[i];
      values.push_back(std::move(result));
    } else {
      assert(node.op() == Op::DotGeneral);
      auto m = a.shape[1], k = a.shape[2], n = b.shape[2];
      Value result{{1, m, n}, std::vector<std::int64_t>(m * n)};
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
          for (std::size_t t = 0; t < k; ++t)
            result.data[i * n + j] += a.data[i * k + t] * b.data[t * n + j];
      values.push_back(std::move(result));
    }
  }
  return values;
}
int main() {
  using G = eggc::EGraph<ir::OpNode, LoraAnalysis>;
  for (bool allowed : {false, true}) {
    LoraAnalysis analysis;
    std::map<std::string, Value> inputs;
    for (auto [name, shape] :
         std::vector<std::pair<std::string, std::vector<std::uint64_t>>>{
             {"x", {1, 4, 8}},
             {"w", {1, 8, 8}},
             {"a", {1, 8, 1}},
             {"b", {1, 1, 8}}}) {
      analysis.inputs.insert(name, {shape, ir::DType::I64});
      auto size = std::accumulate(shape.begin(), shape.end(), std::uint64_t(1),
                                  std::multiplies<>());
      std::vector<std::int64_t> data(size);
      for (std::size_t i = 0; i < size; ++i) data[i] = std::int64_t(i % 5) - 2;
      inputs.emplace(name, Value{shape, data});
    }
    G graph{analysis};
    std::vector<eggc::Id> ids;
    for (auto name : {"x", "w", "a", "b"})
      ids.push_back(graph.add(ir::OpNode::input(name)));
    auto [x, w, a, b] = std::tuple{ids[0], ids[1], ids[2], ids[3]};
    auto ab =
        graph.add(ir::OpNode::from_parts(Op::DotGeneral, {a, b}, dot_attrs()));
    auto sum = graph.add(ir::OpNode::make(Op::Add, {}, {w, ab}));
    auto root = graph.add(
        ir::OpNode::from_parts(Op::DotGeneral, {x, sum}, dot_attrs()));
    graph.rebuild();
    auto original =
        eggc::Extractor<ir::OpNode, LoraAnalysis>(graph).find_best(root).second;
    auto metadata = [](const G& g, eggc::Id id) {
      return g.analysis_data(g.find(id)).info();
    };
    auto rule = ir::rules::lora::rule_lora::build_rewrite_with<LoraAnalysis>(
        metadata, Output{}, Host{allowed});
    auto report = eggc::run(graph, std::vector{rule});
    assert(report.reason == eggc::StopReason::Saturated);
    auto xw = graph.lookup(
        ir::OpNode::from_parts(Op::DotGeneral, {x, w}, dot_attrs()));
    assert(xw.has_value() == allowed);
    if (!allowed) continue;
    auto xa = graph.lookup(
        ir::OpNode::from_parts(Op::DotGeneral, {x, a}, dot_attrs()));
    assert(xa);
    auto xab = graph.lookup(
        ir::OpNode::from_parts(Op::DotGeneral, {*xa, b}, dot_attrs()));
    assert(xab);
    auto expected = graph.lookup(ir::OpNode::make(Op::Add, {}, {*xw, *xab}));
    assert(expected && graph.find(*expected) == graph.find(root));
    auto cost = [&](const ir::OpNode& node,
                    const std::vector<std::size_t>& children)
        -> std::optional<std::size_t> {
      std::size_t total =
          std::accumulate(children.begin(), children.end(), std::size_t(1));
      if (node.op() == Op::DotGeneral) {
        auto lhs = metadata(graph, node.children()[0])->shape;
        auto rhs = metadata(graph, node.children()[1])->shape;
        total += lhs[1] * lhs[2] * rhs[2];
      } else if (node.op() == Op::Add) {
        auto shape = metadata(graph, node.children()[0])->shape;
        total += shape[1] * shape[2];
      }
      return total;
    };
    auto [best_cost, best] =
        eggc::Extractor<ir::OpNode, LoraAnalysis>(graph, cost).find_best(root);
    assert(best.nodes.back().op() == Op::Add);
    assert(best_cost < 393);
    assert(evaluate(original, inputs).back() == evaluate(best, inputs).back());
  }
}
