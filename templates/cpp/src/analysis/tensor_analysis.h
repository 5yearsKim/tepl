#pragma once
#include <eggc/egraph.hpp>

#include "bindings.h"
#include "dtype.h"
#include "shape.h"
#include "tensor_analysis_data.h"

namespace @TEPL_NAMESPACE@::analysis {
inline Inference<TensorInfo> infer_tensor(
    Op op, ::std::span<const TensorInfo> operands, const OpAttrs& attrs) {
  ::std::vector<::std::vector<::std::uint64_t>> shapes;
  ::std::vector<DType> dtypes;
  for (const auto& operand : operands) {
    shapes.push_back(operand.shape);
    dtypes.push_back(operand.dtype);
  }
  auto shape = infer_shape(op, shapes, attrs);
  auto dtype = infer_dtype(op, dtypes, attrs);
  if (shape.kind == Inference<::std::vector<::std::uint64_t>>::Kind::Invalid)
    return Inference<TensorInfo>::invalid(shape.error);
  if (dtype.kind == Inference<DType>::Kind::Invalid)
    return Inference<TensorInfo>::invalid(dtype.error);
  if (!shape.value || !dtype.value) return Inference<TensorInfo>::unknown();
  return Inference<TensorInfo>::known({*shape.value, *dtype.value});
}
struct TensorAnalysis {
  using Data = TensorAnalysisData;
  TensorBindingTable symbols;
  TensorAnalysis() = default;
  explicit TensorAnalysis(TensorBindingTable inputs)
      : symbols(::std::move(inputs)) {}
  Data make(const ::eggc::EGraph<OpNode, TensorAnalysis>& graph,
            const OpNode& node) const {
    if (node.op() == Op::Input) {
      auto info = symbols.info(::std::get<InputAttrs>(node.attrs().value).name);
      return Data::from_inference(info ? Inference<TensorInfo>::known(*info)
                                       : Inference<TensorInfo>::unknown());
    }
    ::std::vector<TensorInfo> operands;
    bool unknown = false;
    for (auto id : node.children()) {
      auto facts = graph.analysis_data(graph.find(id));
      if (facts.is_invalid())
        return Data::from_inference(
            Inference<TensorInfo>::invalid("invalid operand metadata"));
      if (auto info = facts.info())
        operands.push_back(*info);
      else
        unknown = true;
    }
    return Data::from_inference(
        unknown ? Inference<TensorInfo>::unknown()
                : infer_tensor(node.op(), operands, node.attrs()));
  }
  ::eggc::AnalysisMerge merge(Data& into, const Data& from) const {
    return into.merge(from);
  }
};
inline ::std::optional<TensorInfo> tensor_info(
    const ::eggc::EGraph<OpNode, TensorAnalysis>& graph, ::eggc::Id id) {
  return graph.analysis_data(graph.find(id)).info();
}
struct TensorOutputInference {
  ::std::optional<DType> infer_literal(::std::string_view,
                                       ::std::optional<DType> dtype,
                                       const TensorInfo& expected) const {
    return dtype ? dtype : ::std::optional(expected.dtype);
  }
  ::std::optional<TensorInfo> infer_output(
      Op op, ::std::span<const TensorInfo> operands,
      const OpAttrs& attrs) const {
    return infer_tensor(op, operands, attrs).into_option();
  }
};
}  // namespace @TEPL_NAMESPACE@::analysis
