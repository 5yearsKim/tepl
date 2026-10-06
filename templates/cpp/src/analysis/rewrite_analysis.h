#pragma once
#include <eggc/egraph.hpp>

#include "../op_node.h"
#include "tensor.h"

namespace @TEPL_NAMESPACE@::analysis {
// Custom analyses implement these operations once, rather than per rule.
template <class A>
struct RewriteAnalysis {
  static constexpr bool has_tensor_info = A::has_tensor_info;
  static ::std::optional<TensorInfo> tensor_info(
      const ::eggc::EGraph<OpNode, A>& graph, ::eggc::Id id) {
    return A::tensor_info(graph, id);
  }
  static ::std::optional<TensorInfo> infer_output(
      const ::eggc::EGraph<OpNode, A>& graph, Op op,
      ::std::span<const TensorInfo> operands, const OpAttrs& attrs) {
    return A::infer_output(graph, op, operands, attrs);
  }
  static ::std::optional<DType> infer_literal(
      const ::eggc::EGraph<OpNode, A>& graph, ::std::string_view value,
      ::std::optional<DType> dtype, const TensorInfo& expected) {
    if constexpr (requires { A::infer_literal(graph, value, dtype, expected); })
      return A::infer_literal(graph, value, dtype, expected);
    else
      return dtype ? dtype : ::std::optional(expected.dtype);
  }
};
template <>
struct RewriteAnalysis<::eggc::NoAnalysis<OpNode>> {
  static constexpr bool has_tensor_info = false;
  static ::std::optional<TensorInfo> tensor_info(
      const ::eggc::EGraph<OpNode, ::eggc::NoAnalysis<OpNode>>&, ::eggc::Id) {
    return {};
  }
  static ::std::optional<TensorInfo> infer_output(
      const ::eggc::EGraph<OpNode, ::eggc::NoAnalysis<OpNode>>&, Op,
      ::std::span<const TensorInfo>, const OpAttrs&) {
    return {};
  }
  static ::std::optional<DType> infer_literal(
      const ::eggc::EGraph<OpNode, ::eggc::NoAnalysis<OpNode>>&,
      ::std::string_view, ::std::optional<DType> dtype, const TensorInfo&) {
    return dtype;
  }
};
}  // namespace @TEPL_NAMESPACE@::analysis
