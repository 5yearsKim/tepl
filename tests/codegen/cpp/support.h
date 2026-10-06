#pragma once
#include <eggc/all.hpp>
#include <functional>
#include <memory>

#include "ir/generated.h"

namespace support {
using Info = tepl_generated::TensorInfo;
struct TestAnalysis {
  struct Policy;
  using Data = std::shared_ptr<Policy>;
  using Graph = eggc::EGraph<tepl_generated::OpNode, TestAnalysis>;
  struct Policy {
    std::function<std::optional<Info>(const Graph&, eggc::Id)> reader;
    std::function<std::optional<Info>(tepl_generated::Op, std::span<const Info>,
                                      const tepl_generated::OpAttrs&)>
        output;
    std::function<std::optional<tepl_generated::DType>(
        std::string_view, std::optional<tepl_generated::DType>, const Info&)>
        literal;
  };
  Data policy = std::make_shared<Policy>();
  static constexpr bool has_tensor_info = true;
  Data make(const Graph&, const tepl_generated::OpNode&) const {
    return policy;
  }
  eggc::AnalysisMerge merge(Data&, const Data&) const { return {}; }
  static Data state(const Graph& graph) {
    return graph.analysis_data(graph.classes().front());
  }
  static std::optional<Info> tensor_info(const Graph& graph, eggc::Id id) {
    const auto& policy = graph.analysis_data(graph.find(id));
    return policy->reader ? policy->reader(graph, id) : std::nullopt;
  }
  static std::optional<Info> infer_output(
      const Graph& graph, tepl_generated::Op op, std::span<const Info> operands,
      const tepl_generated::OpAttrs& attrs) {
    auto policy = state(graph);
    return policy->output ? policy->output(op, operands, attrs) : std::nullopt;
  }
  static std::optional<tepl_generated::DType> infer_literal(
      const Graph& graph, std::string_view value,
      std::optional<tepl_generated::DType> dtype, const Info& expected) {
    auto policy = state(graph);
    return policy->literal ? policy->literal(value, dtype, expected)
           : dtype         ? dtype
                           : std::optional(expected.dtype);
  }
};
template <class M>
void configure_reader(TestAnalysis::Graph& graph, M metadata) {
  if (!graph.is_clean()) graph.rebuild();
  TestAnalysis::state(graph)->reader = std::move(metadata);
}
template <class M, class I, class R>
R configure_rule(TestAnalysis::Graph& graph, M metadata, I inference, R rule) {
  configure_reader(graph, std::move(metadata));
  auto policy = TestAnalysis::state(graph);
  auto shared = std::make_shared<I>(std::move(inference));
  policy->output = [shared](auto op, auto operands, const auto& attrs) {
    if constexpr (requires { shared->infer_output(op, operands, attrs); })
      return shared->infer_output(op, operands, attrs);
    else
      return (*shared)(op, operands, attrs);
  };
  policy->literal = [shared](std::string_view value, auto dtype,
                             const Info& expected) {
    if constexpr (requires { shared->infer_literal(value, dtype, expected); })
      return shared->infer_literal(value, dtype, expected);
    else
      return dtype ? dtype : std::optional(expected.dtype);
  };
  return rule;
}
template <class M, class R>
R configure_checks(TestAnalysis::Graph& graph, M metadata, R checks) {
  configure_reader(graph, std::move(metadata));
  return checks;
}
}  // namespace support
