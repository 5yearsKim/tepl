#pragma once
#include "../builtins/error.h"
#include "constraints.h"
#include "context.h"

namespace @TEPL_NAMESPACE@::rewriting {
struct MatchBinding {
  enum class Kind { Tensor, Attribute, DType, Dimension, Sequence };
  Kind kind;
  BindingId id;
  bool bound(const TensorMatch& matched, const MetadataBindings& shapes) const {
    switch (kind) {
      case Kind::Tensor:
        return matched.tensors.contains(id);
      case Kind::Attribute:
        return matched.attrs.contains(id);
      case Kind::DType:
        return shapes.dtypes.contains(id);
      case Kind::Dimension:
        return shapes.dimensions.contains(id);
      case Kind::Sequence:
        return shapes.sequences.contains(id);
    }
    return false;
  }
};
template <class A>
struct MatchChecks {
  TensorConstraints tensors;
  ::std::vector<::std::vector<MatchBinding>> conditions;
  ::std::function<::std::optional<bool>(
      ::std::size_t, const ::eggc::EGraph<OpNode, A>&, const TensorMatch&,
      const MetadataBindings&)>
      evaluate;
  void validate(const TensorPattern& pattern) const {
    pattern.validate();
    tensors.validate(pattern);
    ::std::set<BindingId> tensor_ids, attrs;
    pattern.bindings(tensor_ids, attrs);
    ::std::set<::std::pair<MatchBinding::Kind, BindingId>> available;
    for (auto id : tensor_ids)
      available.emplace(MatchBinding::Kind::Tensor, id);
    for (auto id : attrs) available.emplace(MatchBinding::Kind::Attribute, id);
    for (const auto& [id, constraint] : tensors.entries) {
      if (constraint.dtype &&
          constraint.dtype->kind == DTypeConstraint::Kind::Bind)
        available.emplace(MatchBinding::Kind::DType,
                          constraint.dtype->variable);
      for (const auto& part : constraint.shape)
        if (part.symbol)
          available.emplace(part.kind == ShapePart::Kind::Sequence
                                ? MatchBinding::Kind::Sequence
                                : MatchBinding::Kind::Dimension,
                            *part.symbol);
    }
    if (!conditions.empty() && !evaluate)
      throw ::std::invalid_argument("missing condition evaluator");
    for (const auto& dependencies : conditions)
      for (auto dep : dependencies)
        if (!available.contains({dep.kind, dep.id}))
          throw ::std::invalid_argument(
              "condition refers to unavailable binding");
  }
  bool advance(const ::eggc::EGraph<OpNode, A>& graph,
               const TensorMatch& matched, const MetadataBindings& shapes,
               ::std::size_t& next) const {
    while (next < conditions.size()) {
      for (auto dep : conditions[next])
        if (!dep.bound(matched, shapes)) return true;
      try {
        auto accepted = evaluate(next, graph, matched, shapes);
        if (!accepted || !*accepted) return false;
      } catch (const builtins::BuiltinError&) {
        return false;
      }
      ++next;
    }
    return true;
  }
  template <class M>
  ::std::optional<MetadataBindings> check_match(
      const ::eggc::EGraph<OpNode, A>& graph, const TensorMatch& matched,
      const M& metadata) const {
    MetadataBindings shapes;
    for (auto [capture, id] : matched.tensors)
      if (!tensors.check_capture(graph, capture, id, metadata, shapes))
        return {};
    ::std::size_t next = 0;
    if (!advance(graph, matched, shapes, next) || next != conditions.size())
      return {};
    return shapes;
  }
};
}  // namespace @TEPL_NAMESPACE@::rewriting
