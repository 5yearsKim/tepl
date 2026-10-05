#pragma once
#include <eggc/egraph.hpp>

#include "pattern.h"
#include "shape.h"

namespace @TEPL_NAMESPACE@::pattern {
struct DTypeConstraint {
  enum class Kind { Exact, Bind };
  Kind kind;
  DType value = DType::Bool;
  ::std::size_t variable = 0;
  static DTypeConstraint exact(DType dtype) { return {Kind::Exact, dtype, 0}; }
  static DTypeConstraint bind(::std::size_t id) {
    return {Kind::Bind, DType::Bool, id};
  }
  bool check(DType actual, MetadataBindings& bindings) const {
    return kind == Kind::Exact ? value == actual
                               : bindings.bind_dtype(variable, actual);
  }
};
struct TensorConstraint {
  ::std::optional<DTypeConstraint> dtype;
  ::std::vector<ShapePart> shape;
};
struct TensorConstraints {
  ::std::vector<::std::pair<BindingId, TensorConstraint>> entries;
  void validate(const TensorPattern& pattern) const {
    ::std::set<BindingId> tensors, attrs;
    pattern.bindings(tensors, attrs);
    ::std::map<::std::size_t, bool> symbols;
    for (const auto& [id, restriction] : entries) {
      if (!tensors.contains(id))
        throw ::std::invalid_argument(
            "constraint refers to unavailable tensor");
      unsigned sequences = 0;
      for (const auto& part : restriction.shape) {
        bool sequence = part.kind == ShapePart::Kind::Sequence;
        if (sequence) ++sequences;
        if (part.kind == ShapePart::Kind::Dimension && !part.symbol)
          throw ::std::invalid_argument("dimension requires a symbol");
        if (part.symbol) {
          auto [it, added] = symbols.emplace(*part.symbol, sequence);
          if (!added && it->second != sequence)
            throw ::std::invalid_argument("inconsistent shape symbol");
        }
      }
      if (sequences > 1)
        throw ::std::invalid_argument("shape has multiple sequences");
    }
  }
  template <class A, class M>
  bool check_capture(const ::eggc::EGraph<OpNode, A>& graph, BindingId capture,
                     ::eggc::Id id, const M& metadata,
                     MetadataBindings& shapes) const {
    ::std::optional<TensorInfo> info;
    bool fetched = false;
    for (const auto& [target, restriction] : entries) {
      if (target != capture) continue;
      if (!fetched) {
        info = metadata(graph, graph.find(id));
        fetched = true;
      }
      if (!info ||
          (restriction.dtype &&
           !restriction.dtype->check(info->dtype, shapes)) ||
          !shapes.check(info->shape, restriction.shape))
        return false;
    }
    return true;
  }
};
}  // namespace @TEPL_NAMESPACE@::pattern
