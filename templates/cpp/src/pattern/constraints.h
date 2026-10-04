#pragma once
#include <eggc/egraph.hpp>

#include "pattern.h"
#include "shape.h"

namespace @TEPL_NAMESPACE@::pattern {
struct TensorConstraint {
  ::std::optional<DType> dtype;
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
                     ShapeBindings& shapes) const {
    ::std::optional<TensorInfo> info;
    bool fetched = false;
    for (const auto& [target, restriction] : entries) {
      if (target != capture) continue;
      if (!fetched) {
        info = metadata(graph, graph.find(id));
        fetched = true;
      }
      if (!info || (restriction.dtype && info->dtype != *restriction.dtype) ||
          !shapes.check(info->shape, restriction.shape))
        return false;
    }
    return true;
  }
};
}  // namespace @TEPL_NAMESPACE@::pattern
