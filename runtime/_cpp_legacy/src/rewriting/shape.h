#pragma once
#include <map>

#include "../analysis/tensor_info.h"

namespace @TEPL_NAMESPACE@::rewriting {
struct ShapePart {
  enum class Kind { Dimension, Wildcard, Sequence };
  Kind kind;
  ::std::optional<::std::size_t> symbol;
  static ShapePart dimension(::std::size_t id) { return {Kind::Dimension, id}; }
  static ShapePart wildcard() { return {Kind::Wildcard, {}}; }
  static ShapePart sequence(::std::optional<::std::size_t> id = {}) {
    return {Kind::Sequence, id};
  }
};
struct MetadataBindings {
  ::std::map<::std::size_t, DType> dtypes;
  bool bind_dtype(::std::size_t id, DType value) {
    auto [found, added] = dtypes.emplace(id, value);
    return added || found->second == value;
  }
  ::std::optional<DType> dtype(::std::size_t id) const {
    auto i = dtypes.find(id);
    return i == dtypes.end() ? ::std::nullopt : ::std::optional(i->second);
  }
  ::std::map<::std::size_t, ::std::uint64_t> dimensions;
  ::std::map<::std::size_t, ::std::vector<::std::uint64_t>> sequences;
  ::std::optional<::std::uint64_t> dimension(::std::size_t id) const {
    auto i = dimensions.find(id);
    return i == dimensions.end() ? ::std::nullopt : ::std::optional(i->second);
  }
  ::std::optional<::std::vector<::std::uint64_t>> sequence(
      ::std::size_t id) const {
    auto i = sequences.find(id);
    return i == sequences.end() ? ::std::nullopt : ::std::optional(i->second);
  }
  // A failed check may partially bind symbols; discard the branch on failure.
  bool check(::std::span<const ::std::uint64_t> shape,
             const ::std::vector<ShapePart>& parts) {
    ::std::optional<::std::size_t> split;
    for (::std::size_t i = 0; i < parts.size(); ++i)
      if (parts[i].kind == ShapePart::Kind::Sequence) {
        if (split) return false;
        split = i;
      }
    auto fixed = parts.size() - split.has_value();
    if (shape.size() < fixed || (!split && shape.size() != fixed)) return false;
    for (::std::size_t i = 0; i < parts.size(); ++i) {
      const auto& p = parts[i];
      if (p.kind == ShapePart::Kind::Dimension) {
        if (!p.symbol) return false;
        auto axis = split && i > *split ? shape.size() - (parts.size() - i) : i;
        auto [found, added] = dimensions.emplace(*p.symbol, shape[axis]);
        if (!added && found->second != shape[axis]) return false;
      } else if (p.kind == ShapePart::Kind::Sequence && p.symbol) {
        auto last = shape.size() - (parts.size() - i - 1);
        ::std::vector<::std::uint64_t> values(shape.begin() + i,
                                              shape.begin() + last);
        auto [found, added] = sequences.emplace(*p.symbol, values);
        if (!added && found->second != values) return false;
      }
    }
    return true;
  }
};
}  // namespace @TEPL_NAMESPACE@::rewriting
