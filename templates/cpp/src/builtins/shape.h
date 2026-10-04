#pragma once

#include "common.h"

namespace @TEPL_NAMESPACE@::builtins::shape {
template <class V>
auto integers(const V& values) {
  ::std::vector<Integer> result;
  result.reserve(values.size());
  for (auto value : values) result.push_back(static_cast<Integer>(value));
  return result;
}
// Integer and bool attributes may be arbitrarily nested lists.
template <class T>
auto attribute(const T& value) {
  if constexpr (::std::is_same_v<T, bool>)
    return value;
  else if constexpr (requires { typename T::value_type; }) {
    using U = decltype(attribute(typename T::value_type{}));
    ::std::vector<U> result;
    for (const typename T::value_type& item : value)
      result.push_back(attribute(item));
    return result;
  } else
    return static_cast<Integer>(value);
}
inline BuiltinResult<::std::vector<::std::uint64_t>> dimensions(
    const ::std::vector<Integer>& values) {
  return attempt([&] {
    ::std::vector<::std::uint64_t> result;
    for (auto v : values) {
      if (v < 0 || static_cast<unsigned __int128>(v) >
                       ::std::numeric_limits<::std::uint64_t>::max())
        throw BuiltinError("invalid dimension");
      result.push_back(static_cast<::std::uint64_t>(v));
    }
    return result;
  });
}
template <class V, class I>
bool is_valid_axis_list(const V& axes, I rank) {
  try {
    auto r = common::position(rank);
    ::std::vector<typename V::value_type> seen;
    for (auto axis : axes) {
      if (common::position(axis) >= r || common::contains(seen, axis))
        return false;
      seen.push_back(axis);
    }
    return true;
  } catch (const BuiltinError&) {
    return false;
  }
}
template <class V>
BuiltinResult<V> broadcast_shape(const V& a, const V& b) {
  return attempt([&] {
    using T = typename V::value_type;
    for (T v : a)
      if (v < 0) throw BuiltinError("invalid dimension");
    for (T v : b)
      if (v < 0) throw BuiltinError("invalid dimension");
    auto rank = ::std::max(a.size(), b.size());
    V result(rank);
    for (::std::size_t i = 0; i < rank; ++i) {
      T l = i < a.size() ? a[a.size() - i - 1] : T(1),
        r = i < b.size() ? b[b.size() - i - 1] : T(1);
      if (l == r || r == 1)
        result[rank - i - 1] = l;
      else if (l == 1)
        result[rank - i - 1] = r;
      else
        throw BuiltinError("incompatible broadcast");
    }
    return result;
  });
}
}  // namespace @TEPL_NAMESPACE@::builtins::shape
