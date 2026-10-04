#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

#include "error.h"

#if !defined(__SIZEOF_INT128__)
#error "TEPL C++ shape arithmetic requires GCC/Clang signed 128-bit integers"
#endif
namespace @TEPL_NAMESPACE@::builtins {
using Integer = __int128;
namespace common {
template <class T>
concept CheckedInteger =
    (::std::is_integral_v<T> || ::std::is_same_v<T, Integer>) &&
    !::std::is_same_v<T, bool>;
inline BuiltinResult<bool> ensure(bool condition, const char* message) {
  if (!condition) return BuiltinError(message);
  return true;
}
template <CheckedInteger T>
BuiltinResult<T> add(T a, T b) {
  T result;
  if (__builtin_add_overflow(a, b, &result))
    return BuiltinError("integer overflow");
  return result;
}
template <CheckedInteger T>
BuiltinResult<T> sub(T a, T b) {
  T result;
  if (__builtin_sub_overflow(a, b, &result))
    return BuiltinError("integer overflow");
  return result;
}
template <CheckedInteger T>
BuiltinResult<T> mul(T a, T b) {
  T result;
  if (__builtin_mul_overflow(a, b, &result))
    return BuiltinError("integer overflow");
  return result;
}
template <CheckedInteger T>
BuiltinResult<T> div(T a, T b) {
  if (!b) return BuiltinError("division by zero");
  if constexpr (::std::numeric_limits<T>::is_signed)
    if (a == ::std::numeric_limits<T>::min() && b == T(-1))
      return BuiltinError("integer overflow");
  return T(a / b);
}
template <CheckedInteger T>
BuiltinResult<T> rem(T a, T b) {
  if (!b) return BuiltinError("division by zero");
  if constexpr (::std::numeric_limits<T>::is_signed)
    if (a == ::std::numeric_limits<T>::min() && b == T(-1))
      return BuiltinError("integer overflow");
  return T(a % b);
}
template <CheckedInteger T>
BuiltinResult<T> neg(T value) {
  return sub(T(0), value);
}
inline BuiltinResult<double> finite(double value) {
  if (!::std::isfinite(value)) return BuiltinError("nonfinite floating value");
  return value;
}
template <class I>
::std::size_t position(I value) {
  if (value < 0 || static_cast<unsigned __int128>(value) >
                       ::std::numeric_limits<::std::size_t>::max())
    throw BuiltinError("invalid index");
  return static_cast<::std::size_t>(value);
}
template <class V, class I>
auto index(const V& values, I value) -> BuiltinResult<typename V::value_type> {
  return attempt([&]() -> typename V::value_type {
    auto p = position(value);
    if (p >= values.size()) throw BuiltinError("index out of bounds");
    return values[p];
  });
}
template <class V>
::std::uint64_t len(const V& values) {
  return values.size();
}
template <class T = ::std::uint64_t, class I>
BuiltinResult<::std::vector<T>> range(I end) {
  return attempt([&] {
    auto count = position(end);
    ::std::vector<T> values;
    values.reserve(count);
    for (::std::size_t i = 0; i < count; ++i) values.push_back(T(i));
    return values;
  });
}
template <class V, class... Rest>
BuiltinResult<V> concat(const V& first, const Rest&... rest) {
  static_assert(sizeof...(Rest) >= 1);
  return attempt([&] {
    V result;
    auto size = first.size();
    ((size = take(add(size, rest.size()))), ...);
    result.reserve(size);
    result.insert(result.end(), first.begin(), first.end());
    (result.insert(result.end(), rest.begin(), rest.end()), ...);
    return result;
  });
}
template <class V, class T>
bool contains(const V& values, const T& value) {
  return ::std::find(values.begin(), values.end(), value) != values.end();
}
template <class V>
V exclude(const V& values, const V& removed) {
  V result;
  for (const auto& value : values)
    if (!contains(removed, value)) result.push_back(value);
  return result;
}
template <class V>
bool is_disjoint(const V& a, const V& b) {
  for (const auto& v : a)
    if (contains(b, v)) return false;
  return true;
}
template <class V, class I>
BuiltinResult<V> gather(const V& values, const I& indices) {
  return attempt([&] {
    V result;
    result.reserve(indices.size());
    for (auto i : indices) result.push_back(take(index(values, i)));
    return result;
  });
}
template <class V, class I, class J>
BuiltinResult<V> slice(const V& values, I first, J last) {
  return attempt([&] {
    auto a = position(first), b = position(last);
    if (a > b || b > values.size()) throw BuiltinError("invalid slice");
    return V(values.begin() + a, values.begin() + b);
  });
}
template <class V, class I>
BuiltinResult<V> replace(const V& values, I axis,
                         typename V::value_type value) {
  return attempt([&] {
    auto p = position(axis);
    if (p >= values.size()) throw BuiltinError("invalid index");
    auto result = values;
    result[p] = ::std::move(value);
    return result;
  });
}
template <class V>
auto sum(const V& values) -> BuiltinResult<typename V::value_type> {
  return attempt([&] {
    using T = typename V::value_type;
    T result = 0;
    for (T v : values) result = take(add(result, v));
    return result;
  });
}
template <class V>
auto product(const V& values) -> BuiltinResult<typename V::value_type> {
  return attempt([&] {
    using T = typename V::value_type;
    if (contains(values, T(0))) return T(0);
    T result = 1;
    for (T v : values) result = take(mul(result, v));
    return result;
  });
}
template <class V>
bool all(const V& values) {
  for (bool v : values)
    if (!v) return false;
  return true;
}
template <class V>
bool any(const V& values) {
  for (bool v : values)
    if (v) return true;
  return false;
}
template <CheckedInteger T>
T min(T a, T b) {
  return a < b ? a : b;
}
template <CheckedInteger T>
T max(T a, T b) {
  return a > b ? a : b;
}
template <CheckedInteger T>
BuiltinResult<T> floor_div(T a, T b) {
  return attempt([&] {
    if (b <= 0) throw BuiltinError("nonpositive divisor");
    auto q = take(div(a, b));
    return take(rem(a, b)) < 0 ? take(sub(q, T(1))) : q;
  });
}
template <CheckedInteger T>
BuiltinResult<T> ceil_div(T a, T b) {
  return attempt([&] {
    if (b <= 0) throw BuiltinError("nonpositive divisor");
    auto q = take(div(a, b));
    return take(rem(a, b)) > 0 ? take(add(q, T(1))) : q;
  });
}
// Parsing avoids out-of-range C++ integer literals, including i128's minimum.
inline Integer integer(::std::string_view text) {
  bool negative = !text.empty() && text.front() == '-';
  if (!text.empty() && (text.front() == '-' || text.front() == '+'))
    text.remove_prefix(1);
  if (text.empty()) throw BuiltinError("invalid integer");
  Integer result = 0;
  for (char c : text) {
    if (c < '0' || c > '9') throw BuiltinError("invalid integer");
    result = take(mul(result, Integer(10)));
    result = negative ? take(sub(result, Integer(c - '0')))
                      : take(add(result, Integer(c - '0')));
  }
  return result;
}
}  // namespace common
}  // namespace @TEPL_NAMESPACE@::builtins
