#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace @TEPL_NAMESPACE@ {
struct Arity {
  ::std::size_t count;
  bool variadic = false;
  bool accepts(::std::size_t actual) const {
    return variadic ? actual >= count : actual == count;
  }
};
enum class DType {
  Bool,
  I8,
  I16,
  I32,
  I64,
  U8,
  U16,
  U32,
  U64,
  F16,
  BF16,
  F32,
  F64
};
inline ::std::string_view dtype_name(DType dtype) {
  constexpr ::std::string_view names[] = {"bool", "i8",  "i16", "i32", "i64",
                                          "u8",   "u16", "u32", "u64", "f16",
                                          "bf16", "f32", "f64"};
  return names[static_cast<::std::size_t>(dtype)];
}
inline ::std::optional<DType> dtype_from_name(::std::string_view name) {
  for (unsigned i = 0; i < 13; ++i)
    if (dtype_name(static_cast<DType>(i)) == name) return static_cast<DType>(i);
  return {};
}
inline bool valid_literal(::std::string_view value) {
  if (!value.empty() && (value.front() == '+' || value.front() == '-'))
    value.remove_prefix(1);
  const auto digits = [](::std::string_view s) {
    return !s.empty() && ::std::all_of(s.begin(), s.end(), [](char c) {
      return c >= '0' && c <= '9';
    });
  };
  auto dot = value.find('.');
  return dot == value.npos
             ? digits(value)
             : digits(value.substr(0, dot)) && digits(value.substr(dot + 1));
}
inline bool accepts_literal(DType dtype, ::std::string_view value) {
  if (!valid_literal(value)) return false;
  if (dtype >= DType::F16) return true;
  bool negative = value.front() == '-';
  if (value.front() == '-' || value.front() == '+') value.remove_prefix(1);
  auto first = value.find_first_not_of('0');
  value = first == value.npos ? ::std::string_view("0") : value.substr(first);
  bool unsigned_type =
      dtype == DType::Bool || (dtype >= DType::U8 && dtype <= DType::U64);
  constexpr ::std::string_view positive[] = {"1",
                                             "127",
                                             "32767",
                                             "2147483647",
                                             "9223372036854775807",
                                             "255",
                                             "65535",
                                             "4294967295",
                                             "18446744073709551615"};
  constexpr ::std::string_view negative_limits[] = {"1",
                                                    "128",
                                                    "32768",
                                                    "2147483648",
                                                    "9223372036854775808",
                                                    "255",
                                                    "65535",
                                                    "4294967295",
                                                    "18446744073709551615"};
  auto limit = (negative ? negative_limits
                         : positive)[static_cast<::std::size_t>(dtype)];
  return !(unsigned_type && negative) && value.find('.') == value.npos &&
         (value.size() < limit.size() ||
          (value.size() == limit.size() && value <= limit));
}
enum class Precision { Default, High, Highest };
struct DotAlgorithm {
  ::std::string lhs_precision_type, rhs_precision_type, accumulation_type;
  ::std::int64_t lhs_component_count{}, rhs_component_count{},
      num_primitive_operations{};
  bool allow_imprecise_accumulation{};
  bool operator==(const DotAlgorithm&) const = default;
  ::std::size_t hash_value() const;
};
struct ReplicaGroups {
  ::std::variant<::std::vector<::std::vector<::std::int64_t>>,
                 ::std::vector<::std::uint8_t>>
      value;
  bool operator==(const ReplicaGroups&) const = default;
  ::std::size_t hash_value() const;
};
struct Region {
  ::std::vector<::std::uint8_t> bytes;
  bool operator==(const Region&) const = default;
  ::std::size_t hash_value() const;
};
struct Elements {
  ::std::string element_type;
  ::std::vector<::std::uint64_t> shape;
  ::std::vector<::std::uint8_t> data;
  bool operator==(const Elements&) const = default;
  ::std::size_t hash_value() const;
};
namespace detail {
inline void hash_combine(::std::size_t& seed, ::std::size_t value) {
  seed ^= value + 0x9e3779b9u + (seed << 6) + (seed >> 2);
}
template <class T>
::std::size_t hash_value(const T& value);
template <class T>
::std::size_t hash_value(const ::std::vector<T>& values);
template <class T>
::std::size_t hash_value(const ::std::optional<T>& value);
template <class... T>
::std::size_t hash_value(const ::std::variant<T...>& value);
template <class T>
::std::size_t hash_value(const T& value) {
  if constexpr (requires { value.hash_value(); })
    return value.hash_value();
  else
    return ::std::hash<T>{}(value);
}
template <class T>
::std::size_t hash_value(const ::std::vector<T>& values) {
  ::std::size_t seed = values.size();
  for (const T& value : values) hash_combine(seed, hash_value(value));
  return seed;
}
template <class T>
::std::size_t hash_value(const ::std::optional<T>& value) {
  ::std::size_t seed = value.has_value();
  if (value) hash_combine(seed, hash_value(*value));
  return seed;
}
template <class... T>
::std::size_t hash_value(const ::std::variant<T...>& value) {
  ::std::size_t seed = value.index();
  ::std::visit([&](const auto& v) { hash_combine(seed, hash_value(v)); },
               value);
  return seed;
}
}  // namespace detail
// Opaque host metadata participates in node identity.
inline ::std::size_t hash_payload(const DotAlgorithm& value) {
  ::std::size_t seed = detail::hash_value(value.lhs_precision_type);
  detail::hash_combine(seed, detail::hash_value(value.rhs_precision_type));
  detail::hash_combine(seed, detail::hash_value(value.accumulation_type));
  detail::hash_combine(seed, detail::hash_value(value.lhs_component_count));
  detail::hash_combine(seed, detail::hash_value(value.rhs_component_count));
  detail::hash_combine(seed,
                       detail::hash_value(value.num_primitive_operations));
  detail::hash_combine(seed,
                       detail::hash_value(value.allow_imprecise_accumulation));
  return seed;
}
inline ::std::size_t DotAlgorithm::hash_value() const {
  return hash_payload(*this);
}
inline ::std::size_t ReplicaGroups::hash_value() const {
  return detail::hash_value(value);
}
inline ::std::size_t Region::hash_value() const {
  return detail::hash_value(bytes);
}
inline ::std::size_t Elements::hash_value() const {
  auto seed = detail::hash_value(element_type);
  detail::hash_combine(seed, detail::hash_value(shape));
  detail::hash_combine(seed, detail::hash_value(data));
  return seed;
}
// Keep the Rust module's qualified type paths available to C++ consumers.
namespace types {
using ::@TEPL_NAMESPACE@::DotAlgorithm;
using ::@TEPL_NAMESPACE@::DType;
using ::@TEPL_NAMESPACE@::Elements;
using ::@TEPL_NAMESPACE@::Precision;
using ::@TEPL_NAMESPACE@::Region;
using ::@TEPL_NAMESPACE@::ReplicaGroups;
}  // namespace types
}  // namespace @TEPL_NAMESPACE@
