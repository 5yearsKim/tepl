#pragma once
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <set>

#include "error.h"
#include "types.h"

namespace tepl_runtime {
using Integer = __int128_t;
using UnsignedInteger = __uint128_t;
struct Axis {
  std::size_t owner{}, group{}, member{};
  Integer extent{};
  bool operator==(const Axis&) const = default;
};
struct Tensor {
  std::size_t id;
  bool operator==(const Tensor&) const = default;
};
struct Value;
struct Descriptor {
  std::size_t schema;
  std::vector<Value> fields;
  bool operator==(const Descriptor&) const = default;
};
using OpaqueValue = std::variant<DotAlgorithm, ReplicaGroups, Region, Elements>;
struct Value {
  using List = std::vector<Value>;
  using Data = std::variant<std::monostate, Integer, double, bool, std::string,
                            List, Axis, Tensor, Descriptor, OpaqueValue>;
  Data data;
  Value() = default;
  template <class T>
    requires std::constructible_from<Data, T>
  explicit Value(T value) : data(std::move(value)) {}
  bool operator==(const Value&) const = default;
  bool none() const { return data.index() == 0; }
  template <class T>
  const T& get(const char* message) const {
    auto p = std::get_if<T>(&data);
    require(p != nullptr, message);
    return *p;
  }
  Integer integer() const { return get<Integer>("expected integer"); }
  bool boolean() const { return get<bool>("expected bool"); }
  const List& list() const { return get<List>("expected list"); }
  const std::string& string() const {
    return get<std::string>("expected string");
  }
  std::size_t position() const {
    auto x = integer();
    require(
        x >= 0 && UnsignedInteger(x) <= std::numeric_limits<std::size_t>::max(),
        "invalid index");
    return std::size_t(x);
  }
  Value field(std::size_t i) const {
    const auto& d = get<Descriptor>("expected attributes");
    require(i < d.fields.size(), "missing attribute field");
    return d.fields[i];
  }
  std::vector<std::uint64_t> dimensions() const {
    std::vector<std::uint64_t> out;
    for (const auto& v : list()) {
      auto x = v.integer();
      require(x >= 0 && UnsignedInteger(x) <= UINT64_MAX, "index overflow");
      out.push_back(std::uint64_t(x));
    }
    return out;
  }
};
inline Value integer(std::string_view text) {
  bool negative = text.starts_with('-');
  if (negative || text.starts_with('+')) text.remove_prefix(1);
  require(!text.empty(), "invalid integer");
  UnsignedInteger n = 0,
                  limit = (UnsignedInteger(1) << 127) - (negative ? 0 : 1);
  for (char c : text) {
    require(c >= '0' && c <= '9', "invalid integer");
    auto d = unsigned(c - '0');
    require(n <= (limit - d) / 10, "integer overflow");
    n = n * 10 + d;
  }
  return Value(negative ? -Integer(n - (n ? 1 : 0)) - Integer(n != 0)
                        : Integer(n));
}
inline Value floating(std::string_view text) {
  double x{};
  if (text.starts_with('+')) text.remove_prefix(1);
  auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), x);
  require(
      ec == std::errc{} && end == text.data() + text.size() && std::isfinite(x),
      "invalid or nonfinite float");
  return Value(x);
}
inline Value shape(const std::vector<std::uint64_t>& xs) {
  Value::List out;
  for (auto x : xs) out.emplace_back(Integer(x));
  return Value(out);
}
struct Budget {
  std::size_t steps = 1000000, max_list = 1000000;
  void charge(std::size_t n) {
    if (n > steps) throw EvalError(ErrorKind::Limit, "expression step limit");
    steps -= n;
  }
  void tick() { charge(1); }
  void size(std::size_t n) const {
    if (n > max_list) throw EvalError(ErrorKind::Limit, "list size limit");
  }
  void arguments(const std::vector<Value>& args) {
    tick();
    for (const auto& a : args)
      if (auto p = std::get_if<Value::List>(&a.data)) {
        size(p->size());
        charge(p->size());
      }
  }
};
template <class T>
struct AttributeValue;
template <class T>
Value to_value(const T& x) {
  return AttributeValue<T>::to(x);
}
template <class T>
T from_value(const Value& v) {
  return AttributeValue<T>::from(v);
}
template <class T>
  requires std::integral<T>
struct AttributeValue<T> {
  static Value to(T x) {
    if constexpr (std::same_as<T, bool>)
      return Value(x);
    else
      return Value(Integer(x));
  }
  static T from(const Value& v) {
    if constexpr (std::same_as<T, bool>)
      return v.boolean();
    else {
      auto x = v.integer();
      require(x >= Integer(std::numeric_limits<T>::min()) &&
                  x <= Integer(std::numeric_limits<T>::max()),
              "integer conversion overflow");
      return T(x);
    }
  }
};
template <>
struct AttributeValue<std::string> {
  static Value to(const std::string& x) { return Value(x); }
  static std::string from(const Value& x) { return x.string(); }
};
template <>
struct AttributeValue<DType> {
  static Value to(DType x) { return Value(std::string(dtype_name(x))); }
  static DType from(const Value& x) {
    auto d = dtype_from_name(x.string());
    require(d.has_value(), "invalid dtype");
    return *d;
  }
};
template <>
struct AttributeValue<Precision> {
  static Value to(Precision x) {
    return Value(std::string(x == Precision::Default ? "default"
                             : x == Precision::High  ? "high"
                                                     : "highest"));
  }
  static Precision from(const Value& v) {
    auto s = v.string();
    require(s == "default" || s == "high" || s == "highest",
            "invalid precision");
    return s == "default" ? Precision::Default
           : s == "high"  ? Precision::High
                          : Precision::Highest;
  }
};
template <class T>
  requires(std::same_as<T, DotAlgorithm> || std::same_as<T, ReplicaGroups> ||
           std::same_as<T, Region> || std::same_as<T, Elements>)
struct AttributeValue<T> {
  static Value to(const T& x) { return Value(OpaqueValue(x)); }
  static T from(const Value& x) {
    auto& o = x.get<OpaqueValue>("expected opaque value");
    auto p = std::get_if<T>(&o);
    require(p != nullptr, "opaque type mismatch");
    return *p;
  }
};
template <class T>
struct AttributeValue<std::vector<T>> {
  static Value to(const std::vector<T>& xs) {
    Value::List out;
    for (const T& x : xs) out.push_back(to_value(x));
    return Value(out);
  }
  static std::vector<T> from(const Value& v) {
    std::vector<T> out;
    for (const auto& x : v.list()) out.push_back(from_value<T>(x));
    return out;
  }
};
template <class T>
struct AttributeValue<std::optional<T>> {
  static Value to(const std::optional<T>& x) {
    return x ? to_value(*x) : Value();
  }
  static std::optional<T> from(const Value& x) {
    return x.none() ? std::nullopt : std::optional<T>(from_value<T>(x));
  }
};
enum class ScalarType {
  Tensor,
  U64,
  I64,
  F64,
  Bool,
  DType,
  String,
  Precision,
  LogicalAxis,
  LogicalAxisSet,
  Opaque,
  Descriptor
};
inline Value checked(Value v, ScalarType t, std::size_t depth, bool optional,
                     std::size_t schema = 0) {
  if (optional && v.none()) return v;
  if (depth) {
    for (const auto& x : v.list()) checked(x, t, depth - 1, false, schema);
    return v;
  }
  switch (t) {
    case ScalarType::U64:
      from_value<std::uint64_t>(v);
      break;
    case ScalarType::I64:
      from_value<std::int64_t>(v);
      break;
    case ScalarType::Bool:
      v.boolean();
      break;
    case ScalarType::F64:
      require(std::isfinite(v.get<double>("expected f64")), "nonfinite f64");
      break;
    case ScalarType::DType:
      from_value<DType>(v);
      break;
    case ScalarType::Precision:
      from_value<Precision>(v);
      break;
    case ScalarType::String:
      v.string();
      break;
    case ScalarType::Tensor:
      v.get<Tensor>("expected tensor");
      break;
    case ScalarType::LogicalAxis:
      v.get<Axis>("expected logical axis");
      break;
    case ScalarType::LogicalAxisSet:
      for (const auto& x : v.list()) x.get<Axis>("expected logical axis");
      break;
    case ScalarType::Descriptor:
      require(v.get<Descriptor>("expected attributes").schema == schema,
              "descriptor schema mismatch");
      break;
    case ScalarType::Opaque:
      v.get<OpaqueValue>("expected opaque value");
      break;
  }
  return v;
}
// Restores lexical locals even when an expression fails.
struct LocalScope {
  std::map<std::size_t, Value>& locals;
  std::map<std::size_t, Value> saved;
  explicit LocalScope(std::map<std::size_t, Value>& x, bool clear = false)
      : locals(x), saved(x) {
    if (clear) locals.clear();
  }
  ~LocalScope() { locals = std::move(saved); }
};
}  // namespace tepl_runtime
