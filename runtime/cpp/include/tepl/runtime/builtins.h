#pragma once
#include <numeric>

#include "values.h"
namespace tepl_runtime::builtins {
inline Integer arithmetic(std::string_view op, Integer x, Integer y) {
  Integer out{};
  bool bad = true;
  if (op == "+")
    bad = __builtin_add_overflow(x, y, &out);
  else if (op == "-")
    bad = __builtin_sub_overflow(x, y, &out);
  else if (op == "*")
    bad = __builtin_mul_overflow(x, y, &out);
  else if (op == "/" || op == "%") {
    require(y != 0, "division by zero");
    require(!(x == std::numeric_limits<Integer>::min() && y == -1),
            "integer overflow");
    out = op == "/" ? x / y : x % y;
    bad = false;
  }
  require(!bad, "integer overflow or invalid operator");
  return out;
}
inline bool contains(const Value::List& xs, const Value& x, Budget& b) {
  b.charge(xs.size());
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}
inline bool distinct(const Value::List& xs, Budget& b) {
  for (std::size_t i = 0; i < xs.size(); ++i) {
    b.charge(i);
    if (std::find(xs.begin(), xs.begin() + i, xs[i]) != xs.begin() + i)
      return false;
  }
  return true;
}
inline bool valid(const Value::List& xs, Integer n, Budget& b) {
  if (n < 0 || UnsignedInteger(n) > UINT64_MAX || !distinct(xs, b))
    return false;
  for (const auto& x : xs)
    if (x.integer() < 0 || x.integer() >= n) return false;
  return true;
}
inline Value index(const Value& v, const Value& at, Budget& b) {
  b.arguments({v, at});
  auto i = at.position();
  const auto& xs = v.list();
  require(i < xs.size(), "index out of bounds");
  return xs[i];
}
inline Value operate(std::string_view op, const std::vector<Value>& a,
                     Budget& b) {
  b.arguments(a);
  if (op == "==" || op == "!=") {
    require(a.size() == 2, "invalid operator arity");
    return Value((a[0] == a[1]) == (op == "=="));
  }
  if (op == "!") {
    require(a.size() == 1, "invalid operator arity");
    return Value(!a[0].boolean());
  }
  require(a.size() == 2 || (a.size() == 1 && (op == "+" || op == "-")),
          "invalid operator arity");
  if (auto p = std::get_if<double>(&a[0].data)) {
    auto x = *p, y = a.size() == 1 ? 0 : a[1].get<double>("expected float");
    if (op == "<") return Value(x < y);
    if (op == "<=") return Value(x <= y);
    if (op == ">") return Value(x > y);
    if (op == ">=") return Value(x >= y);
    double n{};
    if (a.size() == 1)
      n = op == "-" ? -x : x;
    else if (op == "+")
      n = x + y;
    else if (op == "-")
      n = x - y;
    else if (op == "*")
      n = x * y;
    else if (op == "/")
      n = x / y;
    else
      require(false, "unsupported floating operator");
    require(std::isfinite(n), "nonfinite floating value");
    return Value(n);
  }
  auto x = a[0].integer();
  if (a.size() == 1) return Value(op == "-" ? arithmetic("-", 0, x) : x);
  auto y = a[1].integer();
  if (op == "<") return Value(x < y);
  if (op == "<=") return Value(x <= y);
  if (op == ">") return Value(x > y);
  if (op == ">=") return Value(x >= y);
  return Value(arithmetic(op, x, y));
}
inline Value concat(const std::vector<Value>& a, Budget& b) {
  Value::List out;
  for (const auto& x : a) {
    auto& xs = x.list();
    require(xs.size() <= std::numeric_limits<std::size_t>::max() - out.size(),
            "list size overflow");
    b.size(out.size() + xs.size());
    out.insert(out.end(), xs.begin(), xs.end());
  }
  return Value(out);
}
inline Value exclude(const std::vector<Value>& a, Budget& b) {
  Value::List out;
  for (const auto& x : a[0].list())
    if (!contains(a[1].list(), x, b)) out.push_back(x);
  return Value(out);
}
inline Value disjoint(const std::vector<Value>& a, Budget& b) {
  for (const auto& x : a[0].list())
    if (contains(a[1].list(), x, b)) return Value(false);
  return Value(true);
}
inline Value call(std::string_view name, const std::vector<Value>& a,
                  Budget& b) {
  std::size_t arity = 1;
  if (name == "concat")
    require(a.size() >= 2, "invalid builtin arity");
  else {
    if (name == "gather" || name == "exclude" || name == "contains" ||
        name == "is_disjoint" || name == "is_valid_axis_list" ||
        name == "broadcast_shape" || name == "min" || name == "max" ||
        name == "floor_div" || name == "ceil_div" || name == "filter" ||
        name == "difference" || name == "subsequence")
      arity = 2;
    if (name == "slice" || name == "replace") arity = 3;
    require(a.size() == arity, "invalid builtin arity");
  }
  b.arguments(a);
  if (name == "i64") return to_value(from_value<std::int64_t>(a[0]));
  if (name == "u64") return to_value(from_value<std::uint64_t>(a[0]));
  if (name == "len") return Value(Integer(a[0].list().size()));
  if (name == "range") {
    auto n = a[0].position();
    b.size(n);
    Value::List out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.emplace_back(Integer(i));
    return Value(out);
  }
  if (name == "concat") return concat(a, b);
  if (name == "gather") {
    const auto& xs = a[0].list();
    Value::List out;
    for (const auto& x : a[1].list()) {
      auto p = x.position();
      require(p < xs.size(), "gather out of bounds");
      out.push_back(xs[p]);
    }
    return Value(out);
  }
  if (name == "slice") {
    const auto& xs = a[0].list();
    auto start = a[1].position(), end = a[2].position();
    require(start <= end && end <= xs.size(), "invalid slice");
    return Value(Value::List(xs.begin() + start, xs.begin() + end));
  }
  if (name == "replace") {
    auto xs = a[0].list();
    auto i = a[1].position();
    require(i < xs.size(), "replace out of bounds");
    xs[i] = a[2];
    return Value(xs);
  }
  if (name == "contains") return Value(contains(a[0].list(), a[1], b));
  if (name == "exclude") return exclude(a, b);
  if (name == "is_disjoint") return disjoint(a, b);
  if (name == "filter" || name == "difference") {
    auto& xs = a[0].list();
    auto& ys = a[1].list();
    for (const auto& x : xs) x.get<Axis>("expected logical axis");
    for (const auto& x : ys) x.get<Axis>("expected logical axis");
    Value::List out;
    for (const auto& x : xs) {
      b.tick();
      if (contains(ys, x, b) == (name == "filter")) out.push_back(x);
    }
    return Value(out);
  }
  if (name == "subsequence") {
    const auto& xs = a[0].list();
    std::size_t i = 0;
    for (const auto& x : a[1].list()) {
      b.tick();
      if (i < xs.size() && xs[i] == x) ++i;
    }
    return Value(i == xs.size());
  }
  if (name == "all" || name == "any") {
    for (const auto& x : a[0].list())
      if (x.boolean() != (name == "all")) return Value(name == "any");
    return Value(name == "all");
  }
  if (name == "sum" || name == "product") {
    const auto& xs = a[0].list();
    if (name == "product")
      for (const auto& x : xs)
        if (x.integer() == 0) return Value(Integer(0));
    Integer n = name == "sum" ? 0 : 1;
    for (const auto& x : xs)
      n = arithmetic(name == "sum" ? "+" : "*", n, x.integer());
    return Value(n);
  }
  if (name == "min" || name == "max")
    return Value(name == "min" ? std::min(a[0].integer(), a[1].integer())
                               : std::max(a[0].integer(), a[1].integer()));
  if (name == "floor_div" || name == "ceil_div") {
    auto x = a[0].integer(), y = a[1].integer();
    require(y > 0, "rounded division requires positive divisor");
    auto q = arithmetic("/", x, y), r = arithmetic("%", x, y);
    return Value(r < 0 && name == "floor_div"  ? arithmetic("-", q, 1)
                 : r > 0 && name == "ceil_div" ? arithmetic("+", q, 1)
                                               : q);
  }
  if (name == "is_valid_axis_list")
    return Value(valid(a[0].list(), a[1].integer(), b));
  if (name == "broadcast_shape") {
    auto& xs = a[0].list();
    auto& ys = a[1].list();
    Value::List out;
    for (std::size_t i = 0; i < std::max(xs.size(), ys.size()); ++i) {
      auto x = i < xs.size() ? xs[xs.size() - 1 - i].integer() : 1,
           y = i < ys.size() ? ys[ys.size() - 1 - i].integer() : 1;
      require(x >= 0 && y >= 0, "negative extent");
      require(x == y || x == 1 || y == 1, "incompatible broadcast");
      out.emplace_back(x == 1 ? y : x);
    }
    std::reverse(out.begin(), out.end());
    return Value(out);
  }
  if (name == "is_float" || name == "is_integer" ||
      name == "is_signed_integer" || name == "is_unsigned_integer" ||
      name == "is_numeric") {
    auto& s = a[0].string();
    require(dtype_from_name(s).has_value(), "unknown dtype");
    bool f = s.starts_with('f') || s == "bf16", i = s.starts_with('i'),
         u = s.starts_with('u');
    return Value(name == "is_float"              ? f
                 : name == "is_integer"          ? i || u
                 : name == "is_signed_integer"   ? i
                 : name == "is_unsigned_integer" ? u
                                                 : f || i || u);
  }
  throw EvalError(ErrorKind::Invalid, "unknown builtin: " + std::string(name));
}
}  // namespace tepl_runtime::builtins
