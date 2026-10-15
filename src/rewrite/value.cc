#include "src/rewrite/value.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace tepl::rewrite {
namespace {
void ensure(bool b, const char* text) {
  if (!b) throw EvaluationError(text);
}
Integer arithmetic(const std::string& op, Integer a, Integer b) {
  Integer out = 0;
  if (op == "+")
    ensure(!__builtin_add_overflow(a, b, &out), "integer overflow");
  else if (op == "-")
    ensure(!__builtin_sub_overflow(a, b, &out), "integer overflow");
  else if (op == "*")
    ensure(!__builtin_mul_overflow(a, b, &out), "integer overflow");
  else {
    ensure(b != 0, "division by zero");
    ensure(!(a == std::numeric_limits<Integer>::min() && b == -1),
           "integer overflow");
    out = op == "%" ? a % b : a / b;
  }
  return out;
}
bool contains(const List& xs, const Value& x, Budget& budget) {
  budget.charge(xs.size());
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}
bool distinct(const List& xs, Budget& budget) {
  for (std::size_t i = 0; i < xs.size(); ++i)
    for (std::size_t j = 0; j < i; ++j) {
      budget.tick();
      if (xs[i] == xs[j]) return false;
    }
  return true;
}
bool valid(const List& xs, Integer rank, Budget& budget) {
  if (!distinct(xs, budget)) return false;
  for (const auto& x : xs)
    if (integer(x) < 0 || integer(x) >= rank) return false;
  return true;
}
Value numeric(const std::string& op, const std::vector<Value>& a) {
  if (std::holds_alternative<double>(a[0].data)) {
    double l = std::get<double>(a[0].data),
           r = a.size() == 1 ? 0 : std::get<double>(a[1].data), n = 0;
    if (op == "<") return l < r;
    if (op == "<=") return l <= r;
    if (op == ">") return l > r;
    if (op == ">=") return l >= r;
    if (a.size() == 1)
      n = op == "-" ? -l : l;
    else if (op == "+")
      n = l + r;
    else if (op == "-")
      n = l - r;
    else if (op == "*")
      n = l * r;
    else if (op == "/")
      n = l / r;
    else
      throw EvaluationError("unsupported floating operator");
    ensure(std::isfinite(n), "nonfinite floating value");
    return n;
  }
  auto l = integer(a[0]);
  if (a.size() == 1) return op == "-" ? arithmetic("-", 0, l) : l;
  auto r = integer(a[1]);
  if (op == "<") return l < r;
  if (op == "<=") return l <= r;
  if (op == ">") return l > r;
  if (op == ">=") return l >= r;
  return arithmetic(op, l, r);
}
}  // namespace
Integer integer(const Value& v) {
  if (auto p = std::get_if<Integer>(&v.data)) return *p;
  throw EvaluationError("expected integer");
}
bool boolean(const Value& v) {
  if (auto p = std::get_if<bool>(&v.data)) return *p;
  throw EvaluationError("expected bool");
}
const List& list(const Value& v) {
  if (auto p = std::get_if<List>(&v.data)) return *p;
  throw EvaluationError("expected list");
}
std::size_t position(const Value& v) {
  auto n = integer(v);
  ensure(n >= 0 && static_cast<unsigned __int128>(n) <=
                       std::numeric_limits<std::size_t>::max(),
         "invalid index");
  return static_cast<std::size_t>(n);
}
Integer parseInteger(const std::string& text) {
  std::size_t i = 0;
  bool negative = false;
  if (!text.empty() && (text[0] == '-' || text[0] == '+'))
    negative = text[i++] == '-';
  ensure(i < text.size(), "invalid integer");
  Integer n = 0;
  for (; i < text.size(); ++i) {
    ensure(text[i] >= '0' && text[i] <= '9', "invalid integer");
    n = arithmetic("*", n, 10);
    n = arithmetic(negative ? "-" : "+", n, text[i] - '0');
  }
  return n;
}
std::string integerText(Integer n) {
  bool negative = n < 0;
  auto u = negative ? static_cast<unsigned __int128>(-(n + 1)) + 1
                    : static_cast<unsigned __int128>(n);
  std::string s;
  do {
    s.push_back('0' + u % 10);
    u /= 10;
  } while (u);
  if (negative) s.push_back('-');
  std::reverse(s.begin(), s.end());
  return s;
}
Value attributeValue(const core::AttributeValue& a) {
  using K = core::AttributeValue::Kind;
  switch (a.kind) {
    case K::kInteger:
      return parseInteger(a.text);
    case K::kBool:
      return a.text == "true";
    case K::kString:
    case K::kEnum:
      return a.text;
    case K::kNone:
      return {};
    case K::kList: {
      List xs;
      for (const auto& e : a.elements) xs.push_back(attributeValue(e));
      return xs;
    }
  }
  throw EvaluationError("invalid attribute");
}
Value checkType(Value v, const core::Type& t) {
  if (t.optional) {
    if (std::holds_alternative<std::monostate>(v.data)) return v;
    auto inner = t;
    inner.optional = false;
    return checkType(std::move(v), inner);
  }
  if (auto element = core::elementType(t)) {
    for (const auto& x : list(v)) checkType(x, *element);
    return v;
  }
  using K = core::TypeKind;
  switch (t.kind) {
    case K::kIndex: {
      auto n = integer(v);
      ensure(n >= 0 && static_cast<unsigned __int128>(n) <=
                           std::numeric_limits<std::uint64_t>::max(),
             "index overflow");
      break;
    }
    case K::kI64: {
      auto n = integer(v);
      ensure(n >= std::numeric_limits<std::int64_t>::min() &&
                 n <= std::numeric_limits<std::int64_t>::max(),
             "i64 overflow");
      break;
    }
    case K::kBool:
      boolean(v);
      break;
    case K::kF64:
      ensure(std::holds_alternative<double>(v.data) &&
                 std::isfinite(std::get<double>(v.data)),
             "expected finite f64");
      break;
    case K::kDescriptor:
      ensure(std::holds_alternative<DescriptorValue>(v.data) && t.schema &&
                 std::get<DescriptorValue>(v.data).schema == t.schema->value,
             "descriptor schema mismatch");
      break;
    case K::kTensor:
      ensure(std::holds_alternative<NodeRef>(v.data), "expected tensor");
      break;
    case K::kDType:
      ensure(std::holds_alternative<std::string>(v.data) &&
                 core::resolveDType(std::get<std::string>(v.data)).has_value(),
             "invalid dtype");
      break;
    case K::kPrecision:
      ensure(std::holds_alternative<std::string>(v.data) &&
                 (std::get<std::string>(v.data) == "default" ||
                  std::get<std::string>(v.data) == "high" ||
                  std::get<std::string>(v.data) == "highest"),
             "invalid precision");
      break;
    case K::kString:
      ensure(std::holds_alternative<std::string>(v.data), "expected string");
      break;
    case K::kOpaque:
      throw EvaluationError("opaque attribute values are unavailable");
    case K::kIndexList:
      break;
  }
  return v;
}
core::AttributeValue attribute(const Value& v,
                               const core::AttributeField& field) {
  using K = core::AttributeValue::Kind;
  if (field.optional && std::holds_alternative<std::monostate>(v.data))
    return {K::kNone, {}, {}};
  if (field.list_depth) {
    core::AttributeValue out{K::kList, {}, {}};
    auto f = field;
    --f.list_depth;
    f.optional = false;
    for (const auto& x : list(v)) out.elements.push_back(attribute(x, f));
    return out;
  }
  auto type = core::resolveType(field.type);
  ensure(type.has_value(), "unsupported concrete attribute type");
  checkType(v, *type);
  if (field.type == "index" || field.type == "i64")
    return {K::kInteger, integerText(integer(v)), {}};
  if (field.type == "bool")
    return {K::kBool, boolean(v) ? "true" : "false", {}};
  return {field.type == "string" ? K::kString : K::kEnum,
          std::get<std::string>(v.data),
          {}};
}
Value call(const std::string& n, const std::vector<Value>& a, Budget& budget) {
  budget.tick();
  for (const auto& v : a)
    if (const auto* xs = std::get_if<List>(&v.data)) {
      budget.size(xs->size());
      budget.charge(xs->size());
    }
  if (n == "+") return numeric(n, a);
  if (n == "-" || n == "*" || n == "/" || n == "%" || n == "<" || n == "<=" ||
      n == ">" || n == ">=")
    return numeric(n, a);
  if (n == "==") return a.at(0) == a.at(1);
  if (n == "!=") return a.at(0) != a.at(1);
  if (n == "!") return !boolean(a.at(0));
  if (n.starts_with("dtype_")) {
    ensure(core::resolveDType(n.substr(6)).has_value(), "unknown dtype");
    return n.substr(6);
  }
  if (n == "is_float" || n == "is_integer" || n == "is_signed_integer" ||
      n == "is_unsigned_integer" || n == "is_numeric") {
    auto s = std::get<std::string>(a.at(0).data);
    ensure(core::resolveDType(s).has_value(), "unknown dtype");
    bool f = s[0] == 'f' || s == "bf16", i = s[0] == 'i', u = s[0] == 'u';
    return n == "is_float"              ? f
           : n == "is_integer"          ? (i || u)
           : n == "is_signed_integer"   ? i
           : n == "is_unsigned_integer" ? u
                                        : (f || i || u);
  }
  if (n == "len" || n == "rank") return Integer(list(a.at(0)).size());
  if (n == "range" || n == "all_axes") {
    auto count = n == "range" ? position(a.at(0)) : list(a.at(0)).size();
    budget.size(count);
    List xs;
    for (std::size_t i = 0; i < count; ++i) xs.emplace_back(Integer(i));
    return xs;
  }
  if (n == "shape") return a.at(0);
  if (n == "empty_axes") return List{};
  if (n == "concat") {
    List xs;
    for (const auto& v : a) {
      const auto& part = list(v);
      budget.size(xs.size() + part.size());
      xs.insert(xs.end(), part.begin(), part.end());
    }
    return xs;
  }
  if (n == "index") {
    const auto& xs = list(a.at(0));
    auto i = position(a.at(1));
    ensure(i < xs.size(), "index out of bounds");
    return xs[i];
  }
  if (n == "gather" || n == "permute") {
    List xs;
    const auto& src = list(a.at(0));
    for (const auto& p : list(a.at(1))) {
      auto i = position(p);
      ensure(i < src.size(), "gather out of bounds");
      xs.push_back(src[i]);
    }
    return xs;
  }
  if (n == "slice") {
    const auto& xs = list(a.at(0));
    auto b = position(a.at(1)), e = position(a.at(2));
    ensure(b <= e && e <= xs.size(), "invalid slice");
    return List(xs.begin() + b, xs.begin() + e);
  }
  if (n == "replace") {
    auto xs = list(a.at(0));
    auto i = position(a.at(1));
    ensure(i < xs.size(), "replace out of bounds");
    xs[i] = a.at(2);
    return xs;
  }
  if (n == "contains") return contains(list(a.at(0)), a.at(1), budget);
  if (n == "all" || n == "any") {
    bool out = n == "all";
    for (const auto& v : list(a.at(0)))
      if (boolean(v) != (n == "all")) return n == "any";
    return out;
  }
  if (n == "sum" || n == "product") {
    if (n == "product")
      for (const auto& v : list(a.at(0)))
        if (integer(v) == 0) return Integer(0);
    Integer out = n == "sum" ? 0 : 1;
    for (const auto& v : list(a.at(0)))
      out = arithmetic(n == "sum" ? "+" : "*", out, integer(v));
    return out;
  }
  if (n == "min" || n == "max")
    return n == "min" ? std::min(integer(a.at(0)), integer(a.at(1)))
                      : std::max(integer(a.at(0)), integer(a.at(1)));
  if (n == "floor_div" || n == "ceil_div") {
    auto x = integer(a.at(0)), y = integer(a.at(1));
    ensure(y > 0, "rounded division requires positive divisor");
    auto q = arithmetic("/", x, y), r = arithmetic("%", x, y);
    if (r && ((r < 0) != (y < 0)) && n == "floor_div")
      q = arithmetic("-", q, 1);
    if (r && ((r < 0) == (y < 0)) && n == "ceil_div") q = arithmetic("+", q, 1);
    return q;
  }
  if (n == "is_valid_axis_list" || n == "axis_partition") {
    const auto& xs = list(a.at(0));
    auto rank = integer(a.at(1));
    return valid(xs, rank, budget) &&
           (n != "axis_partition" || Integer(xs.size()) == rank);
  }
  if (n == "inverse") {
    const auto& p = list(a.at(0));
    ensure(valid(p, p.size(), budget), "invalid permutation");
    List out(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) out[position(p[i])] = Integer(i);
    return out;
  }
  if (n == "order_positions") {
    const auto& p = list(a.at(0));
    List out;
    for (std::size_t i = 0; i < p.size(); ++i) out.emplace_back(Integer(i));
    std::stable_sort(out.begin(), out.end(), [&](const auto& x, const auto& y) {
      return integer(p[position(x)]) < integer(p[position(y)]);
    });
    return out;
  }
  if (n == "offset_positions" || n == "offset_extents" ||
      n == "scale_extents") {
    auto xs = list(a.at(0));
    for (auto& x : xs)
      x = arithmetic(n == "scale_extents" ? "*" : "+", integer(x),
                     integer(a.at(1)));
    return xs;
  }
  if (n == "extent_add" || n == "extent_sub" || n == "extent_mul" ||
      n == "extent_floor_div" || n == "extent_le") {
    const auto& x = list(a.at(0));
    const auto& y = list(a.at(1));
    ensure(x.size() == y.size(), "extent lengths differ");
    List out;
    bool le = true;
    for (std::size_t i = 0; i < x.size(); ++i) {
      le &= integer(x[i]) <= integer(y[i]);
      if (n == "extent_floor_div") {
        ensure(integer(y[i]) > 0, "invalid extent divisor");
        out.push_back(call("floor_div", {x[i], y[i]}, budget));
      } else if (n != "extent_le")
        out.emplace_back(arithmetic(n == "extent_add"   ? "+"
                                    : n == "extent_sub" ? "-"
                                                        : "*",
                                    integer(x[i]), integer(y[i])));
    }
    return n == "extent_le" ? Value(le) : Value(out);
  }
  if (n == "is_valid_extents" || n == "is_positive_extents") {
    for (const auto& x : list(a.at(0)))
      if (n == "is_valid_extents" ? integer(x) < 0 : integer(x) <= 0)
        return false;
    return true;
  }
  if (n == "distinct_axes") return distinct(list(a.at(0)), budget);
  if (n == "axis_extents") {
    List out;
    for (const auto& x : list(a.at(0)))
      out.emplace_back(std::get<Axis>(x.data).extent);
    return out;
  }
  if (n == "positions") {
    const auto& src = list(a.at(0));
    const auto& wanted = list(a.at(1));
    ensure(distinct(src, budget) && distinct(wanted, budget),
           "noninjective positions");
    List out;
    for (const auto& v : wanted) {
      budget.charge(src.size());
      auto it = std::find(src.begin(), src.end(), v);
      ensure(it != src.end(), "absent axis");
      out.emplace_back(Integer(it - src.begin()));
    }
    return out;
  }
  if (n == "membership_positions" || n == "matched_positions") {
    const auto& x = list(a.at(0));
    const auto& y = list(a.at(1));
    List out;
    for (std::size_t i = 0; i < x.size(); ++i) {
      budget.charge(y.size());
      auto it = std::find(y.begin(), y.end(), x[i]);
      if (it != y.end())
        out.emplace_back(
            Integer(n == "membership_positions" ? i : it - y.begin()));
    }
    return out;
  }
  if (n == "union_axes" || n == "unique_axes") {
    List out;
    for (const auto& v : a)
      for (const auto& x : list(v))
        if (!contains(out, x, budget)) out.push_back(x);
    return out;
  }
  if (n == "exclude" || n == "ordered_complement" || n == "filter_axes" ||
      n == "intersection_axes" || n == "difference_axes" || n == "same_set" ||
      n == "disjoint" || n == "is_disjoint" || n == "subset") {
    const auto& xs = list(a.at(0));
    const auto& ys = list(a.at(1));
    List both, out;
    bool difference =
        n == "exclude" || n == "ordered_complement" || n == "difference_axes";
    for (const auto& x : xs) {
      bool present = contains(ys, x, budget);
      if (present) both.push_back(x);
      if (difference ? !present : present) out.push_back(x);
    }
    if (n == "disjoint" || n == "is_disjoint") return both.empty();
    if (n == "subset") return both.size() == xs.size();
    if (n == "same_set") {
      for (const auto& y : ys)
        if (!contains(xs, y, budget)) return false;
      return both.size() == xs.size();
    }
    return out;
  }
  if (n == "replace_axes") {
    auto xs = list(a.at(0));
    const auto& from = list(a.at(1));
    const auto& to = list(a.at(2));
    ensure(from.size() == to.size() && distinct(from, budget) &&
               distinct(to, budget),
           "invalid axis replacement");
    for (auto& x : xs) {
      budget.charge(from.size());
      auto it = std::find(from.begin(), from.end(), x);
      if (it != from.end()) x = to[it - from.begin()];
    }
    return xs;
  }
  if (n == "place" || n == "place_axes") {
    const auto& left = list(a.at(0));
    const auto& p = list(a.at(1));
    const auto& right = list(a.at(2));
    ensure(
        left.size() == p.size() && valid(p, left.size() + right.size(), budget),
        "invalid axis placement");
    budget.size(left.size() + right.size());
    List out(left.size() + right.size());
    std::vector<bool> occupied(out.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
      auto at = position(p[i]);
      out[at] = left[i];
      occupied[at] = true;
    }
    std::size_t next = 0;
    for (std::size_t i = 0; i < out.size(); ++i)
      if (!occupied[i]) out[i] = right[next++];
    return out;
  }
  if (n == "broadcast_shape") {
    const auto& x = list(a.at(0));
    const auto& y = list(a.at(1));
    List out(std::max(x.size(), y.size()));
    for (std::size_t i = 0; i < out.size(); ++i) {
      auto l = i < x.size() ? integer(x[x.size() - 1 - i]) : 1,
           r = i < y.size() ? integer(y[y.size() - 1 - i]) : 1;
      ensure(l >= 0 && r >= 0, "negative extent");
      ensure(l == r || l == 1 || r == 1, "incompatible broadcast");
      out[out.size() - 1 - i] = l == 1 ? r : l;
    }
    return out;
  }
  throw EvaluationError("unsupported metadata builtin: " + n);
}
}  // namespace tepl::rewrite
