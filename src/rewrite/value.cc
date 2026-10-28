#include "src/rewrite/value.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "src/rewrite/evaluation_support.h"

namespace tepl::rewrite {
using detail::arithmetic;
using detail::ensure;
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
    case K::kU64: {
      auto n = integer(v);
      ensure(n >= 0 && static_cast<unsigned __int128>(n) <=
                           std::numeric_limits<std::uint64_t>::max(),
             "u64 overflow");
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
      if (t == core::logicalAxisSetType()) {
        for (const auto& x : list(v)) checkType(x, core::logicalAxisType());
        break;
      }
      if (t == core::logicalAxisType()) {
        ensure(std::holds_alternative<Axis>(v.data), "expected logical axis");
        break;
      }
      throw EvaluationError("opaque attribute values are unavailable");
    case K::kU64List:
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
  if (field.type == "u64" || field.type == "i64")
    return {K::kInteger, integerText(integer(v)), {}};
  if (field.type == "bool")
    return {K::kBool, boolean(v) ? "true" : "false", {}};
  return {field.type == "string" ? K::kString : K::kEnum,
          std::get<std::string>(v.data),
          {}};
}
}  // namespace tepl::rewrite
