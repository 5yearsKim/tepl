#include "src/codegen/cpp/expression_emitter.h"

#include <stdexcept>
#include <type_traits>

#include "src/codegen/cpp/names.h"
#include "src/shared/builtins/axes.h"
#include "src/shared/builtins/intrinsics/catalog.h"

namespace tepl::codegen::cpp {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
std::string boolean(bool b) { return b ? "true" : "false"; }
std::string text(const std::string& s) {
  return "rt::Value(::std::string(" + quote(s) + "," + num(s.size()) + "))";
}
std::string scope(std::string body) {
  return "([&]() -> rt::Value { " + body + " }())";
}
std::string tick(std::string s) {
  return scope("ctx.budget.tick(); return " + s + ";");
}
std::string call(std::string_view module, std::string_view name,
                 const std::vector<std::string>& args, bool) {
  std::string xs = "::std::vector<rt::Value>{";
  for (const auto& a : args) xs += a + ",";
  return scope("auto args = " + xs + "}; return rt::" + std::string(module) +
               "(" + quote(name) + ",args,ctx.budget);");
}
std::string logical(std::string_view op, const std::vector<std::string>& args) {
  if (op == "&&" || op == "||")
    return "rt::Value((" + args.at(0) + ").boolean() " + std::string(op) +
           " (" + args.at(1) + ").boolean())";
  if (op == "if")
    return "((" + args.at(0) + ").boolean() ? " + args.at(1) + " : " +
           args.at(2) + ")";
  return {};
}
std::string list(const std::vector<std::string>& args) {
  std::string xs = "rt::Value::List{";
  for (const auto& a : args) xs += a + ",";
  return scope(
      "auto xs = " + xs +
      "}; ctx.budget.size(xs.size()); return rt::Value(::std::move(xs));");
}
std::string comprehension(const std::string& iterable,
                          const std::string& element, std::size_t id,
                          bool rule) {
  auto locals = rule ? "ctx.bindings.locals" : "ctx.locals";
  return scope(
      "auto xs = (" + iterable + ").list(); rt::LocalScope saved(" + locals +
      "); rt::Value::List out; for(const auto& item:xs) { ctx.budget.tick(); " +
      locals + "[" + num(id) + "] = item; out.push_back(" + element +
      "); } return rt::Value(::std::move(out));");
}
}  // namespace
std::string scalarType(const core::Type& t) {
  using K = core::TypeKind;
  switch (t.kind) {
    case K::kTensor:
      return "Tensor";
    case K::kU64:
    case K::kU64List:
      return "U64";
    case K::kI64:
      return "I64";
    case K::kF64:
      return "F64";
    case K::kBool:
      return "Bool";
    case K::kDType:
      return "DType";
    case K::kString:
      return "String";
    case K::kPrecision:
      return "Precision";
    case K::kOpaque:
      if (t.opaque_name == core::logicalAxisType().opaque_name)
        return "LogicalAxis";
      if (t.opaque_name == core::logicalAxisSetType().opaque_name)
        return "LogicalAxisSet";
      return "Opaque";
    case K::kDescriptor:
      return "Descriptor";
  }
  throw std::logic_error("invalid executable type");
}
std::string Expressions::rule(const core::completed::TypedExpr& e) const {
  using namespace core;
  using namespace core::completed;
  const auto& type = program_.types.at(e.type.value);
  auto result = std::visit(
      [&](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, CaptureRef>)
          return "ctx.capture(" + num(v.capture.value) + ")";
        else if constexpr (std::is_same_v<T, DimensionRef>)
          return "ctx.dimension(" + num(v.dimension.value) + ")";
        else if constexpr (std::is_same_v<T, DescriptorRef>)
          return "ctx.descriptor(" + num(v.descriptor.value) + ")";
        else if constexpr (std::is_same_v<T, DescriptorFieldRef>)
          return "ctx.descriptor(" + num(v.descriptor.value) + ").field(" +
                 num(v.field) + ")";
        else if constexpr (std::is_same_v<T, DTypeVariableRef>)
          return "ctx.dtype_variable(" + num(v.variable.value) + ")";
        else if constexpr (std::is_same_v<T, LocalRef>)
          return "ctx.local(" + num(v.local.value) + ")";
        else if constexpr (std::is_same_v<T, DType>)
          return text(std::string(dtypeName(v)));
        else if constexpr (std::is_same_v<T, NumericConstant>)
          return std::string("rt::") +
                 (type.kind == TypeKind::kF64 ? "floating" : "integer") + "(" +
                 quote(v.spelling) + ")";
        else if constexpr (std::is_same_v<T, bool>)
          return "rt::Value(" + boolean(v) + ")";
        else if constexpr (std::is_same_v<T, StringConstant> ||
                           std::is_same_v<T, EnumConstant>)
          return text(v.value);
        else if constexpr (std::is_same_v<T, NoneConstant>)
          return "rt::Value()";
        else if constexpr (std::is_same_v<T, OptionalValue>)
          return rule(*v.value);
        else if constexpr (std::is_same_v<T, BuiltinCall>) {
          std::vector<std::string> args;
          for (const auto& a : v.arguments) args.push_back(rule(*a));
          return call("builtins::call", builtins::signature(v.builtin).name,
                      args, true);
        } else if constexpr (std::is_same_v<T, UnaryExpr>)
          return call("builtins::operate", spelling(v.op), {rule(*v.operand)},
                      true);
        else if constexpr (std::is_same_v<T, BinaryExpr>) {
          std::vector<std::string> args = {rule(*v.lhs), rule(*v.rhs)};
          auto lazy = logical(spelling(v.op), args);
          return lazy.empty()
                     ? call("builtins::operate", spelling(v.op), args, true)
                     : lazy;
        } else if constexpr (std::is_same_v<T, LocalLet>)
          return scope("auto value = " + rule(*v.value) +
                       "; ctx.bindings.locals[" + num(v.local.value) +
                       "] = value; return value;");
        else if constexpr (std::is_same_v<T, ListExpr>) {
          std::vector<std::string> args;
          for (const auto& a : v.elements) args.push_back(rule(*a));
          return list(args);
        } else if constexpr (std::is_same_v<T, IndexExpr>)
          return scope("auto value = " + rule(*v.value) +
                       "; auto at = " + rule(*v.index) +
                       "; return rt::builtins::index(value,at,ctx.budget);");
        else if constexpr (std::is_same_v<T, ConditionalExpr>)
          return logical("if", {rule(*v.condition), rule(*v.then_value),
                                rule(*v.else_value)});
        else if constexpr (std::is_same_v<T, ComprehensionExpr>)
          return comprehension(rule(*v.iterable), rule(*v.element),
                               v.variable.value, true);
        else if constexpr (std::is_same_v<T, DescriptorConstruct>) {
          std::string out =
              "rt::Value(rt::Descriptor{" + num(v.schema.value) + ",{";
          for (const auto& f : v.fields) out += rule(*f) + ",";
          return out + "}})";
        } else if constexpr (std::is_same_v<T, SymbolicExpr>)
          return symbolic(*v.value);
        else
          static_assert(!sizeof(T), "unhandled executable expression");
      },
      e.value);
  auto depth = type.list_depth + (type.kind == TypeKind::kU64List ? 1 : 0);
  return tick("rt::checked(" + result +
              ", rt::ScalarType::" + scalarType(type) + ", " + num(depth) +
              ", " + boolean(type.optional) + ", " +
              num(type.schema ? type.schema->value : 0) + ")");
}
std::string Expressions::symbolic(const core::completed::Symbol& s) const {
  using K = core::completed::Symbol::Kind;
  if (s.kind == K::Integer) return tick("rt::integer(" + quote(s.name) + ")");
  if (s.kind == K::Boolean) return tick("rt::Value(" + s.name + ")");
  if (s.kind == K::Shape) return tick("ctx.shape(" + num(s.owner) + ")");
  if (s.kind == K::DType) return tick("ctx.dtype(" + num(s.owner) + ")");
  if (s.kind == K::Field)
    return tick("ctx.descriptor(" + num(s.owner) + ").field(" + num(s.index) +
                ")");
  if (s.kind == K::AxisGroup)
    return tick("axis_" + num(s.owner) + "_" + num(s.index) + "(ctx)");
  std::vector<std::string> args;
  for (const auto& a : s.args) args.push_back(symbolic(*a));
  if (s.kind == K::List) return tick(list(args));
  auto lazy = logical(s.name, args);
  if (!lazy.empty()) return tick(lazy);
  if (s.name.starts_with("dtype_")) return tick(text(s.name.substr(6)));
  auto module = core::intrinsics::resolve(s.name)        ? "intrinsics::call"
                : core::builtins::resolveBuiltin(s.name) ? "builtins::call"
                                                         : "builtins::operate";
  return tick(call(module, s.name, args, true));
}
std::string Expressions::metadata(const core::metadata::Expr& e) const {
  using namespace core::metadata;
  return tick(std::visit(
      [&](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, SymbolRef>)
          return "ctx.local(" + num(v.symbol.value) + ")";
        else if constexpr (std::is_same_v<T, Integer>)
          return "rt::integer(" + quote(v.spelling) + ")";
        else if constexpr (std::is_same_v<T, bool>)
          return "rt::Value(" + boolean(v) + ")";
        else if constexpr (std::is_same_v<T, core::DType>)
          return text(std::string(core::dtypeName(v)));
        else if constexpr (std::is_same_v<T, AttributeRef>)
          return "ctx.field(" + num(v.field) + ")";
        else if constexpr (std::is_same_v<T, Call>) {
          std::vector<std::string> args;
          for (const auto& a : v.arguments) args.push_back(metadata(*a));
          return call("builtins::call",
                      core::builtins::signature(v.builtin).name, args, false);
        } else if constexpr (std::is_same_v<T, Unary>)
          return call("builtins::operate", spelling(v.op),
                      {metadata(*v.operand)}, false);
        else if constexpr (std::is_same_v<T, Binary>) {
          std::vector<std::string> args = {metadata(*v.lhs), metadata(*v.rhs)};
          auto lazy = logical(spelling(v.op), args);
          return lazy.empty()
                     ? call("builtins::operate", spelling(v.op), args, false)
                     : lazy;
        } else if constexpr (std::is_same_v<T, List>) {
          std::vector<std::string> args;
          for (const auto& a : v.elements) args.push_back(metadata(*a));
          return list(args);
        } else if constexpr (std::is_same_v<T, Index>)
          return scope("auto value = " + metadata(*v.value) +
                       "; auto at = " + metadata(*v.index) +
                       "; return rt::builtins::index(value,at,ctx.budget);");
        else if constexpr (std::is_same_v<T, Conditional>)
          return logical("if", {metadata(*v.condition), metadata(*v.then_value),
                                metadata(*v.else_value)});
        else if constexpr (std::is_same_v<T, Comprehension>)
          return comprehension(metadata(*v.iterable), metadata(*v.element),
                               v.variable.value, false);
        else
          static_assert(!sizeof(T), "unhandled metadata expression");
      },
      e.value));
}
std::string Expressions::axis(const core::axis::Meta& m) const {
  using K = core::axis::Meta::Kind;
  if (m.kind == K::Integer) return tick("rt::integer(" + quote(m.name) + ")");
  if (m.kind == K::Boolean) return tick("rt::Value(" + m.name + ")");
  if (m.kind == K::Operand) return tick("ctx.operand(" + num(m.index) + ")");
  if (m.kind == K::Attribute) return tick("ctx.field(" + num(m.index) + ")");
  if (m.kind == K::Group) return tick("ctx.group(" + num(m.index) + ")");
  std::vector<std::string> args;
  for (const auto& a : m.args) args.push_back(axis(*a));
  if (m.kind == K::List) return tick(list(args));
  auto lazy = logical(m.name, args);
  if (!lazy.empty()) return tick(lazy);
  if (auto* entry = core::builtins::resolveAxisBuiltin(m.name))
    return tick(std::visit(
        [&](auto target) -> std::string {
          using T = decltype(target);
          if constexpr (std::is_same_v<T, core::builtins::Builtin>)
            return call("builtins::call",
                        core::builtins::signature(target).name, args, false);
          else if constexpr (std::is_same_v<T, core::intrinsics::Intrinsic>)
            return call("intrinsics::call",
                        core::intrinsics::signature(target).name, args, false);
          else {
            using Q = core::builtins::AxisQuery;
            if (target == Q::Shape) return args.at(0);
            auto len = call("builtins::call", "len", args, false);
            return target == Q::Rank
                       ? len
                       : call("builtins::call", "range", {len}, false);
          }
        },
        entry->target));
  return tick(call("builtins::operate", m.name, args, false));
}
std::string value(const core::AttributeValue& a) {
  using K = core::AttributeValue::Kind;
  switch (a.kind) {
    case K::kInteger:
      return "rt::integer(" + quote(a.text) + ")";
    case K::kBool:
      return "rt::Value(" + a.text + ")";
    case K::kString:
    case K::kEnum:
      return text(a.text);
    case K::kNone:
      return "rt::Value()";
    case K::kList: {
      std::string out = "rt::Value(rt::Value::List{";
      for (const auto& e : a.elements) out += value(e) + ",";
      return out + "})";
    }
  }
  throw std::logic_error("invalid attribute value");
}
}  // namespace tepl::codegen::cpp
