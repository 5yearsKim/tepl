#include "src/codegen/rust/expression_emitter.h"

#include <stdexcept>
#include <type_traits>

#include "src/codegen/rust/names.h"
#include "src/shared/builtins/axes.h"
#include "src/shared/builtins/intrinsics/catalog.h"

namespace tepl::codegen::rust {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
std::string boolean(bool b) { return b ? "true" : "false"; }
std::string text(const std::string& s) {
  return "Value::String(" + quote(s) + ".into())";
}
std::string tick(std::string s) { return "{ ctx.budget.tick()?; " + s + " }"; }
std::string call(std::string_view module, std::string_view name,
                 const std::vector<std::string>& args, bool rule) {
  std::string xs = "vec![";
  for (const auto& a : args) xs += a + ",";
  return "{ let args = " + xs + "]; rt::" + std::string(module) + "(" +
         quote(name) + ", args, " + (rule ? "&mut ctx.budget" : "ctx.budget") +
         ")? }";
}
std::string logical(std::string_view op, const std::vector<std::string>& args) {
  if (op == "&&" || op == "||")
    return "Value::Boolean((" + args.at(0) + ").boolean()? " + std::string(op) +
           " (" + args.at(1) + ").boolean()?)";
  if (op == "if")
    return "if (" + args.at(0) + ").boolean()? { " + args.at(1) + " } else { " +
           args.at(2) + " }";
  return {};
}
std::string list(const std::vector<std::string>& args) {
  std::string result = "{ let xs = vec![";
  for (const auto& a : args) result += a + ",";
  return result + "]; ctx.budget.size(xs.len())?; Value::List(xs) }";
}
std::string comprehension(const std::string& iterable,
                          const std::string& element, std::size_t id,
                          bool rule) {
  const std::string locals = rule ? "ctx.bindings.locals" : "ctx.locals";
  return "{ let xs = (" + iterable +
         ").list()?.to_vec(); let saved = " + locals +
         ".clone(); "
         "let result = (|| -> rt::EvalResult<Value> { let mut out = "
         "Vec::new(); for item in xs { ctx.budget.tick()?; " +
         locals + ".insert(" + num(id) + ", item); out.push(" + element +
         "); } Ok(Value::List(out)) })(); " + locals + " = saved; result? }";
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
      return "Descriptor(" + num(t.schema.value().value) + ")";
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
          return "ctx.capture(" + num(v.capture.value) + ")?";
        else if constexpr (std::is_same_v<T, DimensionRef>)
          return "ctx.dimension(" + num(v.dimension.value) + ")?";
        else if constexpr (std::is_same_v<T, DescriptorRef>)
          return "ctx.descriptor(" + num(v.descriptor.value) + ")?";
        else if constexpr (std::is_same_v<T, DescriptorFieldRef>)
          return "ctx.descriptor(" + num(v.descriptor.value) + ")?.field(" +
                 num(v.field) + ")?";
        else if constexpr (std::is_same_v<T, DTypeVariableRef>)
          return "ctx.dtype_variable(" + num(v.variable.value) + ")?";
        else if constexpr (std::is_same_v<T, LocalRef>)
          return "ctx.local(" + num(v.local.value) + ")?";
        else if constexpr (std::is_same_v<T, DType>)
          return text(std::string(dtypeName(v)));
        else if constexpr (std::is_same_v<T, NumericConstant>)
          return "rt::values::" +
                 std::string(type.kind == TypeKind::kF64 ? "floating"
                                                         : "integer") +
                 "(" + quote(v.spelling) + ")?";
        else if constexpr (std::is_same_v<T, bool>)
          return "Value::Boolean(" + boolean(v) + ")";
        else if constexpr (std::is_same_v<T, StringConstant> ||
                           std::is_same_v<T, EnumConstant>)
          return text(v.value);
        else if constexpr (std::is_same_v<T, NoneConstant>)
          return "Value::None";
        else if constexpr (std::is_same_v<T, OptionalValue>)
          return rule(*v.value);
        else if constexpr (std::is_same_v<T, BuiltinCall>) {
          std::vector<std::string> args;
          for (const auto& a : v.arguments) args.push_back(rule(*a));
          return call("builtins::call", builtins::signature(v.builtin).name,
                      args, true);
        } else if constexpr (std::is_same_v<T, UnaryExpr>)
          return call("builtins::operator", spelling(v.op), {rule(*v.operand)},
                      true);
        else if constexpr (std::is_same_v<T, BinaryExpr>) {
          std::vector<std::string> args = {rule(*v.lhs), rule(*v.rhs)};
          auto lazy = logical(spelling(v.op), args);
          return lazy.empty()
                     ? call("builtins::operator", spelling(v.op), args, true)
                     : lazy;
        } else if constexpr (std::is_same_v<T, LocalLet>)
          return "{ let value = " + rule(*v.value) +
                 "; ctx.bindings.locals.insert(" + num(v.local.value) +
                 ", value.clone()); value }";
        else if constexpr (std::is_same_v<T, ListExpr>) {
          std::vector<std::string> xs;
          for (const auto& a : v.elements) xs.push_back(rule(*a));
          return list(xs);
        } else if constexpr (std::is_same_v<T, IndexExpr>)
          return "{ let value = " + rule(*v.value) +
                 "; let at = " + rule(*v.index) +
                 "; rt::builtins::index(value, at, &mut ctx.budget)? }";
        else if constexpr (std::is_same_v<T, ConditionalExpr>)
          return logical("if", {rule(*v.condition), rule(*v.then_value),
                                rule(*v.else_value)});
        else if constexpr (std::is_same_v<T, ComprehensionExpr>)
          return comprehension(rule(*v.iterable), rule(*v.element),
                               v.variable.value, true);
        else if constexpr (std::is_same_v<T, DescriptorConstruct>) {
          std::string out =
              "Value::Descriptor { schema: " + num(v.schema.value) +
              ", fields: vec![";
          for (const auto& f : v.fields) out += rule(*f) + ",";
          return out + "] }";
        } else if constexpr (std::is_same_v<T, SymbolicExpr>)
          return symbolic(*v.value);
        else
          static_assert(!sizeof(T), "unhandled executable expression");
      },
      e.value);
  auto depth = type.list_depth + (type.kind == TypeKind::kU64List ? 1 : 0);
  return tick("let value = " + result +
              "; rt::values::checked(value, rt::values::ScalarType::" +
              scalarType(type) + ", " + num(depth) + ", " +
              boolean(type.optional) + ")?");
}
std::string Expressions::symbolic(const core::completed::Symbol& s) const {
  using K = core::completed::Symbol::Kind;
  if (s.kind == K::Integer)
    return tick("rt::values::integer(" + quote(s.name) + ")?");
  if (s.kind == K::Boolean) return tick("Value::Boolean(" + s.name + ")");
  if (s.kind == K::Shape) return tick("ctx.shape(" + num(s.owner) + ")?");
  if (s.kind == K::DType) return tick("ctx.dtype(" + num(s.owner) + ")?");
  if (s.kind == K::Field)
    return tick("ctx.descriptor(" + num(s.owner) + ")?.field(" + num(s.index) +
                ")?");
  if (s.kind == K::AxisGroup)
    return tick("axis_" + num(s.owner) + "_" + num(s.index) + "(ctx)?");
  std::vector<std::string> args;
  for (const auto& a : s.args) args.push_back(symbolic(*a));
  if (s.kind == K::List) return tick(list(args));
  auto lazy = logical(s.name, args);
  if (!lazy.empty()) return tick(lazy);
  if (s.name.starts_with("dtype_")) return tick(text(s.name.substr(6)));
  const auto module = core::intrinsics::resolve(s.name) ? "intrinsics::call"
                      : core::builtins::resolveBuiltin(s.name)
                          ? "builtins::call"
                          : "builtins::operator";
  return tick(call(module, s.name, args, true));
}
std::string Expressions::metadata(const core::metadata::Expr& e) const {
  using namespace core::metadata;
  return tick(std::visit(
      [&](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, SymbolRef>)
          return "ctx.local(" + num(v.symbol.value) + ")?";
        else if constexpr (std::is_same_v<T, Integer>)
          return "rt::values::integer(" + quote(v.spelling) + ")?";
        else if constexpr (std::is_same_v<T, bool>)
          return "Value::Boolean(" + boolean(v) + ")";
        else if constexpr (std::is_same_v<T, core::DType>)
          return text(std::string(core::dtypeName(v)));
        else if constexpr (std::is_same_v<T, AttributeRef>)
          return "ctx.field(" + num(v.field) + ")?";
        else if constexpr (std::is_same_v<T, Call>) {
          std::vector<std::string> xs;
          for (const auto& a : v.arguments) xs.push_back(metadata(*a));
          return call("builtins::call",
                      core::builtins::signature(v.builtin).name, xs, false);
        } else if constexpr (std::is_same_v<T, Unary>)
          return call("builtins::operator", spelling(v.op),
                      {metadata(*v.operand)}, false);
        else if constexpr (std::is_same_v<T, Binary>) {
          std::vector<std::string> xs = {metadata(*v.lhs), metadata(*v.rhs)};
          auto lazy = logical(spelling(v.op), xs);
          return lazy.empty()
                     ? call("builtins::operator", spelling(v.op), xs, false)
                     : lazy;
        } else if constexpr (std::is_same_v<T, List>) {
          std::vector<std::string> xs;
          for (const auto& a : v.elements) xs.push_back(metadata(*a));
          return list(xs);
        } else if constexpr (std::is_same_v<T, Index>)
          return "{ let value = " + metadata(*v.value) +
                 "; let at = " + metadata(*v.index) +
                 "; rt::builtins::index(value, at, ctx.budget)? }";
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
  if (m.kind == K::Integer)
    return tick("rt::values::integer(" + quote(m.name) + ")?");
  if (m.kind == K::Boolean) return tick("Value::Boolean(" + m.name + ")");
  if (m.kind == K::Operand) return tick("ctx.operand(" + num(m.index) + ")?");
  if (m.kind == K::Attribute) return tick("ctx.field(" + num(m.index) + ")?");
  if (m.kind == K::Group) return tick("ctx.group(" + num(m.index) + ")?");
  std::vector<std::string> xs;
  for (const auto& a : m.args) xs.push_back(axis(*a));
  if (m.kind == K::List) return tick(list(xs));
  auto lazy = logical(m.name, xs);
  if (!lazy.empty()) return tick(lazy);
  if (auto* entry = core::builtins::resolveAxisBuiltin(m.name)) {
    return tick(std::visit(
        [&](auto target) -> std::string {
          using T = decltype(target);
          if constexpr (std::is_same_v<T, core::builtins::Builtin>)
            return call("builtins::call",
                        core::builtins::signature(target).name, xs, false);
          else if constexpr (std::is_same_v<T, core::intrinsics::Intrinsic>)
            return call("intrinsics::call",
                        core::intrinsics::signature(target).name, xs, false);
          else {
            using Q = core::builtins::AxisQuery;
            if (target == Q::Shape) return xs.at(0);
            auto len = call("builtins::call", "len", xs, false);
            return target == Q::Rank
                       ? len
                       : call("builtins::call", "range", {len}, false);
          }
        },
        entry->target));
  }
  return tick(call("builtins::operator", m.name, xs, false));
}
std::string value(const core::AttributeValue& a) {
  using K = core::AttributeValue::Kind;
  switch (a.kind) {
    case K::kInteger:
      return "rt::values::integer(" + quote(a.text) + ")?";
    case K::kBool:
      return "Value::Boolean(" + a.text + ")";
    case K::kString:
    case K::kEnum:
      return text(a.text);
    case K::kNone:
      return "Value::None";
    case K::kList: {
      std::string out = "Value::List(vec![";
      for (const auto& e : a.elements) out += value(e) + ",";
      return out + "])";
    }
  }
  throw std::logic_error("invalid attribute value");
}
}  // namespace tepl::codegen::rust
