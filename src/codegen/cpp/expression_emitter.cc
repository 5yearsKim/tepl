#include "src/codegen/cpp/expression_emitter.h"

#include <type_traits>

#include "src/codegen/cpp/builtin_emitter.h"

namespace tepl::codegen::cpp {
namespace {
std::string checkedType(const Names& names, const core::Type& t) {
  auto result = type(t);
  if (t.kind == core::TypeKind::kTensor ||
      t.kind == core::TypeKind::kDescriptor || t.kind == core::TypeKind::kDType)
    result = names.root + "::" + result;
  return result;
}
std::string arithmetic(BinaryOp op) {
  switch (op) {
    case BinaryOp::kAdd:
      return "add";
    case BinaryOp::kSubtract:
      return "sub";
    case BinaryOp::kMultiply:
      return "mul";
    case BinaryOp::kDivide:
      return "div";
    case BinaryOp::kRemainder:
      return "rem";
    default:
      return {};
  }
}
}  // namespace
std::string emitExpression(const core::Program& program, const Names& names,
                           const core::TypedExpr& expr) {
  auto runtime = names.root + "::builtins";
  const auto emit = [&](const core::TypedExprPtr& e) {
    return emitExpression(program, names, *e);
  };
  const auto& result_type = program.types.at(expr.type.value);
  auto result = std::visit(
      [&](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, core::CaptureRef>)
          return runtime + "::require(ctx.tensor(" +
                 std::to_string(value.capture.value) + "))";
        else if constexpr (std::is_same_v<T, core::DType>)
          return names.root + "::" + dtype(value);
        else if constexpr (std::is_same_v<T, core::DTypeVariableRef>)
          return runtime + "::require(dimensions.dtype(" +
                 std::to_string(value.variable.value) + "))";
        else if constexpr (std::is_same_v<T, core::DescriptorFieldRef>)
          return "([&]() { auto descriptor=" + runtime +
                 "::require(ctx.attrs(" +
                 std::to_string(value.descriptor.value) +
                 ")); auto fields=" + names.root + "::attrs_schema_" +
                 std::to_string(value.schema.value) +
                 "(descriptor); if (!fields) throw " + runtime +
                 "::BuiltinError(\"descriptor schema mismatch\"); return "
                 "fields->" +
                 names.field(value.schema, value.field) + "; }())";
        else if constexpr (std::is_same_v<T, core::DimensionRef>)
          return runtime + "::require(dimensions." +
                 (result_type.kind == core::TypeKind::kIndexList
                      ? "sequence("
                      : "dimension(") +
                 std::to_string(value.dimension.value) + "))";
        else if constexpr (std::is_same_v<T, core::DescriptorRef>)
          return runtime + "::require(ctx.attrs(" +
                 std::to_string(value.descriptor.value) + "))";
        else if constexpr (std::is_same_v<T, core::NumericConstant>) {
          if (result_type.kind == core::TypeKind::kF64) {
            auto number = value.spelling;
            std::string sign;
            if (number.starts_with('-') || number.starts_with('+')) {
              sign = number.substr(0, 1);
              number.erase(0, 1);
            }
            auto first = number.find_first_not_of('0');
            if (first == std::string::npos)
              number = "0";
            else if (first && number[first] == '.')
              number.erase(0, first - 1);
            else
              number.erase(0, first);
            if (number.find('.') == std::string::npos) number += ".0";
            return "double(" + sign + number + ")";
          }
          return "static_cast<" + checkedType(names, result_type) + ">(" +
                 runtime + "::common::integer(" + quote(value.spelling) + "))";
        } else if constexpr (std::is_same_v<T, bool>)
          return value ? "true" : "false";
        else if constexpr (std::is_same_v<T, core::BuiltinCall>) {
          std::vector<std::string> args;
          for (const auto& arg : value.arguments) args.push_back(emit(arg));
          return emitBuiltin(value.builtin, args, runtime, false);
        } else if constexpr (std::is_same_v<T, core::HostCall>) {
          const auto& fn = program.host_functions.at(value.function.value);
          std::string out = "([&]() { ";
          for (std::size_t i = 0; i < value.arguments.size(); ++i)
            out += "auto arg_" + std::to_string(i) + " = " +
                   emit(value.arguments[i]) + "; ";
          std::string call = "functions." + names.host(fn.id) + "(";
          for (std::size_t i = 0; i < value.arguments.size(); ++i) {
            if (i) call += ",";
            call += "arg_" + std::to_string(i);
          }
          call += ")";
          if (fn.fallible) call = runtime + "::require(" + call + ")";
          return out + "return " + call + "; }())";
        } else if constexpr (std::is_same_v<T, core::UnaryExpr>) {
          auto operand = emit(value.operand);
          if (value.op == UnaryOp::kPlus) return "(" + operand + ")";
          if (value.op == UnaryOp::kLogicalNot) return "!(" + operand + ")";
          if (result_type.kind == core::TypeKind::kF64)
            return "-(" + operand + ")";
          return runtime + "::take(" + runtime + "::common::neg(" + operand +
                 "))";
        } else {
          auto lhs = emit(value.lhs), rhs = emit(value.rhs);
          if (category(value.op) == BinaryCategory::kLogical)
            return "(" + lhs + " " + std::string(spelling(value.op)) + " " +
                   rhs + ")";
          auto operation = arithmetic(value.op);
          auto body =
              operation.empty() || result_type.kind == core::TypeKind::kF64
                  ? "(left " + std::string(spelling(value.op)) + " right)"
                  : runtime + "::take(" + runtime + "::common::" + operation +
                        "(left,right))";
          return "([&]() { auto left=" + lhs + "; auto right=" + rhs +
                 "; return " + body + "; }())";
        }
      },
      expr.value);
  if (result_type.kind == core::TypeKind::kF64)
    result =
        runtime + "::take(" + runtime + "::common::finite(" + result + "))";
  if (result_type.kind == core::TypeKind::kDescriptor && result_type.schema)
    result = runtime + "::require((" + result + ").checked_schema(" +
             std::to_string(result_type.schema->value) + "))";
  return result;
}
}  // namespace tepl::codegen::cpp
