#include "src/codegen/cpp/shape_expression_emitter.h"

#include <type_traits>

#include "src/codegen/cpp/builtin_emitter.h"
#include "src/codegen/cpp/names.h"

namespace tepl::codegen::cpp {
std::string shapeType(core::shape::Type type, const std::string& root) {
  auto result = type.kind == core::shape::Type::Kind::kBoolean
                    ? "bool"
                    : root + "::builtins::Integer";
  for (std::size_t i = 0; i < type.list_depth; ++i)
    result = "::std::vector<" + result + ">";
  return result;
}
std::string shapeSymbol(core::shape::SymbolId id) {
  return "symbol_" + std::to_string(id.value);
}
std::string shapeExpression(const core::shape::Program& program,
                            const core::shape::Expr& expr,
                            const std::string& root) {
  namespace s = core::shape;
  auto runtime = root + "::builtins";
  const auto emit = [&](const s::ExprPtr& e) {
    return shapeExpression(program, *e, root);
  };
  return std::visit(
      [&](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, s::SymbolRef>)
          return shapeSymbol(value.symbol);
        else if constexpr (std::is_same_v<T, s::AttributeRef>)
          return "attribute_" + std::to_string(value.field);
        else if constexpr (std::is_same_v<T, s::Integer>)
          return runtime + "::common::integer(" + quote(value.spelling) + ")";
        else if constexpr (std::is_same_v<T, bool>)
          return value ? "true" : "false";
        else if constexpr (std::is_same_v<T, s::Call>) {
          std::vector<std::string> args;
          for (const auto& arg : value.arguments) args.push_back(emit(arg));
          return emitBuiltin(value.builtin, args, runtime, true);
        } else if constexpr (std::is_same_v<T, s::Unary>) {
          auto operand = emit(value.operand);
          if (value.op == UnaryOp::kPlus) return "(" + operand + ")";
          if (value.op == UnaryOp::kLogicalNot) return "!(" + operand + ")";
          return runtime + "::take(" + runtime + "::common::neg(" + operand +
                 "))";
        } else if constexpr (std::is_same_v<T, s::Binary>) {
          auto lhs = emit(value.lhs), rhs = emit(value.rhs);
          if (category(value.op) == BinaryCategory::kLogical)
            return "(" + lhs + " " + std::string(spelling(value.op)) + " " +
                   rhs + ")";
          std::string operation;
          switch (value.op) {
            case BinaryOp::kAdd:
              operation = "add";
              break;
            case BinaryOp::kSubtract:
              operation = "sub";
              break;
            case BinaryOp::kMultiply:
              operation = "mul";
              break;
            case BinaryOp::kDivide:
              operation = "div";
              break;
            case BinaryOp::kRemainder:
              operation = "rem";
              break;
            default:
              break;
          }
          auto body =
              operation.empty()
                  ? "(left " + std::string(spelling(value.op)) + " right)"
                  : runtime + "::take(" + runtime + "::common::" + operation +
                        "(left,right))";
          return "([&]() { auto left=" + lhs + "; auto right=" + rhs +
                 "; return " + body + "; }())";
        } else if constexpr (std::is_same_v<T, s::List>) {
          std::string out =
              shapeType(program.types.at(expr.type.value), root) + "{";
          for (const auto& element : value.elements) out += emit(element) + ",";
          return out + "}";
        } else if constexpr (std::is_same_v<T, s::Index>)
          return "([&]() { auto values=" + emit(value.value) +
                 "; auto axis=" + emit(value.index) + "; return " + runtime +
                 "::take(" + runtime + "::common::index(values,axis)); }())";
        else if constexpr (std::is_same_v<T, s::Conditional>)
          return "(" + emit(value.condition) + " ? " + emit(value.then_value) +
                 " : " + emit(value.else_value) + ")";
        else
          return "([&]() { " +
                 shapeType(program.types.at(expr.type.value), root) +
                 " output; auto iterable=" + emit(value.iterable) +
                 "; for (const auto " + shapeSymbol(value.variable) +
                 " : iterable) output.push_back(" + emit(value.element) +
                 "); return output; }())";
      },
      expr.value);
}
}  // namespace tepl::codegen::cpp
