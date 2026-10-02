#include "src/codegen/rust/expression_emitter.h"

#include <type_traits>

#include "src/codegen/rust/names.h"

namespace tepl::codegen::rust {
namespace {
std::string emit(const core::Program& program,
                 const core::TypedExpr& expression) {
  const auto& result_type = program.types.at(expression.type.value);
  std::string result = std::visit(
      [&](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, core::CaptureRef>)
          return "ctx.tensor(" +
                 quote("?c" + std::to_string(value.capture.value)) +
                 ".parse::<Var>().expect(\"generated capture ID\"))?";
        else if constexpr (std::is_same_v<T, core::DimensionRef>) {
          return "dimensions." +
                 std::string(result_type.kind == core::TypeKind::kIndexList
                                 ? "sequence("
                                 : "dimension(") +
                 std::to_string(value.dimension.value) + ")?" +
                 (result_type.kind == core::TypeKind::kIndexList ? ".to_vec()"
                                                                 : "");
        } else if constexpr (std::is_same_v<T, core::DescriptorRef>)
          return descriptor(value.descriptor.value) + ".clone()";
        else if constexpr (std::is_same_v<T, core::NumericConstant>)
          return quote(value.spelling) + ".parse::<" + type(result_type) +
                 ">().ok()?";
        else if constexpr (std::is_same_v<T, bool>)
          return value ? "true" : "false";
        else if constexpr (std::is_same_v<T, core::HostCall>) {
          const auto& host = program.host_functions.at(value.function.value);
          std::string call = "functions." + identifier(host.name) + "(";
          for (std::size_t i = 0; i < value.arguments.size(); ++i) {
            if (i) call += ", ";
            auto argument = emit(program, *value.arguments[i]);
            const auto kind =
                program.types.at(host.signature.arguments.at(i).value).kind;
            if (kind == core::TypeKind::kTensor ||
                kind == core::TypeKind::kDescriptor ||
                kind == core::TypeKind::kIndexList)
              argument = "&(" + argument + ")";
            call += argument;
          }
          return call + ")" + (host.fallible ? "?" : "");
        } else if constexpr (std::is_same_v<T, core::UnaryExpr>) {
          const auto operand = "(" + emit(program, *value.operand) + ")";
          if (value.op == core::UnaryOp::kPlus) return operand;
          if (value.op == core::UnaryOp::kLogicalNot) return "!" + operand;
          return result_type.kind == core::TypeKind::kF64
                     ? "-" + operand
                     : operand + ".checked_neg()?";
        } else {
          const auto lhs = "(" + emit(program, *value.lhs) + ")";
          const auto rhs = "(" + emit(program, *value.rhs) + ")";
          if (category(value.op) == BinaryCategory::kArithmetic &&
              result_type.kind != core::TypeKind::kF64) {
            std::string method;
            switch (value.op) {
              case core::BinaryOp::kAdd:
                method = "checked_add";
                break;
              case core::BinaryOp::kSubtract:
                method = "checked_sub";
                break;
              case core::BinaryOp::kMultiply:
                method = "checked_mul";
                break;
              case core::BinaryOp::kDivide:
                method = "checked_div";
                break;
              case core::BinaryOp::kRemainder:
                method = "checked_rem";
                break;
              default:
                break;
            }
            return lhs + "." + method + "(" + rhs + ")?";
          }
          // Keep the RHS inside &&/|| so fallible calls retain short-circuit
          // order.
          return "(" + lhs + " " + std::string(spelling(value.op)) + " " + rhs +
                 ")";
        }
      },
      expression.value);
  if (result_type.kind == core::TypeKind::kF64)
    result = "finite(" + result + ")?";
  if (result_type.kind == core::TypeKind::kDescriptor && result_type.schema)
    result = "(" + result + ").checked_schema(" +
             std::to_string(result_type.schema->value) + ")?";
  return result;
}
}  // namespace
std::string emitExpression(const core::Program& program,
                           const core::TypedExpr& expression) {
  return emit(program, expression);
}
}  // namespace tepl::codegen::rust
