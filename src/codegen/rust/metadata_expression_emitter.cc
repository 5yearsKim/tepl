#include "src/codegen/rust/metadata_expression_emitter.h"

#include "src/codegen/rust/builtin_emitter.h"
#include "src/codegen/rust/names.h"

namespace tepl::codegen::rust {
namespace {
namespace metadata = core::metadata;

std::string integer(std::string spelling) {
  bool negative = spelling.starts_with('-');
  if (negative || spelling.starts_with('+')) spelling.erase(0, 1);
  const auto first = spelling.find_first_not_of('0');
  spelling = first == std::string::npos ? "0" : spelling.substr(first);
  return (negative ? "-" : "") + spelling + "_i128";
}

class ExpressionEmitter {
 public:
  explicit ExpressionEmitter(const metadata::Program& program)
      : program_(program) {}

  std::string emit(const metadata::Expr& expression) const {
    return std::visit(
        [&](const auto& value) { return lower(value, expression); },
        expression.value);
  }

 private:
  std::string lower(const metadata::SymbolRef& value,
                    const metadata::Expr& expr) const {
    return metadataSymbol(value.symbol) +
           (program_.types.at(expr.type.value).list_depth ? ".clone()" : "");
  }
  std::string lower(const metadata::Integer& value,
                    const metadata::Expr&) const {
    return integer(value.spelling);
  }
  std::string lower(core::DType value, const metadata::Expr&) const {
    return dtype(value);
  }
  std::string lower(bool value, const metadata::Expr&) const {
    return value ? "true" : "false";
  }
  std::string lower(const metadata::AttributeRef& value,
                    const metadata::Expr& expr) const {
    return "attribute_" + std::to_string(value.field) +
           (program_.types.at(expr.type.value).list_depth ? ".clone()" : "");
  }
  std::string lower(const metadata::Unary& value, const metadata::Expr&) const {
    const auto operand = emit(*value.operand);
    switch (value.op) {
      case UnaryOp::kPlus:
        return "(" + operand + ")";
      case UnaryOp::kNegate:
        return "super::super::builtins::common::sub(0_i128, " + operand + ")?";
      case UnaryOp::kLogicalNot:
        return "!(" + operand + ")";
    }
    return {};
  }
  std::string lower(const metadata::Binary& value,
                    const metadata::Expr&) const {
    const auto lhs = emit(*value.lhs), rhs = emit(*value.rhs);
    const char* arithmetic = nullptr;
    switch (value.op) {
      case BinaryOp::kAdd:
        arithmetic = "add";
        break;
      case BinaryOp::kSubtract:
        arithmetic = "sub";
        break;
      case BinaryOp::kMultiply:
        arithmetic = "mul";
        break;
      case BinaryOp::kDivide:
        arithmetic = "div";
        break;
      case BinaryOp::kRemainder:
        arithmetic = "rem";
        break;
      default:
        break;
    }
    if (arithmetic)
      return "super::super::builtins::common::" + std::string(arithmetic) +
             "(" + lhs + ", " + rhs + ")?";
    return "(" + lhs + " " + std::string(spelling(value.op)) + " " + rhs + ")";
  }
  std::string lower(const metadata::List& value,
                    const metadata::Expr& expr) const {
    auto element = program_.types.at(expr.type.value);
    --element.list_depth;
    std::string result = "Vec::<" + metadataType(element) + ">::from([";
    for (std::size_t i = 0; i < value.elements.size(); ++i) {
      if (i) result += ", ";
      result += emit(*value.elements[i]);
    }
    return result + "])";
  }
  std::string lower(const metadata::Index& value, const metadata::Expr&) const {
    return "super::super::builtins::common::index(&(" + emit(*value.value) +
           "), " + emit(*value.index) + ")?.clone()";
  }
  std::string lower(const metadata::Conditional& value,
                    const metadata::Expr&) const {
    return "(if " + emit(*value.condition) + " { " + emit(*value.then_value) +
           " } else { " + emit(*value.else_value) + " })";
  }
  std::string lower(const metadata::Comprehension& value,
                    const metadata::Expr& expr) const {
    auto element = program_.types.at(expr.type.value);
    --element.list_depth;
    return "(" + emit(*value.iterable) + ").into_iter().map(|" +
           metadataSymbol(value.variable) +
           "| -> super::super::builtins::BuiltinResult<" +
           metadataType(element) + "> { Ok(" + emit(*value.element) +
           ") }).collect::<super::super::builtins::BuiltinResult<" +
           metadataType(program_.types.at(expr.type.value)) + ">>()?";
  }
  std::string lower(const metadata::Call& value, const metadata::Expr&) const {
    std::vector<std::string> arguments;
    for (const auto& argument : value.arguments)
      arguments.push_back(emit(*argument));
    return emitBuiltin(value.builtin, arguments, "super::super::builtins",
                       true);
  }
  const metadata::Program& program_;
};
}  // namespace

std::string metadataType(core::metadata::Type type) {
  std::string result = type.kind == core::metadata::Type::Kind::kInteger
                           ? "i128"
                       : type.kind == core::metadata::Type::Kind::kBoolean
                           ? "bool"
                           : "super::super::DType";
  for (std::size_t i = 0; i < type.list_depth; ++i)
    result = "Vec<" + result + ">";
  return result;
}
std::string metadataSymbol(core::metadata::SymbolId id) {
  return "symbol_" + std::to_string(id.value);
}
std::string metadataExpression(const core::metadata::Program& program,
                               const core::metadata::Expr& expression) {
  return ExpressionEmitter(program).emit(expression);
}
}  // namespace tepl::codegen::rust
