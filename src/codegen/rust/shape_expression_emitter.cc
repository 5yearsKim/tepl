#include "src/codegen/rust/shape_expression_emitter.h"

namespace tepl::codegen::rust {
namespace {
namespace shape = core::shape;

std::string integer(std::string spelling) {
  bool negative = spelling.starts_with('-');
  if (negative || spelling.starts_with('+')) spelling.erase(0, 1);
  const auto first = spelling.find_first_not_of('0');
  spelling = first == std::string::npos ? "0" : spelling.substr(first);
  return (negative ? "-" : "") + spelling + "_i128";
}

class ExpressionEmitter {
 public:
  explicit ExpressionEmitter(const shape::Program& program)
      : program_(program) {}

  std::string emit(const shape::Expr& expression) const {
    return std::visit(
        [&](const auto& value) { return lower(value, expression); },
        expression.value);
  }

 private:
  std::string lower(const shape::SymbolRef& value,
                    const shape::Expr& expr) const {
    return shapeSymbol(value.symbol) +
           (program_.types.at(expr.type.value).list_depth ? ".clone()" : "");
  }
  std::string lower(const shape::Integer& value, const shape::Expr&) const {
    return integer(value.spelling);
  }
  std::string lower(bool value, const shape::Expr&) const {
    return value ? "true" : "false";
  }
  std::string lower(const shape::AttributeRef& value,
                    const shape::Expr& expr) const {
    return "attribute_" + std::to_string(value.field) +
           (program_.types.at(expr.type.value).list_depth ? ".clone()" : "");
  }
  std::string lower(const shape::Unary& value, const shape::Expr&) const {
    const auto operand = emit(*value.operand);
    switch (value.op) {
      case UnaryOp::kPlus:
        return "(" + operand + ")";
      case UnaryOp::kNegate:
        return "b::sub(0_i128, " + operand + ")?";
      case UnaryOp::kLogicalNot:
        return "!(" + operand + ")";
    }
    return {};
  }
  std::string lower(const shape::Binary& value, const shape::Expr&) const {
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
      return "b::" + std::string(arithmetic) + "(" + lhs + ", " + rhs + ")?";
    return "(" + lhs + " " + std::string(spelling(value.op)) + " " + rhs + ")";
  }
  std::string lower(const shape::List& value, const shape::Expr& expr) const {
    auto element = program_.types.at(expr.type.value);
    --element.list_depth;
    std::string result = "Vec::<" + shapeType(element) + ">::from([";
    for (std::size_t i = 0; i < value.elements.size(); ++i) {
      if (i) result += ", ";
      result += emit(*value.elements[i]);
    }
    return result + "])";
  }
  std::string lower(const shape::Index& value, const shape::Expr&) const {
    return "b::index(&(" + emit(*value.value) + "), " + emit(*value.index) +
           ")?.clone()";
  }
  std::string lower(const shape::Conditional& value, const shape::Expr&) const {
    return "(if " + emit(*value.condition) + " { " + emit(*value.then_value) +
           " } else { " + emit(*value.else_value) + " })";
  }
  std::string lower(const shape::Comprehension& value,
                    const shape::Expr& expr) const {
    auto element = program_.types.at(expr.type.value);
    --element.list_depth;
    return "(" + emit(*value.iterable) + ").into_iter().map(|" +
           shapeSymbol(value.variable) + "| -> b::ShapeResult<" +
           shapeType(element) + "> { Ok(" + emit(*value.element) +
           ") }).collect::<b::ShapeResult<" +
           shapeType(program_.types.at(expr.type.value)) + ">>()?";
  }
  std::string lower(const shape::Call& value, const shape::Expr&) const {
    using B = shape::Builtin;
    const auto arg = [&](std::size_t i) {
      return emit(*value.arguments.at(i));
    };
    const auto ref = [&](std::size_t i) { return "&(" + arg(i) + ")"; };
    const std::string fn =
        "b::" + std::string(shape::builtinName(value.builtin));
    switch (value.builtin) {
      case B::kLen:
        return "(" + fn + "(" + ref(0) + ") as i128)";
      case B::kRange:
        return "b::integers(&b::range(" + arg(0) + ")?)";
      case B::kConcat: {
        std::string result = fn + "(&[";
        for (std::size_t i = 0; i < value.arguments.size(); ++i) {
          if (i) result += ", ";
          result += ref(i);
        }
        return result + "])?";
      }
      case B::kGather:
        return fn + "(" + ref(0) + ", " + ref(1) + ")?";
      case B::kExclude:
      case B::kIsDisjoint:
        return fn + "(" + ref(0) + ", " + ref(1) + ")";
      case B::kSlice:
        return fn + "(" + ref(0) + ", " + arg(1) + ", " + arg(2) + ")?";
      case B::kReplace:
        return fn + "(" + ref(0) + ", " + arg(1) + ", " + arg(2) + ")?";
      case B::kSum:
      case B::kProduct:
        return fn + "(" + ref(0) + ")?";
      case B::kAll:
      case B::kAny:
        return fn + "(" + ref(0) + ")";
      case B::kContains:
        return fn + "(" + ref(0) + ", " + ref(1) + ")";
      case B::kIsValidAxisList:
        // Negative and unrepresentable ranks cannot contain any valid axes.
        return "{ let axes = " + arg(0) + "; let rank = " + arg(1) +
               "; usize::try_from(rank).is_ok_and(|rank| " + fn +
               "(&axes, rank)) }";
      case B::kBroadcastShape:
        return fn + "(" + ref(0) + ", " + ref(1) + ")?";
      case B::kMin:
      case B::kMax:
        return fn + "(" + arg(0) + ", " + arg(1) + ")";
      case B::kFloorDiv:
      case B::kCeilDiv:
        return fn + "(" + arg(0) + ", " + arg(1) + ")?";
    }
    return {};
  }
  const shape::Program& program_;
};
}  // namespace

std::string shapeType(core::shape::Type type) {
  std::string result =
      type.kind == core::shape::Type::Kind::kInteger ? "i128" : "bool";
  for (std::size_t i = 0; i < type.list_depth; ++i)
    result = "Vec<" + result + ">";
  return result;
}
std::string shapeSymbol(core::shape::SymbolId id) {
  return "symbol_" + std::to_string(id.value);
}
std::string shapeExpression(const core::shape::Program& program,
                            const core::shape::Expr& expression) {
  return ExpressionEmitter(program).emit(expression);
}
}  // namespace tepl::codegen::rust
