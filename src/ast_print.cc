#include "src/ast_print.h"

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

namespace tepl {
namespace {

void indent(std::ostream& out, int depth) {
  out << std::string(depth * 2, ' ');
}

std::string formatDimension(const ast::ShapeDimension& dimension) {
  return std::visit(
      [](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, ast::NamedDimension>) {
          return value.name;
        } else if constexpr (std::is_same_v<T, ast::WildcardDimension>) {
          return "_";
        } else {
          return value.name.value_or("") + "...";
        }
      },
      dimension.value);
}

void printGraph(std::ostream& out, const ast::GraphExpr& expression,
                int depth) {
  indent(out, depth);
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, ast::NameRef>) {
          out << "variable " << value.name << '\n';
        } else if constexpr (std::is_same_v<T, ast::BinderRef>) {
          out << "binder reference ?" << value.name << '\n';
        } else if constexpr (std::is_same_v<T, ast::Operator>) {
          out << "operator " << value.name;
          if (value.attribute) out << " @" << value.attribute->name;
          out << '\n';
          for (const auto& operand : value.operands) {
            printGraph(out, *operand, depth + 1);
          }
        } else if constexpr (std::is_same_v<T, ast::Binding>) {
          out << "bind ?" << value.binder.name << '\n';
          printGraph(out, *value.expression, depth + 1);
        } else {
          out << "get[" << value.index.digits << "]\n";
          printGraph(out, *value.tuple, depth + 1);
        }
      },
      expression.value);
}

std::string_view unarySymbol(ast::UnaryOp op) {
  switch (op) {
    case ast::UnaryOp::kPlus:
      return "+";
    case ast::UnaryOp::kNegate:
      return "-";
    case ast::UnaryOp::kLogicalNot:
      return "!";
  }
  return "?";
}

std::string_view binarySymbol(ast::BinaryOp op) {
  switch (op) {
    case ast::BinaryOp::kAdd:
      return "+";
    case ast::BinaryOp::kSubtract:
      return "-";
    case ast::BinaryOp::kMultiply:
      return "*";
    case ast::BinaryOp::kDivide:
      return "/";
    case ast::BinaryOp::kRemainder:
      return "%";
    case ast::BinaryOp::kLess:
      return "<";
    case ast::BinaryOp::kLessEqual:
      return "<=";
    case ast::BinaryOp::kGreater:
      return ">";
    case ast::BinaryOp::kGreaterEqual:
      return ">=";
    case ast::BinaryOp::kEqual:
      return "==";
    case ast::BinaryOp::kNotEqual:
      return "!=";
    case ast::BinaryOp::kLogicalAnd:
      return "&&";
    case ast::BinaryOp::kLogicalOr:
      return "||";
  }
  return "?";
}

void printConstraint(std::ostream& out, const ast::ConstraintExpr& expression,
                     int depth) {
  indent(out, depth);
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, ast::NameRef>) {
          out << "name " << value.name << '\n';
        } else if constexpr (std::is_same_v<T, ast::AttributeRef>) {
          out << "attribute @" << value.name << '\n';
        } else if constexpr (std::is_same_v<T, ast::BinderRef>) {
          out << "binder reference ?" << value.name << '\n';
        } else if constexpr (std::is_same_v<T, ast::IntegerLiteral>) {
          out << "integer " << value.digits << '\n';
        } else if constexpr (std::is_same_v<T, ast::BooleanLiteral>) {
          out << "boolean " << (value.value ? "true" : "false") << '\n';
        } else if constexpr (std::is_same_v<T, ast::Call>) {
          out << "call " << value.callee << '\n';
          for (const auto& argument : value.arguments) {
            printConstraint(out, *argument, depth + 1);
          }
        } else if constexpr (std::is_same_v<T, ast::UnaryExpr>) {
          out << "unary " << unarySymbol(value.op) << '\n';
          printConstraint(out, *value.operand, depth + 1);
        } else {
          out << "binary " << binarySymbol(value.op) << '\n';
          printConstraint(out, *value.lhs, depth + 1);
          printConstraint(out, *value.rhs, depth + 1);
        }
      },
      expression.value);
}

void printDeclaration(std::ostream& out, const ast::Declaration& declaration) {
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, ast::ScalarDecl>) {
          out << "    scalar " << value.name << '\n';
        } else {
          out << "    tensor " << value.name << " [";
          for (std::size_t index = 0; index < value.shape.size(); ++index) {
            if (index != 0) out << ", ";
            out << formatDimension(value.shape[index]);
          }
          out << "]\n";
        }
      },
      declaration.value);
}

}  // namespace

std::string formatAst(const ast::Program& program) {
  std::ostringstream out;
  out << "program " << program.source_name << '\n';
  for (const auto& rule : program.rules) {
    out << "  rule " << rule.name << '\n';
    for (const auto& declaration : rule.declarations) {
      printDeclaration(out, declaration);
    }
    out << "    lhs\n";
    printGraph(out, *rule.lhs, 3);
    out << "    rhs\n";
    printGraph(out, *rule.rhs, 3);
    for (const auto& condition : rule.conditions) {
      out << "    where\n";
      printConstraint(out, *condition, 3);
    }
    for (const auto& derivation : rule.derivations) {
      out << "    derive @" << derivation.target.name << '\n';
      printConstraint(out, *derivation.value, 3);
    }
  }
  return out.str();
}

}  // namespace tepl
