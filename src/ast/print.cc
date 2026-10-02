#include "src/ast/print.h"

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
        } else if constexpr (std::is_same_v<T, ast::IntegerLiteral>) {
          out << "integer " << value.digits;
          if (value.dtype) out << ":" << value.dtype->name;
          out << '\n';
        } else if constexpr (std::is_same_v<T, ast::FloatLiteral>) {
          out << "float " << value.digits;
          if (value.dtype) out << ":" << value.dtype->name;
          out << '\n';
        } else if constexpr (std::is_same_v<T, ast::Operator>) {
          out << "operator " << value.name;
          if (value.attribute) out << " @" << value.attribute->name;
          out << '\n';
          for (const auto& operand : value.operands) {
            printGraph(out, *operand, depth + 1);
          }
        } else if constexpr (std::is_same_v<T, ast::Binding>) {
          out << "let " << value.binder.name << '\n';
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
        } else if constexpr (std::is_same_v<T, ast::IntegerLiteral>) {
          out << "integer " << value.digits;
          if (value.dtype) out << ":" << value.dtype->name;
          out << '\n';
        } else if constexpr (std::is_same_v<T, ast::FloatLiteral>) {
          out << "float " << value.digits;
          if (value.dtype) out << ":" << value.dtype->name;
          out << '\n';
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
          out << "    tensor " << value.name << " ";
          if (value.dtype) out << value.dtype->name;
          out << "[";
          for (std::size_t index = 0; index < value.shape.size(); ++index) {
            if (index != 0) out << ", ";
            out << formatDimension(value.shape[index]);
          }
          out << "]\n";
        }
      },
      declaration.value);
}

void printRule(std::ostream& out, const ast::Rule& rule) {
  out << "  " << (rule.is_abstract ? "abstract rule " : "rule ") << rule.name
      << '\n';
  for (const auto& parameter : rule.parameters) {
    out << "    parameter " << parameter.name << ": "
        << (parameter.kind == ast::RuleParameterKind::kOperation ? "op" : "fn")
        << "<(";
    for (std::size_t index = 0; index < parameter.operand_types.size();
         ++index) {
      if (index != 0) out << ", ";
      out << parameter.operand_types[index].name;
    }
    out << ") -> " << parameter.result_type.name << ">\n";
  }
  if (rule.inheritance) {
    out << "    extends " << rule.inheritance->base << '\n';
    for (const auto& binding : rule.inheritance->bindings) {
      out << "      " << binding.parameter << " = " << binding.value << '\n';
    }
  }
  for (const auto& declaration : rule.declarations) {
    printDeclaration(out, declaration);
  }
  if (rule.lhs) {
    out << "    lhs\n";
    printGraph(out, *rule.lhs, 3);
  }
  if (rule.rhs) {
    out << "    rhs\n";
    printGraph(out, *rule.rhs, 3);
  }
  for (const auto& condition : rule.conditions) {
    out << "    where\n";
    printConstraint(out, *condition, 3);
  }
  for (const auto& derivation : rule.derivations) {
    out << "    derive @" << derivation.target.name << '\n';
    printConstraint(out, *derivation.value, 3);
  }
}

}  // namespace

std::string formatAst(const ast::Program& program) {
  std::ostringstream out;
  out << "program " << program.source_name << '\n';
  for (const auto& imported : program.imports) {
    if (!imported.rules.empty()) {
      out << "  from \"" << imported.path << "\" import {";
      for (std::size_t index = 0; index < imported.rules.size(); ++index) {
        if (index != 0) out << ", ";
        out << imported.rules[index].name;
      }
      out << "}\n";
    } else if (imported.dialect) {
      out << "  from \"" << imported.path << "\" import " << *imported.dialect;
      if (imported.alias && imported.alias != imported.dialect) {
        out << " as " << *imported.alias;
      }
      out << '\n';
    } else {
      out << "  import \"" << imported.path << "\"\n";
    }
  }
  for (const auto& used : program.uses) {
    out << "  use " << used.alias;
    if (!used.operations.empty()) {
      out << "::{";
      for (std::size_t index = 0; index < used.operations.size(); ++index) {
        if (index != 0) out << ", ";
        out << used.operations[index];
      }
      out << '}';
    }
    out << '\n';
  }
  for (const auto& dialect : program.dialects) {
    out << "  dialect " << dialect.name << '\n';
    for (const auto& schema : dialect.schemas) {
      out << "    attrs " << schema.name << '\n';
      for (const auto& field : schema.fields) {
        out << "      " << field.name << ": " << field.type;
        if (field.list) out << "[]";
        if (field.empty_default) out << " = []";
        out << '\n';
      }
    }
    for (const auto& op : dialect.operations) {
      out << "    op " << op.name << '(';
      for (std::size_t index = 0; index < op.operands.size(); ++index) {
        if (index != 0) out << ", ";
        const auto& operand = op.operands[index];
        out << operand.name << ": " << operand.type;
        if (operand.variadic) out << "...";
      }
      out << ") -> " << op.result_type;
      if (op.alias) out << " alias " << *op.alias;
      if (op.attrs) {
        if (const auto* shared = std::get_if<ast::SharedAttrs>(&*op.attrs)) {
          out << " attrs " << shared->name;
        }
      }
      out << '\n';
      if (op.attrs) {
        if (const auto* inline_attrs =
                std::get_if<ast::InlineAttrs>(&*op.attrs)) {
          for (const auto& field : inline_attrs->fields) {
            out << "      " << field.name << ": " << field.type;
            if (field.list) out << "[]";
            if (field.empty_default) out << " = []";
            out << '\n';
          }
        }
      }
    }
  }
  for (const auto& rule : program.rules) {
    printRule(out, rule);
  }
  for (const auto& rule : program.imported_rules) {
    out << "  imported from " << rule.source_name << '\n';
    printRule(out, rule);
  }
  return out.str();
}

}  // namespace tepl
