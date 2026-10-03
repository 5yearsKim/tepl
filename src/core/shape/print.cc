#include "src/core/shape/print.h"

#include <type_traits>

#include "src/core/ir.h"

namespace tepl::core::shape {
namespace {
void expression(std::ostream& out, const Expr& expr, const Program& shape,
                const core::Program& program) {
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, SymbolRef>) {
          out << "Symbol(#" << value.symbol.value << ')';
        } else if constexpr (std::is_same_v<T, Integer>) {
          out << "Integer(" << value.spelling << ')';
        } else if constexpr (std::is_same_v<T, bool>) {
          out << (value ? "true" : "false");
        } else if constexpr (std::is_same_v<T, AttributeRef>) {
          out << "Attribute(#" << value.schema.value << '.' << value.field
              << " "
              << program.attribute_schemas.at(value.schema.value)
                     .fields.at(value.field)
                     .name
              << ')';
        } else if constexpr (std::is_same_v<T, Call>) {
          out << "Builtin(" << builtinName(value.builtin);
          for (const auto& argument : value.arguments) {
            out << ", ";
            expression(out, *argument, shape, program);
          }
          out << ')';
        } else if constexpr (std::is_same_v<T, Unary>) {
          out << "Unary(" << spelling(value.op) << ", ";
          expression(out, *value.operand, shape, program);
          out << ')';
        } else if constexpr (std::is_same_v<T, Binary>) {
          out << "Binary(" << spelling(value.op) << ", ";
          expression(out, *value.lhs, shape, program);
          out << ", ";
          expression(out, *value.rhs, shape, program);
          out << ')';
        } else if constexpr (std::is_same_v<T, List>) {
          out << "List(";
          for (std::size_t i = 0; i < value.elements.size(); ++i) {
            if (i) out << ", ";
            expression(out, *value.elements[i], shape, program);
          }
          out << ')';
        } else if constexpr (std::is_same_v<T, Index>) {
          out << "Index(";
          expression(out, *value.value, shape, program);
          out << ", ";
          expression(out, *value.index, shape, program);
          out << ')';
        } else if constexpr (std::is_same_v<T, Conditional>) {
          out << "If(";
          expression(out, *value.condition, shape, program);
          out << ", ";
          expression(out, *value.then_value, shape, program);
          out << ", ";
          expression(out, *value.else_value, shape, program);
          out << ')';
        } else if constexpr (std::is_same_v<T, Comprehension>) {
          out << "Comprehension(#" << value.variable.value << ", ";
          expression(out, *value.iterable, shape, program);
          out << ", ";
          expression(out, *value.element, shape, program);
          out << ')';
        }
      },
      expr.value);
  out << ':' << typeName(shape.types.at(expr.type.value));
}
}  // namespace

void print(std::ostream& out, const Program& shape,
           const core::Program& program) {
  out << "    shape\n";
  for (const auto& parameter : shape.parameters) {
    const auto& symbol = shape.symbols.at(parameter.symbol.value);
    out << "      parameter #" << symbol.id.value << ' ' << symbol.name << ": "
        << typeName(shape.types.at(symbol.type.value))
        << " operand=" << parameter.operand;
    if (parameter.variadic) out << "...";
    out << '\n';
  }
  for (const auto& statement : shape.statements) {
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, Let>) {
            const auto& symbol = shape.symbols.at(value.symbol.value);
            out << "      let #" << symbol.id.value << ' ' << symbol.name
                << ": " << typeName(shape.types.at(symbol.type.value)) << " = ";
            expression(out, *value.value, shape, program);
          } else {
            out << "      assert ";
            expression(out, *value.condition, shape, program);
          }
        },
        statement.value);
    out << '\n';
  }
  out << "      yield ";
  expression(out, *shape.result.value, shape, program);
  out << '\n';
}
}  // namespace tepl::core::shape
