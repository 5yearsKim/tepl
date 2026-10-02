#include "src/core/print.h"

#include <sstream>
#include <type_traits>

namespace tepl::core {
namespace {

class Printer {
 public:
  explicit Printer(const Program& program) : program_(program) {}

  std::string run() {
    out_ << "CheckedProgram\n";
    for (const auto& op : program_.operations) {
      out_ << "  operation #" << op.id.value << ' ' << op.dialect << '.'
           << op.name << '(';
      for (std::size_t i = 0; i < op.operands.size(); ++i) {
        if (i) out_ << ", ";
        out_ << op.operands[i].name << ": " << type(op.operands[i].type);
        if (op.operands[i].variadic) out_ << "...";
      }
      out_ << ") -> " << type(op.result);
      if (op.attributes) out_ << " attrs=#" << op.attributes->value;
      out_ << '\n';
    }
    for (const auto& schema : program_.attribute_schemas) {
      out_ << "  attr_schema #" << schema.id.value << ' ' << schema.dialect
           << '.' << schema.name << " {";
      for (const auto& field : schema.fields) {
        out_ << ' ' << field.name << ": "
             << (field.kind == AttributeField::Kind::kIndex ? "index"
                                                            : "string");
        if (field.list) out_ << "[]";
        if (field.empty_default) out_ << " = []";
        out_ << ';';
      }
      out_ << " }\n";
    }
    for (const auto& fn : program_.host_functions) {
      out_ << "  host_fn #" << fn.id.value << ' ' << fn.name << '(';
      for (std::size_t i = 0; i < fn.signature.arguments.size(); ++i) {
        if (i) out_ << ", ";
        out_ << type(fn.signature.arguments[i]);
      }
      out_ << ") -> " << type(fn.signature.result)
           << (fn.fallible ? " (fallible)" : "") << '\n';
    }
    for (const auto& rule : program_.rules) printRule(rule);
    return out_.str();
  }

 private:
  std::string type(TypeId id) const {
    return typeName(program_.types.at(id.value));
  }
  void location(const SourceOrigin& origin) {
    const auto& at = origin.definition;
    out_ << at.source_name << ':' << at.span.begin.line << ':'
         << at.span.begin.column;
    for (const auto& expansion : origin.expansions)
      out_ << " expanded at " << expansion.source_name << ':'
           << expansion.span.begin.line << ':' << expansion.span.begin.column;
  }
  void literal(const GraphLiteral& value) {
    out_ << (value.kind == GraphLiteral::Kind::kInteger ? "Integer("
                                                        : "Decimal(")
         << value.spelling;
    if (value.dtype) out_ << ':' << dtypeName(*value.dtype);
    out_ << ')';
  }
  void pattern(const Pattern& node) {
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, CapturePattern>)
            out_ << "Capture(#" << value.capture.value << ')';
          else if constexpr (std::is_same_v<T, GraphLiteral>)
            literal(value);
          else if constexpr (std::is_same_v<T, BindPattern>) {
            out_ << "Bind(#" << value.capture.value << ", ";
            pattern(*value.expression);
            out_ << ')';
          } else {
            operation("MatchOp", value.operation, value.descriptor);
            for (const auto& operand : value.operands) {
              out_ << ", ";
              pattern(*operand);
            }
            out_ << ')';
          }
        },
        node.value);
  }
  void operation(const char* prefix, OpId id,
                 std::optional<DescriptorId> descriptor) {
    const auto& op = program_.operations.at(id.value);
    out_ << prefix << '(' << op.dialect << '.' << op.name << " #" << id.value;
    if (descriptor) out_ << " @#" << descriptor->value;
  }
  void build(const BuildExpr& node) {
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, CaptureRef>)
            out_ << "CaptureRef(#" << value.capture.value << ')';
          else if constexpr (std::is_same_v<T, GraphLiteral>)
            literal(value);
          else {
            operation("BuildOp", value.operation, value.descriptor);
            for (const auto& operand : value.operands) {
              out_ << ", ";
              build(*operand);
            }
            out_ << ')';
          }
        },
        node.value);
  }
  void expression(const TypedExpr& node) {
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, CaptureRef>)
            out_ << "CaptureRef(#" << value.capture.value << ')';
          else if constexpr (std::is_same_v<T, DimensionRef>)
            out_ << "DimensionRef(#" << value.dimension.value << ')';
          else if constexpr (std::is_same_v<T, DescriptorRef>)
            out_ << "DescriptorRef(@#" << value.descriptor.value << ')';
          else if constexpr (std::is_same_v<T, NumericConstant>)
            out_ << value.spelling;
          else if constexpr (std::is_same_v<T, bool>)
            out_ << (value ? "true" : "false");
          else if constexpr (std::is_same_v<T, HostCall>) {
            out_ << program_.host_functions.at(value.function.value).name
                 << " #" << value.function.value << '(';
            for (std::size_t i = 0; i < value.arguments.size(); ++i) {
              if (i) out_ << ", ";
              expression(*value.arguments[i]);
            }
            out_ << ')';
          } else if constexpr (std::is_same_v<T, UnaryExpr>) {
            out_ << spelling(value.op) << '(';
            expression(*value.operand);
            out_ << ')';
          } else {
            out_ << '(';
            expression(*value.lhs);
            out_ << ' ' << spelling(value.op) << ' ';
            expression(*value.rhs);
            out_ << ')';
          }
        },
        node.value);
    out_ << ':' << type(node.type);
  }
  void printRule(const Rule& rule) {
    out_ << "  rule #" << rule.id.value << ' ' << rule.name << " [";
    location(rule.origin);
    out_ << "]\n";
    for (const auto& capture : rule.captures)
      out_ << "    capture #" << capture.id.value << ' ' << capture.name << ": "
           << type(capture.type) << '\n';
    for (const auto& dimension : rule.dimensions)
      out_ << "    dimension #" << dimension.id.value << ' ' << dimension.name
           << ": " << type(dimension.type) << '\n';
    for (const auto& descriptor : rule.descriptors)
      out_ << "    descriptor @#" << descriptor.id.value << ' '
           << descriptor.name << ": " << type(descriptor.type)
           << (descriptor.kind == Descriptor::Kind::kCaptured ? " captured"
                                                              : " derived")
           << '\n';
    for (const auto& constraint : rule.constraints) {
      out_ << "    constraint #" << constraint.capture.value << " dtype=";
      out_ << (constraint.dtype ? dtypeName(*constraint.dtype) : "any")
           << " shape=[";
      for (std::size_t i = 0; i < constraint.shape.size(); ++i) {
        if (i) out_ << ", ";
        const auto& element = constraint.shape[i];
        if (element.symbol)
          out_ << '#' << element.symbol->value;
        else if (element.kind == ShapeElement::Kind::kWildcard)
          out_ << '_';
        if (element.kind == ShapeElement::Kind::kSequence) out_ << "...";
      }
      out_ << "]\n";
    }
    out_ << "    match: ";
    pattern(*rule.lhs);
    out_ << '\n';
    for (const auto& condition : rule.conditions) {
      out_ << "    where: ";
      expression(*condition);
      out_ << '\n';
    }
    for (const auto& derivation : rule.derivations) {
      out_ << "    derive @#" << derivation.target.value << " = ";
      expression(*derivation.value);
      out_ << '\n';
    }
    out_ << "    replacement: ";
    build(*rule.rhs);
    out_ << '\n';
  }
  const Program& program_;
  std::ostringstream out_;
};

}  // namespace

std::string formatProgram(const Program& program) {
  return Printer(program).run();
}

}  // namespace tepl::core
