#include "src/host_codegen.h"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tepl {
namespace {

struct Signature {
  std::vector<std::string> arguments;
  std::string result;
};

using Types = std::map<std::string, std::string>;
using Functions = std::map<std::string, Signature>;

void addDiagnostic(std::vector<HostDiagnostic>& diagnostics,
                   const ast::SourceSpan& span, std::string message) {
  diagnostics.push_back({span, std::move(message)});
}

void collectBoundNames(const ast::GraphExpr& expression, Types& types) {
  if (const auto* binding = std::get_if<ast::Binding>(&expression.value)) {
    types.emplace(binding->binder.name, "&TensorInfo");
    collectBoundNames(*binding->expression, types);
  } else if (const auto* op = std::get_if<ast::Operator>(&expression.value)) {
    for (const auto& operand : op->operands) {
      collectBoundNames(*operand, types);
    }
  } else if (const auto* projection =
                 std::get_if<ast::Projection>(&expression.value)) {
    collectBoundNames(*projection->tuple, types);
  }
}

Types declaredTypes(const ast::Rule& rule) {
  Types types;
  for (const auto& declaration : rule.declarations) {
    if (const auto* tensor = std::get_if<ast::TensorDecl>(&declaration.value)) {
      types[tensor->name] = "&TensorInfo";
      for (const auto& dimension : tensor->shape) {
        if (const auto* named =
                std::get_if<ast::NamedDimension>(&dimension.value)) {
          types[named->name] = "usize";
        } else if (const auto* sequence =
                       std::get_if<ast::SequenceDimension>(&dimension.value);
                   sequence && sequence->name) {
          types[*sequence->name] = "&[usize]";
        }
      }
    } else if (const auto* scalar =
                   std::get_if<ast::ScalarDecl>(&declaration.value)) {
      // `scalar` has no element type yet, so a Rust signature cannot be
      // inferred from the declaration alone.
      types[scalar->name] = "";
    }
  }
  collectBoundNames(*rule.lhs, types);
  collectBoundNames(*rule.rhs, types);
  return types;
}

std::string argumentType(const ast::ConstraintExpr& argument,
                         const Types& types,
                         std::vector<HostDiagnostic>& diagnostics) {
  if (const auto* name = std::get_if<ast::NameRef>(&argument.value)) {
    auto found = types.find(name->name);
    if (found != types.end() && !found->second.empty()) {
      return found->second;
    }
    if (found != types.end()) {
      addDiagnostic(
          diagnostics, argument.span,
          "scalar host argument '" + name->name + "' needs an explicit type");
      return {};
    }
    addDiagnostic(diagnostics, argument.span,
                  "unknown host argument '" + name->name + "'");
  } else if (std::holds_alternative<ast::AttributeRef>(argument.value)) {
    return "&OpAttrs";
  } else if (std::holds_alternative<ast::IntegerLiteral>(argument.value)) {
    return "usize";
  } else if (std::holds_alternative<ast::BooleanLiteral>(argument.value)) {
    return "bool";
  } else {
    addDiagnostic(diagnostics, argument.span,
                  "host template needs a named, attribute, or literal "
                  "argument; type complex expressions before generating it");
  }
  return {};
}

bool containsCall(const ast::ConstraintExpr& expression) {
  if (std::holds_alternative<ast::Call>(expression.value)) return true;
  if (const auto* unary = std::get_if<ast::UnaryExpr>(&expression.value)) {
    return containsCall(*unary->operand);
  }
  if (const auto* binary = std::get_if<ast::BinaryExpr>(&expression.value)) {
    return containsCall(*binary->lhs) || containsCall(*binary->rhs);
  }
  return false;
}

void collectCalls(const ast::ConstraintExpr& expression, const Types& types,
                  const std::string& expected, Functions& functions,
                  std::vector<HostDiagnostic>& diagnostics) {
  if (const auto* call = std::get_if<ast::Call>(&expression.value)) {
    Signature signature{{}, expected};
    for (const auto& argument : call->arguments) {
      signature.arguments.push_back(
          argumentType(*argument, types, diagnostics));
    }
    auto [previous, inserted] = functions.emplace(call->callee, signature);
    if (!inserted && (previous->second.arguments != signature.arguments ||
                      previous->second.result != signature.result)) {
      addDiagnostic(
          diagnostics, expression.span,
          "conflicting signatures for host function '" + call->callee + "'");
    }
  } else if (const auto* unary =
                 std::get_if<ast::UnaryExpr>(&expression.value)) {
    collectCalls(*unary->operand, types,
                 unary->op == ast::UnaryOp::kLogicalNot ? "bool" : "usize",
                 functions, diagnostics);
  } else if (const auto* binary =
                 std::get_if<ast::BinaryExpr>(&expression.value)) {
    std::string operand_type;
    switch (binary->op) {
      case ast::BinaryOp::kLogicalAnd:
      case ast::BinaryOp::kLogicalOr:
        operand_type = "bool";
        break;
      case ast::BinaryOp::kEqual:
      case ast::BinaryOp::kNotEqual:
        if (containsCall(expression)) {
          addDiagnostic(diagnostics, expression.span,
                        "host calls in equality expressions need explicit "
                        "type information");
        }
        return;
      default:
        operand_type = "usize";
        break;
    }
    collectCalls(*binary->lhs, types, operand_type, functions, diagnostics);
    collectCalls(*binary->rhs, types, operand_type, functions, diagnostics);
  }
}

void writeMethod(std::ostringstream& output, const std::string& name,
                 const Signature& signature, bool implementation) {
  output << "    fn " << name << "(&self";
  for (size_t index = 0; index < signature.arguments.size(); ++index) {
    output << (implementation ? ", _arg" : ", arg") << index << ": "
           << signature.arguments[index];
  }
  output << ") -> Option<" << signature.result << ">";
  if (implementation) {
    output << " {\n        todo!(\"implement " << name << "\")\n    }\n";
  } else {
    output << ";\n";
  }
}

}  // namespace

HostTemplateResult generateHostTemplate(const ast::Program& program,
                                        bool implementation) {
  HostTemplateResult result;
  Functions functions;
  for (const auto& rule : program.rules) {
    const Types types = declaredTypes(rule);
    for (const auto& condition : rule.conditions) {
      collectCalls(*condition, types, "bool", functions, result.diagnostics);
    }
    for (const auto& derivation : rule.derivations) {
      collectCalls(*derivation.value, types, "InferredTensor", functions,
                   result.diagnostics);
    }
  }
  if (!result.ok()) return result;

  std::ostringstream output;
  output << "// Generated host functions. Keep user implementations in a "
            "separate file.\n";
  std::set<std::string> used_types;
  for (const auto& [name, signature] : functions) {
    used_types.insert(signature.result);
    for (const auto& argument : signature.arguments) {
      used_types.insert(argument);
    }
  }
  if (used_types.contains("&OpAttrs")) {
    output << "use rust_egg::ir::OpAttrs;\n";
  }
  if (used_types.contains("&TensorInfo") ||
      used_types.contains("InferredTensor")) {
    output << "use rust_egg::ir::patterns::{";
    if (used_types.contains("InferredTensor")) output << "InferredTensor";
    if (used_types.contains("&TensorInfo")) {
      if (used_types.contains("InferredTensor")) output << ", ";
      output << "TensorInfo";
    }
    output << "};\n";
  }
  output << '\n';
  if (implementation) {
    output << "// Save this once and fill in the functions. Do not overwrite "
              "your implementation when regenerating the interface.\n";
    output << "use crate::generated_host::HostFunctions;\n\n";
    output << "pub struct UserFunctions;\n\nimpl HostFunctions for "
              "UserFunctions {\n";
  } else {
    output << "pub trait HostFunctions: Send + Sync {\n";
  }
  for (const auto& [name, signature] : functions) {
    writeMethod(output, name, signature, implementation);
  }
  output << "}\n";
  result.source = output.str();
  return result;
}

}  // namespace tepl
