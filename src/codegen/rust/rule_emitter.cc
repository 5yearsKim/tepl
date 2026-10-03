#include "src/codegen/rust/rule_emitter.h"

#include <set>
#include <stdexcept>
#include <type_traits>

#include "src/codegen/common/rule_plan.h"
#include "src/codegen/rust/code_writer.h"
#include "src/codegen/rust/expression_emitter.h"

namespace tepl::codegen::rust {
namespace {
std::string var(std::size_t id) {
  return quote("?c" + std::to_string(id)) +
         ".parse::<Var>().expect(\"generated capture ID\")";
}
std::string attrVar(std::size_t id) {
  return "AttrVar::from(" + quote("d" + std::to_string(id)) + ")";
}
std::string literalDtype(const core::GraphLiteral& value) {
  return value.dtype ? "Some(" + dtype(*value.dtype) + ")" : "None";
}

std::string pattern(const core::Pattern& node, const Names& names) {
  return std::visit(
      [&](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, core::CapturePattern>)
          return "TensorPattern::Var(" + var(value.capture.value) + ")";
        else if constexpr (std::is_same_v<T, core::GraphLiteral>)
          return "TensorPattern::literal(" + quote(value.spelling) + ", " +
                 literalDtype(value) + ")";
        else if constexpr (std::is_same_v<T, core::BindPattern>)
          return "TensorPattern::bind(" + var(value.capture.value) + ", " +
                 pattern(*value.expression, names) + ")";
        else {
          std::string result =
              "TensorPattern::op(" + names.operation(value.operation) + ", " +
              (value.descriptor ? "AttrPattern::Bind(" +
                                      attrVar(value.descriptor->value) + ")"
                                : "AttrPattern::Exact(OpAttrs::None)") +
              ", vec![";
          for (const auto& operand : value.operands)
            result += pattern(*operand, names) + ", ";
          return result + "])";
        }
      },
      node.value);
}
std::string build(const core::BuildExpr& node, const core::Rule& rule,
                  const Names& names) {
  return std::visit(
      [&](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, core::CaptureRef>)
          return "TensorExpr::Var(" + var(value.capture.value) + ")";
        else if constexpr (std::is_same_v<T, core::GraphLiteral>)
          return "TensorExpr::literal(" + quote(value.spelling) + ", " +
                 literalDtype(value) + ")";
        else {
          std::string attrs = "AttrExpr::Exact(OpAttrs::None)";
          if (value.descriptor) {
            const auto& desc = rule.descriptors.at(value.descriptor->value);
            attrs = std::string(desc.kind == core::Descriptor::Kind::kCaptured
                                    ? "AttrExpr::Captured("
                                    : "AttrExpr::Derived(") +
                    attrVar(desc.id.value) + ")";
          }
          std::string result = "TensorExpr::op(" +
                               names.operation(value.operation) + ", " + attrs +
                               ", vec![";
          for (const auto& operand : value.operands)
            result += build(*operand, rule, names) + ", ";
          return result + "])";
        }
      },
      node.value);
}

void emitRule(CodeWriter& out, const core::Program& program,
              const core::Rule& rule, const Names& names,
              const std::string& module) {
  const auto plan = planRule(rule);
  out.open("pub mod rule_" + rule.name);
  out.line("use super::*;");
  out.open("pub trait Functions: Send + Sync");
  std::set<std::string> function_names;
  for (const auto id : plan.host_functions) {
    const auto& fn = program.host_functions.at(id.value);
    if (!function_names.insert(fn.name).second)
      throw std::invalid_argument("host function name collision within rule " +
                                  module + "::" + rule.name);
    out.line("/// Host implementation of TEPL `$" + fn.name + "(...)`.");
    std::string signature = "fn " + identifier(fn.name) + "(&self";
    for (std::size_t i = 0; i < fn.signature.arguments.size(); ++i)
      signature +=
          ", arg" + std::to_string(i) + ": " +
          type(program.types.at(fn.signature.arguments[i].value), true);
    auto result = type(program.types.at(fn.signature.result.value));
    if (fn.fallible) result = "Option<" + result + ">";
    out.line(signature + ") -> " + result + ";");
  }
  out.close();
  if (plan.host_functions.empty()) out.line("impl Functions for () {}");
  out.open("pub fn pattern() -> TensorPattern");
  out.line(pattern(*rule.lhs, names));
  out.close();
  out.open("pub fn expression() -> TensorExpr");
  out.line(build(*rule.rhs, rule, names));
  out.close();
  out.line(
      "pub fn build_rewrite<N, M, I, F>(metadata: M, inference: I, functions: "
      "F) -> Result<Rewrite<OpNode, N>, String>");
  out.line(
      "where N: Analysis<OpNode>, M: TensorMetadata<N> + 'static, I: "
      "OutputInference + 'static, F: Functions + 'static");
  out.open("");
  out.line("let metadata = Arc::new(metadata);");
  out.line("let checker_metadata = metadata.clone();");
  out.line("tensor_rewrite_checked(");
  out.line(quote(module + "::" + rule.name) + ", pattern(), expression(),");
  out.line(
      "move |graph: &EGraph<OpNode, N>, id| metadata.info(graph, id), "
      "inference,");
  out.open("move |graph, matched|");
  out.line(
      "let ctx = MatchContext::new(graph, matched, "
      "checker_metadata.as_ref());");
  for (const auto id : plan.constraint_captures)
    out.line("let " + capture(id.value) + " = ctx.tensor(" + var(id.value) +
             ")?;");
  for (const auto& desc : rule.descriptors) {
    if (desc.kind != core::Descriptor::Kind::kCaptured) continue;
    out.line("let " + descriptor(desc.id.value) + " = ctx.attrs(" +
             attrVar(desc.id.value) + ")?.clone();");
    const auto& descriptor_type = program.types.at(desc.type.value);
    if (descriptor_type.schema)
      out.line("let " + descriptor(desc.id.value) + " = " +
               descriptor(desc.id.value) + ".checked_schema(" +
               std::to_string(descriptor_type.schema->value) + ")?;");
  }
  out.line("let mut dimensions = ShapeBindings::default();");
  for (const auto& constraint : rule.constraints) {
    const auto tensor = capture(constraint.capture.value);
    if (constraint.dtype)
      out.line("if " + tensor + ".dtype != " + dtype(*constraint.dtype) +
               " { return None; }");
    std::string shape = "dimensions.check(&" + tensor + ".shape, &[";
    for (const auto& element : constraint.shape) {
      switch (element.kind) {
        case core::ShapeElement::Kind::kDimension:
          shape += "ShapePart::Dimension(" +
                   std::to_string(element.symbol->value) + ")";
          break;
        case core::ShapeElement::Kind::kWildcard:
          shape += "ShapePart::Wildcard";
          break;
        case core::ShapeElement::Kind::kSequence:
          shape += "ShapePart::Sequence(" +
                   (element.symbol
                        ? "Some(" + std::to_string(element.symbol->value) + ")"
                        : "None") +
                   ")";
          break;
      }
      shape += ", ";
    }
    out.line(shape + "])?;");
  }
  for (const auto& condition : rule.conditions)
    out.line("if !(" + emitExpression(program, *condition) +
             ") { return None; }");
  for (const auto& derivation : rule.derivations)
    out.line("let " + descriptor(derivation.target.value) + " = " +
             emitExpression(program, *derivation.value) + ";");
  if (rule.derivations.empty())
    out.line("Some(Default::default())");
  else {
    out.line("Some(HashMap::from([");
    for (const auto& derivation : rule.derivations)
      out.line("(" + attrVar(derivation.target.value) + ", " +
               descriptor(derivation.target.value) + "),");
    out.line("]))");
  }
  out.close(",");
  out.line(")");
  out.close();
  out.close();
}
}  // namespace
std::string emitRules(const core::Program& program, const Names& names,
                      const std::vector<const core::Rule*>& rules,
                      const std::string& module, const std::string& root_path) {
  CodeWriter out;
  out.line("// Generated by TEPL from checked concrete rules.");
  out.line(
      "#![allow(unused_imports, unused_variables, unused_mut, unused_parens)]");
  out.line("use std::{collections::HashMap, sync::Arc};");
  out.line("use egg::{Analysis, EGraph, Rewrite, Var};");
  out.line("use " + root_path + "::{DType, OpNode, OpAttrs, Op};");
  out.line("use " + root_path + "::pattern::*;");
  out.line("use " + root_path + "::dialects::*;");
  for (const auto* rule : rules) {
    out.line();
    emitRule(out, program, *rule, names, module);
  }
  return out.str();
}
}  // namespace tepl::codegen::rust
