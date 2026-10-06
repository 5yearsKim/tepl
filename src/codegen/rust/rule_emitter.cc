#include "src/codegen/rust/rule_emitter.h"

#include <type_traits>

#include "src/codegen/common/rule_plan.h"
#include "src/codegen/rust/code_writer.h"
#include "src/codegen/rust/expression_emitter.h"
#include "src/codegen/rust/paths.h"

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

std::string pattern(const core::Pattern& node, const Names& names,
                    const std::string& root) {
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
                 pattern(*value.expression, names, root) + ")";
        else {
          std::string result =
              "TensorPattern::op(" + names.operation(value.operation, root) +
              ", " +
              (value.descriptor ? "AttrPattern::Bind(" +
                                      attrVar(value.descriptor->value) + ")"
                                : "AttrPattern::Exact(OpAttrs::None)") +
              ", vec![";
          for (const auto& operand : value.operands)
            result += pattern(*operand, names, root) + ", ";
          return result + "])";
        }
      },
      node.value);
}
std::string build(const core::BuildExpr& node, const core::Rule& rule,
                  const Names& names, const std::string& root) {
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
                               names.operation(value.operation, root) + ", " +
                               attrs + ", vec![";
          for (const auto& operand : value.operands)
            result += build(*operand, rule, names, root) + ", ";
          return result + "])";
        }
      },
      node.value);
}

void emitRule(CodeWriter& out, const core::Program& program,
              const core::Rule& rule, const Names& names,
              const std::string& module, const std::string& root) {
  const auto plan = planRule(rule);
  out.open("pub mod " + names.rule(rule.id));
  out.line("use " + paths::external("std", "collections::HashMap") + ";");
  out.line("use " + paths::external("egg", "{EGraph, Rewrite, Var}") + ";");
  out.line("use " + paths::within(root, "{DType, OpNode, OpAttrs}") + ";");
  out.line("use " +
           paths::within(root, "pattern::{RewriteAnalysis, AnalysisMetadata}") +
           ";");
  out.line(
      "use " +
      paths::within(root,
                    "pattern::{AttrExpr, AttrPattern, AttrVar, TensorExpr, "
                    "TensorPattern, TensorConstraints, TensorConstraint, "
                    "DTypeConstraint, "
                    "ShapePart, MetadataBindings, MatchBinding, MatchChecks, "
                    "TensorInfo, "
                    "MatchContext, tensor_rewrite_with_checks}") +
      ";");
  const auto builtins_path = paths::within(root, "builtins");
  out.open("pub trait Functions: Send + Sync");
  for (const auto id : plan.host_functions) {
    const auto& fn = program.host_functions.at(id.value);
    out.line("/// Host implementation of TEPL `$" + fn.name + "(...)`.");
    std::string signature = "fn " + names.host(fn.id) + "(&self";
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
  out.line(pattern(*rule.lhs, names, root));
  out.close();
  out.open("pub fn expression() -> TensorExpr");
  out.line(build(*rule.rhs, rule, names, root));
  out.close();
  out.open("pub fn constraints() -> TensorConstraints");
  out.line("TensorConstraints::new(vec![");
  for (const auto& constraint : rule.constraints) {
    out.line("(" + var(constraint.capture.value) + ", TensorConstraint {");
    out.line(
        "dtype: " +
        (constraint.dtype
             ? "Some(" +
                   (std::holds_alternative<core::DType>(*constraint.dtype)
                        ? "DTypeConstraint::Exact(" +
                              dtype(std::get<core::DType>(*constraint.dtype)) +
                              ")"
                        : "DTypeConstraint::Bind(" +
                              std::to_string(std::get<core::DTypeVariableId>(
                                                 *constraint.dtype)
                                                 .value) +
                              ")") +
                   ")"
             : "None") +
        ",");
    std::string shape = "shape: vec![";
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
    out.line(shape + "], }),");
  }
  out.line("])");
  out.close();
  for (std::size_t i = 0; i < plan.early_conditions.size(); ++i) {
    out.open(
        "fn condition_" + std::to_string(i) +
        "<N: RewriteAnalysis>(ctx: &MatchContext<'_, N, "
        "AnalysisMetadata<N>>, dimensions: &MetadataBindings) -> Option<bool>");
    out.line(
        "Some(" +
        emitExpression(program, names, *rule.conditions[i], builtins_path) +
        ")");
    out.close();
  }
  out.line("/// Shape declarations and the ordered builtin-only where prefix.");
  out.open("pub fn match_checks<N: RewriteAnalysis>() -> MatchChecks<N>");
  if (plan.early_conditions.empty()) {
    out.line("MatchChecks::tensors(constraints())");
  } else {
    out.line("let metadata = AnalysisMetadata::<N>::new();");
    out.line("MatchChecks::new(constraints(), vec![");
    for (const auto& condition : plan.early_conditions) {
      std::string deps = "vec![";
      for (auto id : condition.captures)
        deps += "MatchBinding::Tensor(" + var(id.value) + "), ";
      for (auto id : condition.dimensions) {
        deps += std::string(rule.dimensions.at(id.value).sequence
                                ? "MatchBinding::Sequence("
                                : "MatchBinding::Dimension(") +
                std::to_string(id.value) + "), ";
      }
      for (auto id : condition.descriptors)
        deps += "MatchBinding::Attribute(" + attrVar(id.value) + "), ";
      for (auto id : condition.dtypes)
        deps += "MatchBinding::DType(" + std::to_string(id.value) + "), ";
      out.line(deps + "],");
    }
    out.line("], move |index, graph, matched, dimensions| {");
    out.line("let ctx = MatchContext::new(graph, matched, &metadata);");
    out.open("match index");
    for (std::size_t i = 0; i < plan.early_conditions.size(); ++i)
      out.line(std::to_string(i) + " => condition_" + std::to_string(i) +
               "(&ctx, dimensions),");
    out.line("_ => None,");
    out.close();
    out.line("})");
  }
  out.close();
  out.line("/// Uses the graph's analysis; () supports structural rules.");
  out.line(
      "pub fn build_rewrite<N, F>(functions: F) -> Result<Rewrite<OpNode, N>, "
      "String>");
  out.line("where N: RewriteAnalysis, F: Functions + 'static");
  out.open("");
  out.line("let checks = match_checks::<N>();");
  out.line("let metadata = AnalysisMetadata::<N>::new();");
  out.line("tensor_rewrite_with_checks(");
  out.line(quote(module + "::" + rule.name) +
           ", pattern(), expression(), checks, " +
           (plan.requires_tensor_info ? "true," : "false,"));
  out.open("move |graph, matched, dimensions|");
  out.line("let ctx = MatchContext::new(graph, matched, &metadata);");
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
  for (std::size_t i = plan.early_conditions.size(); i < rule.conditions.size();
       ++i)
    out.line(
        "if !(" +
        emitExpression(program, names, *rule.conditions[i], builtins_path) +
        ") { return None; }");
  for (const auto& derivation : rule.derivations)
    out.line("let " + descriptor(derivation.target.value) + " = " +
             emitExpression(program, names, *derivation.value, builtins_path) +
             ";");
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
  for (const auto* rule : rules) {
    out.line();
    emitRule(out, program, *rule, names, module, paths::parent(root_path));
  }
  return out.str();
}
}  // namespace tepl::codegen::rust
