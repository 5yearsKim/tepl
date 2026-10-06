#include "src/codegen/cpp/rule_emitter.h"

#include <type_traits>

#include "src/codegen/common/rule_plan.h"
#include "src/codegen/cpp/code_writer.h"
#include "src/codegen/cpp/expression_emitter.h"
#include "src/codegen/cpp/paths.h"

namespace tepl::codegen::cpp {
namespace {
std::string literalDtype(const core::GraphLiteral& value) {
  return value.dtype ? "::std::optional(" + dtype(*value.dtype) + ")"
                     : "::std::nullopt";
}
std::string pattern(const core::Pattern& node, const Names& names) {
  return std::visit(
      [&](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, core::CapturePattern>)
          return "TensorPattern::var(" + std::to_string(value.capture.value) +
                 ")";
        else if constexpr (std::is_same_v<T, core::GraphLiteral>)
          return "TensorPattern::literal(" + quote(value.spelling) + "," +
                 literalDtype(value) + ")";
        else if constexpr (std::is_same_v<T, core::BindPattern>)
          return "TensorPattern::bind(" + std::to_string(value.capture.value) +
                 "," + pattern(*value.expression, names) + ")";
        else {
          auto out = "TensorPattern::op(" +
                     names.operation(value.operation, names.root) + "," +
                     (value.descriptor
                          ? "AttrPattern::bind(" +
                                std::to_string(value.descriptor->value) + ")"
                          : "AttrPattern::value({})") +
                     ",{";
          for (const auto& child : value.operands)
            out += pattern(*child, names) + ",";
          return out + "})";
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
          return "TensorExpr::var(" + std::to_string(value.capture.value) + ")";
        else if constexpr (std::is_same_v<T, core::GraphLiteral>)
          return "TensorExpr::literal(" + quote(value.spelling) + "," +
                 literalDtype(value) + ")";
        else {
          auto attrs = std::string("AttrExpr::exact()");
          if (value.descriptor)
            attrs =
                "AttrExpr::" +
                std::string(rule.descriptors.at(value.descriptor->value).kind ==
                                    core::Descriptor::Kind::kCaptured
                                ? "captured("
                                : "derived(") +
                std::to_string(value.descriptor->value) + ")";
          auto out = "TensorExpr::op(" +
                     names.operation(value.operation, names.root) + "," +
                     attrs + ",{";
          for (const auto& child : value.operands)
            out += build(*child, rule, names) + ",";
          return out + "})";
        }
      },
      node.value);
}
void emitRule(CodeWriter& out, const core::Program& program, const Names& names,
              const RuleModuleNames& module, const core::Rule& rule) {
  const auto plan = planRule(rule);
  out.open("namespace " + names.root.substr(2) +
           "::rules::" + module.qualified + "::" + names.rule(rule.id));
  for (auto name : {"TensorPattern", "TensorExpr", "AttrPattern", "AttrExpr",
                    "TensorConstraints", "DTypeConstraint", "ShapePart",
                    "MetadataBindings", "MatchBinding", "DerivedAttrs"})
    out.line("using " + names.root + "::rewriting::" + name + ";");
  for (auto name : {"OpAttrs", "DType", "TensorInfo"})
    out.line("using " + names.root + "::" + name + ";");
  if (plan.host_functions.empty())
    out.line("template<class F> concept HostFunctions=true;");
  else {
    out.line("template<class F> concept HostFunctions =");
    bool first = true;
    for (auto id : plan.host_functions) {
      const auto& fn = program.host_functions.at(id.value);
      std::string declaration = "requires(const F& functions";
      for (std::size_t i = 0; i < fn.signature.arguments.size(); ++i)
        declaration +=
            "," +
            type(program.types.at(fn.signature.arguments[i].value), true) +
            " arg_" + std::to_string(i);
      declaration += ") { { functions." + names.host(id) + "(";
      for (std::size_t i = 0; i < fn.signature.arguments.size(); ++i) {
        if (i) declaration += ",";
        declaration += "arg_" + std::to_string(i);
      }
      auto result = type(program.types.at(fn.signature.result.value));
      if (fn.fallible) result = "::std::optional<" + result + ">";
      out.line((first ? "" : "&& ") + declaration + ") } -> ::std::same_as<" +
               result + ">; }");
      first = false;
    }
    out.line(";");
  }
  out.line("inline TensorPattern pattern() { return " +
           pattern(*rule.lhs, names) + "; }");
  out.line("inline TensorExpr expression() { return " +
           build(*rule.rhs, rule, names) + "; }");
  out.open("inline TensorConstraints constraints()");
  out.line("return {{");
  for (const auto& c : rule.constraints) {
    std::string shape = "{";
    for (const auto& part : c.shape) {
      switch (part.kind) {
        case core::ShapeElement::Kind::kDimension:
          shape += "ShapePart::dimension(" +
                   std::to_string(part.symbol->value) + ")";
          break;
        case core::ShapeElement::Kind::kWildcard:
          shape += "ShapePart::wildcard()";
          break;
        case core::ShapeElement::Kind::kSequence:
          shape += "ShapePart::sequence(" +
                   (part.symbol ? "::std::optional<::std::size_t>(" +
                                      std::to_string(part.symbol->value) + ")"
                                : "::std::nullopt") +
                   ")";
          break;
      }
      shape += ",";
    }
    out.line("{" + std::to_string(c.capture.value) + ",{" +
             (c.dtype
                  ? "::std::optional(" +
                        (std::holds_alternative<core::DType>(*c.dtype)
                             ? "DTypeConstraint::exact(" +
                                   dtype(std::get<core::DType>(*c.dtype)) + ")"
                             : "DTypeConstraint::bind(" +
                                   std::to_string(
                                       std::get<core::DTypeVariableId>(*c.dtype)
                                           .value) +
                                   ")") +
                        ")"
                  : "::std::nullopt") +
             "," + shape + "}}},");
  }
  out.line("}};");
  out.close();
  for (std::size_t i = 0; i < plan.early_conditions.size(); ++i) {
    out.line("template<class C>");
    out.open("inline ::std::optional<bool> condition_" + std::to_string(i) +
             "([[maybe_unused]] const C& ctx, [[maybe_unused]] const "
             "MetadataBindings& dimensions)");
    out.line("try { return " +
             emitExpression(program, names, *rule.conditions[i]) +
             "; } catch(const " + names.root +
             "::builtins::BuiltinError&) { return {}; }");
    out.close();
  }
  out.line("template<class A>");
  out.open("inline " + names.root +
           "::rewriting::MatchChecks<A> match_checks()");
  if (plan.early_conditions.empty())
    out.line("return {constraints(),{}, {}};");
  else {
    out.line("auto metadata=[](const ::eggc::EGraph<" + names.root +
             "::OpNode,A>& graph, ::eggc::Id id) { return " + names.root +
             "::RewriteAnalysis<A>::tensor_info(graph,id); }; ");
    out.line("return {constraints(),{");
    for (const auto& c : plan.early_conditions) {
      std::string dependencies = "{";
      for (auto id : c.captures)
        dependencies +=
            "{MatchBinding::Kind::Tensor," + std::to_string(id.value) + "},";
      for (auto id : c.descriptors)
        dependencies +=
            "{MatchBinding::Kind::Attribute," + std::to_string(id.value) + "},";
      for (auto id : c.dimensions)
        dependencies +=
            "{MatchBinding::Kind::" +
            std::string(rule.dimensions.at(id.value).sequence ? "Sequence"
                                                              : "Dimension") +
            "," + std::to_string(id.value) + "},";
      for (auto id : c.dtypes)
        dependencies +=
            "{MatchBinding::Kind::DType," + std::to_string(id.value) + "},";
      out.line(dependencies + "},");
    }
    out.open(
        "},[metadata=::std::move(metadata)](::std::size_t index, const "
        "::eggc::EGraph<" +
        names.root + "::OpNode,A>& graph, const " + names.root +
        "::rewriting::TensorMatch& matched, const MetadataBindings& "
        "dimensions)->::std::optional<bool>");
    out.line(names.root +
             "::rewriting::MatchContext ctx{graph,matched,metadata};");
    out.open("switch(index)");
    for (std::size_t i = 0; i < plan.early_conditions.size(); ++i)
      out.line("case " + std::to_string(i) + ": return condition_" +
               std::to_string(i) + "(ctx,dimensions);");
    out.line("default: return {};");
    out.close();
    out.close("};");
  }
  out.close();
  out.line("template<class A=" + names.root +
           "::analysis::TensorAnalysis, HostFunctions F=" + names.root +
           "::rewriting::NoFunctions>");
  out.open("inline ::eggc::Rewrite<" + names.root +
           "::OpNode,A> build(F functions={})");
  out.line("auto host=::std::make_shared<F>(::std::move(functions));");
  out.line("auto reader=[](const ::eggc::EGraph<" + names.root +
           "::OpNode,A>& graph, ::eggc::Id id) { return " + names.root +
           "::RewriteAnalysis<A>::tensor_info(graph,id); };");
  out.line("auto checks=match_checks<A>();");
  out.line("return " + names.root +
           "::rewriting::tensor_rewrite_with_checks<A>(" +
           quote(module.qualified + "::" + rule.name) +
           ",pattern(),expression(),::std::move(checks)," +
           (plan.requires_tensor_info ? "true," : "false,"));
  out.open("[host,reader](const ::eggc::EGraph<" + names.root +
           "::OpNode,A>& graph, const " + names.root +
           "::rewriting::TensorMatch& matched, [[maybe_unused]] const "
           "MetadataBindings& dimensions)->::std::optional<DerivedAttrs>");
  out.line("[[maybe_unused]] const auto& functions=*host;");
  out.line("DerivedAttrs derived;");
  out.line("[[maybe_unused]] " + names.root +
           "::rewriting::MatchContext ctx{graph,matched,reader,&derived};");
  out.open("try");
  for (const auto& d : rule.descriptors)
    if (d.kind == core::Descriptor::Kind::kCaptured) {
      const auto& t = program.types.at(d.type.value);
      if (t.schema)
        out.line(names.root + "::builtins::require(" + names.root +
                 "::builtins::require(ctx.attrs(" + std::to_string(d.id.value) +
                 ")).checked_schema(" + std::to_string(t.schema->value) +
                 "));");
    }
  for (std::size_t i = plan.early_conditions.size(); i < rule.conditions.size();
       ++i)
    out.line("if(!(" + emitExpression(program, names, *rule.conditions[i]) +
             ")) return {};");
  for (const auto& d : rule.derivations)
    out.line("derived.emplace(" + std::to_string(d.target.value) + "," +
             emitExpression(program, names, *d.value) + ");");
  out.line("return derived;");
  out.close();
  out.line("catch(const " + names.root +
           "::builtins::BuiltinError&) { return {}; }");
  out.close(");");
  out.close();
  out.close();
}
}  // namespace
std::string emitRules(const core::Program& program, const Names& names,
                      const std::vector<const core::Rule*>& rules,
                      const RuleModuleNames& module) {
  CodeWriter out;
  out.line("#pragma once");
  out.line("#include \"" + parentInclude(module.path) +
           "rewriting/rewrite.h\"");
  out.line("#include \"" + parentInclude(module.path) +
           "analysis/analysis.h\"");
  for (auto rule : rules) emitRule(out, program, names, module, *rule);
  return out.str();
}
}  // namespace tepl::codegen::cpp
