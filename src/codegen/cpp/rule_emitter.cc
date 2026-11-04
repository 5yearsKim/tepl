#include "src/codegen/cpp/rule_emitter.h"

#include <set>

#include "src/codegen/cpp/code_writer.h"
#include "src/codegen/cpp/dialect_emitter.h"
#include "src/codegen/cpp/expression_emitter.h"
#include "src/codegen/cpp/paths.h"
#include "src/shared/catalog/axis_conditions.h"
namespace tepl::codegen::cpp {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
std::string optional(std::optional<std::size_t> id) {
  return id ? "::std::optional<::std::size_t>(" + num(*id) + ")"
            : "::std::nullopt";
}
std::string dtypeOption(std::optional<core::DType> d) {
  return d ? "::std::optional<rt::DType>(rt::" + dtype(*d) + ")"
           : "::std::nullopt";
}
std::string pattern(const core::Pattern& p) {
  return std::visit(
      [&](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, core::CapturePattern>)
          return "rt::matching::Pattern::capture(" + num(v.capture.value) + ")";
        else if constexpr (std::is_same_v<T, core::BindPattern>)
          return "rt::matching::Pattern::bind(" + num(v.capture.value) + "," +
                 pattern(*v.expression) + ")";
        else if constexpr (std::is_same_v<T, core::GraphLiteral>)
          return "rt::matching::Pattern::literal(" + quote(v.spelling) + "," +
                 dtypeOption(v.dtype) + ")";
        else {
          std::string out =
              "rt::matching::Pattern::operation(" + num(v.operation.value) +
              "," +
              optional(v.descriptor ? std::optional(v.descriptor->value)
                                    : std::nullopt) +
              ",{";
          for (const auto& x : v.operands) out += pattern(*x) + ",";
          return out + "})";
        }
      },
      p.value);
}
std::string rhs(const executable::ProgramData& p, const core::BuildExpr& expr) {
  return "([&]() -> Replacement {ctx.budget.tick(); return " +
         std::visit(
             [&](const auto& v) -> std::string {
               using T = std::decay_t<decltype(v)>;
               if constexpr (std::is_same_v<T, core::CaptureRef>)
                 return "ctx.captured_rhs(" + num(v.capture.value) + ")";
               else if constexpr (std::is_same_v<T, core::GraphLiteral>)
                 return "ctx.literal_rhs(" + quote(v.spelling) + "," +
                        dtypeOption(v.dtype) + ")";
               else {
                 std::string attrs = "rt::Value()";
                 const auto& op = p.operations.at(v.operation.value);
                 if (v.descriptor)
                   attrs = "ctx.descriptor(" + num(v.descriptor->value) + ")";
                 else if (op.attributes) {
                   attrs = "rt::Value(rt::Descriptor{" +
                           num(op.attributes->value) + ",{";
                   for (const auto& f :
                        p.attribute_schemas.at(op.attributes->value).fields)
                     attrs += f.empty_default ? "rt::Value(rt::Value::List{}),"
                                              : "rt::Value(),";
                   attrs += "}})";
                 }
                 std::string out =
                     "([&]() -> Replacement { auto children = "
                     "::std::vector<Replacement>{";
                 for (const auto& c : v.operands) out += rhs(p, *c) + ",";
                 return out + "}; auto attrs = " + attrs +
                        "; return ctx.operation_rhs(" + num(v.operation.value) +
                        ",::std::move(attrs),::std::move(children)); }())";
               }
             },
             expr.value) +
         "; }())";
}
std::string constraints(const core::completed::RuleData& rule) {
  std::string out = "{";
  for (const auto& c : rule.constraints) {
    out += "{" + num(c.capture.value) + ",{";
    for (const auto& e : c.shape)
      out +=
          "{" +
          std::string(e.kind == core::ShapeElement::Kind::kSequence ? "true"
                                                                    : "false") +
          "," +
          optional(e.symbol ? std::optional(e.symbol->value) : std::nullopt) +
          "},";
    out += "},";
    if (!c.dtype)
      out += "::std::nullopt";
    else if (auto d = std::get_if<core::DType>(&*c.dtype))
      out +=
          "rt::matching::DTypeConstraint{rt::" + dtype(*d) + ",::std::nullopt}";
    else
      out += "rt::matching::DTypeConstraint{::std::nullopt," +
             num(std::get<core::DTypeVariableId>(*c.dtype).value) + "}";
    out += "},";
  }
  return out + "}";
}
}  // namespace
std::string emitRules(const executable::ExecutableProgram& executable,
                      const Names& n, const RuleModuleNames& module) {
  const auto& p = executable.ir();
  Expressions e(p);
  CodeWriter out;
  out.line("#pragma once\n#include \"" + parentInclude(module.path) +
           "program.h\"");
  for (auto id : module.rules) {
    const auto& rule = p.rules.at(id.value);
    out.open("namespace " + n.root.substr(2) + "::rules::" + module.qualified +
             "::" + n.rule(id));
    out.line(
        "namespace rt = ::tepl_runtime;\nusing Context = rt::RuleContext<" +
        n.root + "::Program>;\nusing Replacement = rt::PreparedReplacement<" +
        n.root + "::Program>;");
    std::set<std::pair<std::size_t, std::size_t>> emitted;
    for (const auto& axis : executable.axes(id.value))
      if (emitted.emplace(axis->owner, axis->index).second)
        out.line("inline rt::Value axis_" + num(axis->owner) + "_" +
                 num(axis->index) + "(Context& ctx);");
    for (const auto& axis : executable.axes(id.value)) {
      auto key = std::pair(axis->owner, axis->index);
      if (!emitted.erase(key)) continue;
      out.open("inline rt::Value axis_" + num(axis->owner) + "_" +
               num(axis->index) + "(Context& ctx)");
      out.line("if(auto value=ctx.cached_axis(" + num(axis->owner) + "," +
               num(axis->index) + ")) return *value;");
      out.line("ctx.enter_axis(" + num(axis->owner) + "," + num(axis->index) +
               ");");
      out.line("auto extents = " + e.symbolic(*axis->args.at(0)) + ";");
      out.line("return ctx.finish_axis(" + num(axis->owner) + "," +
               num(axis->index) + ",extents);");
      out.close();
    }
    for (const auto& derivation : rule.derivations)
      if (derivation.target) {
        auto schema = *p.types.at(derivation.value->type.value).schema;
        auto t = schemaType(n, schema);
        out.open("inline rt::EvalResult<" + t + "> derive_d" +
                 num(derivation.target->value) +
                 "([[maybe_unused]] Context& ctx)");
        out.open("return rt::evaluate([&]() -> " + t);
        out.line("auto value = " + e.rule(*derivation.value) + ";");
        out.line("return " + t + "::from_value(value).value();");
        out.close(");");
        out.close();
      }
    out.open("inline void derive([[maybe_unused]] Context& ctx)");
    for (const auto& d : rule.derivations)
      if (d.target)
        out.line("ctx.bindings.descriptors[" + num(d.target->value) +
                 "] = derive_d" + num(d.target->value) +
                 "(ctx).value().to_value();");
      else
        out.line("(void)(" + e.rule(*d.value) + ");");
    out.close();
    out.open(
        "inline rt::EvalResult<bool> check_where([[maybe_unused]] Context& "
        "ctx)");
    out.open("return rt::evaluate([&]() -> bool");
    auto emitConditions = [&](const auto& statements) {
      for (const auto& c : statements)
        if (std::holds_alternative<core::completed::LocalLet>(c->value))
          out.line("(void)(" + e.rule(*c) + ");");
        else
          out.line("if(!(" + e.rule(*c) + ").boolean()) return false;");
    };
    auto schedule = core::completed::scheduleWhere(rule);
    emitConditions(schedule.source);
    for (const auto& q : rule.feasibility)
      out.line("if (!(" + e.symbolic(*q) + ").boolean()) return false;");
    out.line(
        "{ rt::LocalScope scope(ctx.bindings.locals,true); derive(ctx); }");
    emitConditions(schedule.replacement);
    out.line("return true;");
    out.close(");");
    out.close();
    out.open(
        "inline rt::EvalResult<::std::optional<Replacement>> prepare(Context& "
        "ctx)");
    out.open("return rt::evaluate([&]() -> ::std::optional<Replacement>");
    out.line("if(!check_where(ctx).value()) return ::std::nullopt;");
    out.line("return " + rhs(p, *rule.rhs) + ";");
    out.close(");");
    out.close();
    out.open("inline rt::RuleDefinition<" + n.root + "::Program> definition()");
    out.line("return {" + quote(module.qualified + "::" + rule.name) + "," +
             quote(rule.origin.definition.source_name + ":" +
                   num(rule.origin.definition.span.begin.line)) +
             "," + pattern(*rule.lhs) + "," + constraints(rule) + ",prepare};");
    out.close();
    out.close();
  }
  return out.str();
}
}  // namespace tepl::codegen::cpp
