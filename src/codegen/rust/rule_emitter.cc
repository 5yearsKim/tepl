#include "src/codegen/rust/rule_emitter.h"

#include <set>

#include "src/codegen/rust/code_writer.h"
#include "src/codegen/rust/expression_emitter.h"
#include "src/shared/catalog/axis_conditions.h"
namespace tepl::codegen::rust {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
std::string descriptor(std::optional<core::DescriptorId> id) {
  return id ? "Some(" + num(id->value) + ")" : "None";
}
std::string pattern(const core::Pattern& p) {
  return std::visit(
      [&](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, core::CapturePattern>)
          return "Pattern::Capture(" + num(v.capture.value) + ")";
        else if constexpr (std::is_same_v<T, core::BindPattern>)
          return "Pattern::Bind(" + num(v.capture.value) + ", Box::new(" +
                 pattern(*v.expression) + "))";
        else if constexpr (std::is_same_v<T, core::GraphLiteral>)
          return "Pattern::Literal { value: " + quote(v.spelling) +
                 ".into(), dtype: " +
                 (v.dtype ? "Some(rt::" + dtype(*v.dtype) + ")" : "None") +
                 " }";
        else {
          std::string result =
              "Pattern::Operation { operation: " + num(v.operation.value) +
              ", descriptor: " + descriptor(v.descriptor) + ", operands: vec![";
          for (const auto& a : v.operands) result += pattern(*a) + ",";
          return result + "] }";
        }
      },
      p.value);
}
std::string rhs(const executable::ProgramData& p, const core::BuildExpr& e) {
  return "{ ctx.budget.tick()?; " +
         std::visit(
             [&](const auto& v) -> std::string {
               using T = std::decay_t<decltype(v)>;
               if constexpr (std::is_same_v<T, core::CaptureRef>)
                 return "ctx.captured_rhs(" + num(v.capture.value) + ")?";
               else if constexpr (std::is_same_v<T, core::GraphLiteral>)
                 return "ctx.literal_rhs(" + quote(v.spelling) + ", " +
                        (v.dtype ? "Some(rt::" + dtype(*v.dtype) + ")"
                                 : "None") +
                        ")?";
               else {
                 std::string attrs = "Value::None";
                 const auto& op = p.operations.at(v.operation.value);
                 if (v.descriptor)
                   attrs = "ctx.descriptor(" + num(v.descriptor->value) + ")?";
                 else if (op.attributes) {
                   attrs = "Value::Descriptor { schema: " +
                           num(op.attributes->value) + ", fields: vec![";
                   for (const auto& f :
                        p.attribute_schemas.at(op.attributes->value).fields)
                     attrs += f.empty_default ? "Value::List(vec![]),"
                                              : "Value::None,";
                   attrs += "] }";
                 }
                 std::string result = "{ let children = vec![";
                 for (const auto& child : v.operands)
                   result += rhs(p, *child) + ",";
                 return result + "]; let attrs = " + attrs +
                        "; ctx.operation_rhs(" + num(v.operation.value) +
                        ", attrs, children)? }";
               }
             },
             e.value) +
         " }";
}
std::string constraints(const core::completed::RuleData& rule) {
  std::string out = "vec![";
  for (const auto& c : rule.constraints) {
    out += "Constraint { capture: " + num(c.capture.value) + ", shape: vec![";
    for (const auto& e : c.shape)
      out += "ShapeElement::" +
             std::string(e.kind == core::ShapeElement::Kind::kSequence
                             ? "Sequence"
                             : "Dimension") +
             "(" + (e.symbol ? "Some(" + num(e.symbol->value) + ")" : "None") +
             "),";
    out += "], dtype: ";
    if (c.dtype)
      out += "Some(" +
             std::visit(
                 [](auto d) -> std::string {
                   using T = decltype(d);
                   if constexpr (std::is_same_v<T, core::DType>)
                     return "DTypeConstraint::Exact(rt::" + dtype(d) + ")";
                   else
                     return "DTypeConstraint::Variable(" + num(d.value) + ")";
                 },
                 *c.dtype) +
             ")";
    else
      out += "None";
    out += " },";
  }
  return out + "]";
}
}  // namespace
std::string emitRules(const executable::ExecutableProgram& executable,
                      const Names& names, const RuleModuleNames& module) {
  const auto& p = executable.ir();
  Expressions e(p);
  CodeWriter out;
  for (auto id : module.rules) {
    const auto& rule = p.rules.at(id.value);
    out.open("pub mod " + names.rule(id));
    out.line("use ::tepl_runtime::{self as rt, Value, EvalResult};");
    out.line(
        "use rt::matching::{Pattern, Constraint, ShapeElement, "
        "DTypeConstraint};");
    out.line("use rt::values::AttributeValue;");
    const auto root = "super::" + module.root;
    out.line("type Context<'a> = rt::RuleContext<'a, " + root + "::Program>;");
    out.line("type Replacement = rt::PreparedReplacement<" + root +
             "::Program>;");
    std::set<std::pair<std::size_t, std::size_t>> emitted_axes;
    for (const auto& axis : executable.axes(id.value)) {
      if (!emitted_axes.emplace(axis->owner, axis->index).second) continue;
      out.open("fn axis_" + num(axis->owner) + "_" + num(axis->index) +
               "(ctx: &mut Context<'_>) -> EvalResult<Value>");
      out.line("if let Some(value) = ctx.cached_axis(" + num(axis->owner) +
               ", " + num(axis->index) + ") { return Ok(value); }");
      out.line("ctx.enter_axis(" + num(axis->owner) + ", " + num(axis->index) +
               ")?;");
      out.line("let extents = " + e.symbolic(*axis->args.at(0)) + ";");
      out.line("ctx.finish_axis(" + num(axis->owner) + ", " + num(axis->index) +
               ", extents)");
      out.close();
    }
    const auto schedule = core::completed::scheduleWhere(rule);
    auto conditions = [&](const auto& statements) {
      for (const auto& condition : statements) {
        if (std::holds_alternative<core::completed::LocalLet>(condition->value))
          out.line("let _ = " + e.rule(*condition) + ";");
        else
          out.line("if !(" + e.rule(*condition) +
                   ").boolean()? { return Ok(false); }");
      }
    };
    out.open("pub fn check_where(ctx: &mut Context<'_>) -> EvalResult<bool>");
    conditions(schedule.source);
    for (const auto& q : rule.feasibility)
      out.line("if !(" + e.symbolic(*q) + ").boolean()? { return Ok(false); }");
    out.line("let where_locals = ::std::mem::take(&mut ctx.bindings.locals);");
    out.line("let result = derive(ctx);");
    out.line("ctx.bindings.locals = where_locals;");
    out.line("result?;");
    conditions(schedule.replacement);
    out.line("Ok(true)");
    out.close();
    // Each derived descriptor has a concrete schema at this boundary.
    for (const auto& derivation : rule.derivations) {
      if (!derivation.target) continue;
      const auto& type = p.types.at(derivation.value->type.value);
      const auto schema = *type.schema;
      std::string rust_type;
      for (const auto& d : names.dialects)
        for (auto sid : d.schemas)
          if (sid == schema)
            rust_type =
                root + "::dialects::" + d.module + "::" + names.schema(sid);
      out.open("pub fn derive_d" + num(derivation.target->value) +
               "(ctx: &mut Context<'_>) -> EvalResult<" + rust_type + ">");
      out.line("let value = " + e.rule(*derivation.value) + ";");
      out.line("AttributeValue::from_value(&value)");
      out.close();
    }
    out.open("pub fn derive(ctx: &mut Context<'_>) -> EvalResult<()>");
    for (const auto& derivation : rule.derivations) {
      if (derivation.target) {
        out.line("let attrs = derive_d" + num(derivation.target->value) +
                 "(ctx)?;");
        out.line("ctx.bindings.descriptors.insert(" +
                 num(derivation.target->value) + ", attrs.to_value());");
      } else
        out.line("let _ = " + e.rule(*derivation.value) + ";");
    }
    out.line("Ok(())");
    out.close();
    out.open(
        "pub fn prepare(ctx: &mut Context<'_>) -> "
        "EvalResult<Option<Replacement>>");
    out.line("if !check_where(ctx)? { return Ok(None); }");
    out.line("Ok(Some(" + rhs(p, *rule.rhs) + "))");
    out.close();
    out.open("pub fn definition() -> rt::RuleDefinition<" + root +
             "::Program>");
    out.open("rt::RuleDefinition");
    out.line("name: " + quote(module.qualified + "::" + rule.name) + ",");
    out.line("source: " +
             quote(rule.origin.definition.source_name + ":" +
                   num(rule.origin.definition.span.begin.line)) +
             ",");
    out.line("lhs: " + pattern(*rule.lhs) + ",");
    out.line("constraints: " + constraints(rule) + ",");
    out.line("prepare,");
    out.close();
    out.close();
    out.close();
  }
  return out.str();
}
}  // namespace tepl::codegen::rust
