#include "src/codegen/rust/verification_emitter.h"

#include <functional>
#include <map>

#include "src/codegen/rust/expression_emitter.h"
#include "src/codegen/rust/paths.h"
#include "src/core/semantics/solver.h"

namespace tepl::codegen::rust {
namespace {
std::string variable(std::size_t id) {
  return quote("?c" + std::to_string(id)) + ".parse::<Var>().ok()?";
}
std::string attribute(std::size_t id) {
  return "AttrVar::from(" + quote("d" + std::to_string(id)) + ")";
}
}  // namespace
void emitVerification(CodeWriter& out, const core::Program& program,
                      const core::Rule& rule, const Names& names,
                      const std::string& root) {
  auto runtime = paths::within(root, "builtins::verification");
  auto builtins = paths::within(root, "builtins");
  out.line(
      "/// Bounded early rejection using sparse inputs and checked rule "
      "expressions.");
  out.line(
      "/// Uses ideal arithmetic; passing does not prove arbitrary-rank "
      "validity.");
  out.open("pub fn verify_small(options: &" + runtime +
           "::VerificationOptions) -> " + runtime + "::VerificationReport");
  bool supported = rule.attribute_holes.empty() && rule.inference &&
                   rule.inference->semantic;
  if (supported) {
    for (auto& field : rule.inference->semantic->known) {
      const auto& f = program.attribute_schemas.at(field.schema.value)
                          .fields.at(field.field);
      supported &= f.type == "index" && f.list_depth == 1 && !f.optional;
    }
    // Pinned opaque attributes have no concrete validation representation yet.
    supported &= rule.inference->pinned.empty();
  }
  if (!supported) {
    out.line(runtime +
             "::VerificationReport::unknown(\"requires explicit attributes and "
             "supported polynomial semantics with index[] fields\")");
    out.close();
    return;
  }
  const auto& plan = *rule.inference->semantic;
  auto schemaAttrs = [&](core::AttributeSchemaId id,
                         const std::string& fields) {
    const auto& schema = program.attribute_schemas.at(id.value);
    for (auto& dialect : names.dialects)
      if (dialect.name == schema.dialect)
        return "OpAttrs::" + dialect.variant + "(" +
               paths::within(root, "dialects::" + dialect.module +
                                       "::OpAttrs::" + names.schema(id)) +
               " { " + fields + " })";
    throw std::logic_error("missing attribute dialect");
  };
  std::map<std::size_t, std::size_t> captures;
  std::map<std::size_t, const core::MatchOperation*> operations;
  std::function<void(const core::Pattern&, std::size_t)> source =
      [&](const auto& p, std::size_t node) {
        if (auto c = std::get_if<core::CapturePattern>(&p.value))
          captures[c->capture.value] = node;
        else if (auto b = std::get_if<core::BindPattern>(&p.value)) {
          captures[b->capture.value] = node;
          source(*b->expression, node);
        } else if (auto o = std::get_if<core::MatchOperation>(&p.value)) {
          operations[node] = o;
          for (std::size_t i = 0; i < o->operands.size(); ++i)
            source(*o->operands[i], plan.nodes.at(node).inputs.at(i));
        }
      };
  source(*rule.lhs, plan.source_root);
  out.line("const PLAN: &str = " + quote(core::semantics::encodePlan(plan)) +
           ";");
  out.open(runtime + "::run(PLAN, " + std::to_string(plan.captures.size()) +
           ", options, |sample|");
  out.line("let mut ctx = " + runtime + "::ValidationContext::default();");
  out.line("let mut dimensions = MetadataBindings::default();");
  out.line("let declarations = constraints();");
  for (auto& desc : rule.descriptors) {
    if (desc.kind != core::Descriptor::Kind::kCaptured) continue;
    auto schema = program.types.at(desc.type.value).schema;
    if (!schema) throw std::logic_error("validation descriptor needs schema");
    std::string fields;
    const auto& definition = program.attribute_schemas.at(schema->value);
    for (std::size_t f = 0; f < definition.fields.size(); ++f) {
      std::size_t index = plan.known.size();
      for (std::size_t i = 0; i < plan.known.size(); ++i)
        if (plan.known[i].descriptor == desc.id && plan.known[i].field == f) {
          index = i;
          break;
        }
      fields += names.field(*schema, f) + ": sample.known.get(" +
                std::to_string(index) + ")?.clone(), ";
    }
    auto name = descriptor(desc.id.value);
    out.line("let " + name + " = " + schemaAttrs(*schema, fields) + ";");
    out.line("ctx.attributes.insert(" + attribute(desc.id.value) + ", " + name +
             ".clone());");
  }
  // Evaluate the actual generated dtype/shape programs before evaluating rule
  // expressions. This prevents unreachable source configurations from causing
  // rejection.
  out.line("let mut source_info: Vec<Option<TensorInfo>> = vec![None; " +
           std::to_string(plan.source_end) + "];");
  for (std::size_t node = 0; node < plan.source_end; ++node) {
    const auto& n = plan.nodes.at(node);
    if (n.capture >= 0) {
      auto capture = plan.captures.at(n.capture).value;
      std::string selected_dtype = "options.dtype";
      for (auto& c : rule.constraints)
        if (c.capture.value == capture && c.dtype &&
            std::holds_alternative<core::DType>(*c.dtype))
          selected_dtype = dtype(std::get<core::DType>(*c.dtype));
      out.line("if !" + builtins + "::dtype::is_numeric(" + selected_dtype +
               ") { return None; }");
      out.line("source_info[" + std::to_string(node) +
               "] = Some(TensorInfo { shape: sample.node_shapes.get(" +
               std::to_string(node) + ")?.clone(), dtype: " + selected_dtype +
               " });");
    } else {
      const auto& operation = *operations.at(node);
      std::string operands = "vec![";
      for (auto input : n.inputs)
        operands += "source_info.get(" + std::to_string(input) +
                    ")?.as_ref()?.clone(),";
      operands += "]";
      auto attrs = operation.descriptor
                       ? descriptor(operation.descriptor->value)
                       : "OpAttrs::None";
      out.line("source_info[" + std::to_string(node) + "] = Some(" +
               paths::within(root, "analysis::infer_tensor") + "(" +
               names.operation(operation.operation, root) + ".into(), &(" +
               operands + "), &(" + attrs + ")).into_option()?);");
    }
  }
  for (auto& [capture, node] : captures) {
    out.line("let info = source_info.get(" + std::to_string(node) +
             ")?.as_ref()?.clone();");
    out.line("declarations.check_capture(" + variable(capture) +
             ", &info, &mut dimensions)?;");
    out.line("ctx.tensors.insert(" + variable(capture) + ", info);");
  }
  // Same ordered expressions and Option propagation as matching/application.
  for (auto& condition : rule.conditions)
    out.line("if !(" + emitExpression(program, names, *condition, builtins) +
             ") { return None; }");
  for (auto& derivation : rule.derivations) {
    auto name = descriptor(derivation.target->value);
    out.line("let " + name + " = " +
             emitExpression(program, names, *derivation.value, builtins) + ";");
    out.line("ctx.attributes.insert(" + attribute(derivation.target->value) +
             ", " + name + ".clone());");
  }
  std::string known = "vec![";
  for (auto& field : plan.known) {
    known += "(match &(" + descriptor(field.descriptor.value) + ") { " +
             schemaAttrs(field.schema, names.field(field.schema, field.field) +
                                           ": value, ..") +
             " => Some(value.clone()), _ => None })?,";
  }
  known += "]";
  out.line("let known = " + known + ";");
  std::map<std::size_t, const core::BuildOperation*> rhs_operations;
  std::function<void(const core::BuildExpr&, std::size_t)> rhs =
      [&](const auto& expression, std::size_t node) {
        if (auto op = std::get_if<core::BuildOperation>(&expression.value)) {
          rhs_operations[node] = op;
          for (std::size_t i = 0; i < op->operands.size(); ++i)
            rhs(*op->operands[i], plan.nodes.at(node).inputs.at(i));
        }
      };
  rhs(*rule.rhs, plan.rhs_root);
  out.line("let mut infos = source_info.clone();");
  out.line("infos.resize(" + std::to_string(plan.nodes.size()) + ", None);");
  auto inference = paths::within(root, "analysis::Inference");
  for (std::size_t node = plan.source_end; node < plan.nodes.size(); ++node) {
    const auto& n = plan.nodes.at(node);
    const auto& op = *rhs_operations.at(node);
    std::string operands = "vec![";
    for (auto input : n.inputs)
      operands +=
          "infos.get(" + std::to_string(input) + ")?.as_ref()?.clone(),";
    operands += "]";
    auto attrs =
        op.descriptor ? descriptor(op.descriptor->value) : "OpAttrs::None";
    out.open("match " + paths::within(root, "analysis::infer_tensor") + "(" +
             names.operation(op.operation, root) + ".into(), &(" + operands +
             "), &(" + attrs + "))");
    out.line(inference + "::Known(info) => infos[" + std::to_string(node) +
             "] = Some(info),");
    out.line(inference + "::Invalid(reason) => return Some(" + runtime +
             "::Prepared { known, pinned: vec![], metadata: " + runtime +
             "::MetadataCheck::Invalid(format!(\"RHS metadata invalid: {}\", "
             "reason)) }),");
    out.line(inference + "::Unknown => return Some(" + runtime +
             "::Prepared { known, pinned: vec![], metadata: " + runtime +
             "::MetadataCheck::Unknown(\"RHS metadata inference "
             "unavailable\".into()) }),");
    out.close();
  }
  out.line("let lhs = infos.get(" + std::to_string(plan.source_root) +
           ")?.as_ref()?;");
  out.line("let rhs = infos.get(" + std::to_string(plan.rhs_root) +
           ")?.as_ref()?;");
  out.line("let metadata = if lhs != rhs { " + runtime +
           "::MetadataCheck::Invalid(format!(\"output metadata differs: "
           "LHS={:?}; RHS={:?}\", lhs, rhs)) } else { " +
           runtime + "::MetadataCheck::Valid };");
  out.line("Some(" + runtime +
           "::Prepared { known, pinned: vec![], metadata })");
  out.close(")");
  out.close();
}
}  // namespace tepl::codegen::rust
