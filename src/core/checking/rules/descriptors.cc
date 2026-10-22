#include "src/core/checking/rules/descriptors.h"

#include "src/core/checking/rules/check_context.h"
#include "src/core/checking/rules/expression.h"

namespace tepl::core::detail {
namespace {
DescriptorId addDescriptor(RuleCheckContext& context, const std::string& name,
                           Descriptor::Kind kind, const SourceOrigin& at,
                           AttributeSchemaId schema) {
  const DescriptorId id{context.rule.descriptors.size()};
  context.rule.descriptors.push_back(
      {id, name, kind,
       context.analysis.types.concrete({TypeKind::kDescriptor, schema}, at),
       at});
  return id;
}
}  // namespace

void registerDerivations(RuleCheckContext& context) {
  // Validate providers before RHS checking. Each named RHS operation supplies
  // the schema for exactly one explicit attribute definition.
  for (const auto& d : context.input.derivations) {
    if (d.local) continue;
    if (context.scope.nodes.contains(d.target)) {
      context.analysis.report(d.origin, "source node attributes are read-only");
      continue;
    }
    const syntax::GraphExpr* node = nullptr;
    auto find = [&](auto&& self, const syntax::GraphExpr& e) -> void {
      if (auto* op = std::get_if<syntax::Operator>(&e.value)) {
        if (op->label == d.target) node = &e;
        for (const auto& child : op->operands) self(self, *child);
      }
    };
    find(find, *context.input.rhs);
    if (!node) {
      context.analysis.report(d.origin, "unknown RHS node '" + d.target + "'");
      continue;
    }
    const auto& op = std::get<syntax::Operator>(node->value);
    if (op.descriptor_node) {
      context.analysis.report(d.origin,
                              "RHS node has competing attribute definitions");
      continue;
    }
    if (context.scope.attribute_definitions.contains(d.target)) {
      context.analysis.report(d.origin, "duplicate attribute definition for '" +
                                            d.target + ".attrs'");
      continue;
    }
    auto operation =
        context.input.graphs.at(node->resolution_id).operation.value();
    const auto schema =
        context.analysis.output.operations[operation.value].attributes;
    if (!schema) {
      context.analysis.report(d.origin, "node has no attribute schema");
      continue;
    }
    auto id = addDescriptor(context, d.target, Descriptor::Kind::kDerived,
                            d.origin, *schema);
    context.scope.attribute_definitions.emplace(d.target, id);
  }
}

std::optional<DescriptorId> checkDescriptor(RuleCheckContext& context,
                                            const syntax::Operator& op,
                                            OpId operation, bool lhs,
                                            const SourceOrigin& at) {
  const auto schema =
      context.analysis.output.operations[operation.value].attributes;
  if (op.descriptor_node) {
    auto source = context.scope.nodes.find(*op.descriptor_node);
    if (!schema) {
      context.analysis.report(at,
                              "operation '" + op.name + "' has no attributes");
      return {};
    }
    if (source == context.scope.nodes.end() || !source->second.descriptor ||
        (lhs && source->second.rhs)) {
      context.analysis.report(at,
                              "inline attributes require an available named "
                              "source or derived node");
      return {};
    }
    auto id = *source->second.descriptor;
    if (context.rule.descriptors[id.value].kind ==
        Descriptor::Kind::kInferred) {
      context.analysis.report(
          at, "inferred RHS attributes cannot define another descriptor");
      return {};
    }
    context.analysis.types.unify(
        context.rule.descriptors[id.value].type,
        context.analysis.types.concrete({TypeKind::kDescriptor, schema}, at),
        at);
    return id;
  }
  if (!schema) return {};
  if (!lhs && op.label) {
    auto provider = context.scope.attribute_definitions.find(*op.label);
    if (provider != context.scope.attribute_definitions.end())
      return provider->second;
  }
  auto id = addDescriptor(
      context,
      op.label ? *op.label
               : (lhs ? "capture#" : "inferred#") +
                     std::to_string(context.rule.descriptors.size()),
      lhs ? Descriptor::Kind::kCaptured : Descriptor::Kind::kInferred, at,
      *schema);
  if (lhs)
    context.scope.available_descriptors.insert(id.value);
  else
    context.rule.attribute_holes.push_back({id, *schema});
  return id;
}

void checkDerivations(RuleCheckContext& context,
                      ExpressionChecker& expressions) {
  // All providers already have their RHS schemas. Values become available in
  // declaration order, so forward references and cycles remain errors.
  for (const auto& d : context.input.derivations) {
    if (d.local) {
      if (auto checked = expressions.local(d.target, *d.value, d.origin))
        context.rule.derivations.push_back({{}, checked, d.origin});
      continue;
    }
    auto provider = context.scope.attribute_definitions.find(d.target);
    if (provider == context.scope.attribute_definitions.end()) continue;
    const auto id = provider->second;
    if (context.scope.available_descriptors.contains(id.value)) continue;
    if (auto checked = expressions.check(
            *d.value, context.rule.descriptors[id.value].type))
      context.rule.derivations.push_back({id, std::move(checked), d.origin});
    context.scope.available_descriptors.insert(id.value);
  }
}
}  // namespace tepl::core::detail
