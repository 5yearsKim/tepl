#include "src/core/rule/descriptors.h"

#include <utility>

#include "src/core/rule/check_context.h"
#include "src/core/rule/expression.h"

namespace tepl::core::detail {

namespace {

DescriptorId addDescriptor(RuleCheckContext& context, const std::string& name,
                           Descriptor::Kind kind, const SourceOrigin& at) {
  const DescriptorId id{context.rule.descriptors.size()};
  context.rule.descriptors.push_back(
      {id, name, kind,
       context.analysis.types.concrete({TypeKind::kDescriptor}, at), at});
  context.scope.descriptors.emplace(name, id);
  return id;
}

}  // namespace

void registerDerivations(RuleCheckContext& context) {
  // RHS descriptor uses determine schemas before expressions are checked;
  // availability still advances in source order after each assignment.
  for (const auto& derivation : context.input.derivations) {
    if (context.scope.descriptors.contains(derivation.target)) {
      context.analysis.report(
          derivation.origin,
          "duplicate descriptor definition '@" + derivation.target + "'");
      continue;
    }
    addDescriptor(context, derivation.target, Descriptor::Kind::kDerived,
                  derivation.origin);
  }
}

std::optional<DescriptorId> checkDescriptor(RuleCheckContext& context,
                                            const ast::Operator& op,
                                            OpId operation, bool lhs,
                                            const SourceOrigin& at) {
  const auto schema =
      context.analysis.output.operations[operation.value].attributes;
  if (!op.attribute) {
    if (schema)
      context.analysis.report(
          at, "operation '" + op.name + "' requires an attribute descriptor");
    return std::nullopt;
  }
  if (!schema) {
    context.analysis.report(at,
                            "operation '" + op.name + "' has no attributes");
    return std::nullopt;
  }
  const auto& name = op.attribute->name;
  auto found = context.scope.descriptors.find(name);
  if (found == context.scope.descriptors.end()) {
    if (!lhs) {
      context.analysis.report(at, "unknown RHS descriptor '@" + name + "'");
      return std::nullopt;
    }
    const auto id =
        addDescriptor(context, name, Descriptor::Kind::kCaptured, at);
    context.scope.available_descriptors.insert(id.value);
    found = context.scope.descriptors.find(name);
  }
  const auto id = found->second;
  context.analysis.types.unify(
      context.rule.descriptors[id.value].type,
      context.analysis.types.concrete({TypeKind::kDescriptor, schema}, at), at);
  return id;
}

void checkDerivations(RuleCheckContext& context,
                      ExpressionChecker& expressions) {
  for (const auto& derivation : context.input.derivations) {
    const auto id = context.scope.descriptors.at(derivation.target);
    const auto& descriptor = context.rule.descriptors.at(id.value);
    if (descriptor.kind != Descriptor::Kind::kDerived ||
        context.scope.available_descriptors.contains(id.value))
      continue;
    if (auto checked = expressions.check(*derivation.value, descriptor.type))
      context.rule.derivations.push_back(
          {id, std::move(checked), derivation.origin});
    context.scope.available_descriptors.insert(id.value);
  }
}

}  // namespace tepl::core::detail
