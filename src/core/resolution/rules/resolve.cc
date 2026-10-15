#include "src/core/resolution/rules/resolve.h"

#include <set>

#include "src/core/resolution/context.h"

namespace tepl::core::detail {

namespace {

void checkSignatureType(ResolutionContext& context,
                        const syntax::SignatureType& type,
                        const std::string& source) {
  const auto resolved = resolveType(type.name);
  if (!resolved || resolved->kind != TypeKind::kTensor)
    context.report(origin(source, type.span),
                   "unsupported parameter signature type '" + type.name + "'");
}

void registerRule(ResolutionContext& context, RuleRegistry& rules,
                  const syntax::Rule& rule) {
  if (!rules
           .emplace(DeclarationKey{sourceKey(rule.source_name), rule.name},
                    &rule)
           .second)
    context.report(origin(rule.source_name, rule.span),
                   "duplicate rule '" + rule.name + "'");
  std::set<std::string> names;
  for (const auto& parameter : rule.parameters) {
    if (!names.insert(parameter.name).second)
      context.report(origin(rule.source_name, parameter.span),
                     "duplicate rule parameter '" + parameter.name + "'");
    for (const auto& type : parameter.operand_types)
      checkSignatureType(context, type, rule.source_name);
    checkSignatureType(context, parameter.result_type, rule.source_name);
  }
}

}  // namespace

RuleRegistry resolveRules(ResolutionContext& context) {
  RuleRegistry rules;
  for (const auto& rule : context.input.declarations.rules)
    registerRule(context, rules, rule);
  for (const auto& rule : context.input.imported_rules)
    registerRule(context, rules, rule);
  return rules;
}

}  // namespace tepl::core::detail
