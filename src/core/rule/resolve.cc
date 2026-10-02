#include "src/core/rule/resolve.h"

#include <set>

#include "src/core/analysis_context.h"

namespace tepl::core::detail {

namespace {

void checkSignatureType(AnalysisContext& context,
                        const ast::SignatureType& type,
                        ast::RuleParameterKind kind,
                        const std::string& source) {
  const auto resolved = resolveType(type.name);
  if (!resolved || (kind == ast::RuleParameterKind::kOperation &&
                    resolved->kind != TypeKind::kTensor))
    context.report(origin(source, type.span),
                   "unsupported parameter signature type '" + type.name + "'");
}

void registerRule(AnalysisContext& context, RuleRegistry& rules,
                  const ast::Rule& rule) {
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
      checkSignatureType(context, type, parameter.kind, rule.source_name);
    checkSignatureType(context, parameter.result_type, parameter.kind,
                       rule.source_name);
  }
}

}  // namespace

RuleRegistry resolveRules(AnalysisContext& context) {
  RuleRegistry rules;
  for (const auto& rule : context.input.rules)
    registerRule(context, rules, rule);
  for (const auto& rule : context.input.imported_rules)
    registerRule(context, rules, rule);
  return rules;
}

}  // namespace tepl::core::detail
