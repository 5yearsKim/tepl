#pragma once

#include "src/core/analysis_context.h"
#include "src/core/rule/expanded_rule.h"
#include "src/core/rule/scope.h"

namespace tepl::core::detail {

// Lives for one rule check. ExpressionChecker receives read-only rule/scope
// views.
struct RuleCheckContext {
  RuleCheckContext(AnalysisContext& analysis, const ExpandedRule& input)
      : analysis(analysis),
        input(input),
        tensor(analysis.types.concrete({TypeKind::kTensor}, input.origin)) {
    rule.id = RuleId{analysis.output.rules.size()};
    rule.name = input.name;
    rule.origin = input.origin;
  }

  AnalysisContext& analysis;
  const ExpandedRule& input;
  Rule rule;
  TypeId tensor;
  RuleScope scope;
};

}  // namespace tepl::core::detail
