#pragma once

#include "src/core/checking/context.h"
#include "src/core/checking/rules/scope.h"
#include "src/core/resolution/rules/expanded_rule.h"

namespace tepl::core::detail {

struct RuleCheckContext {
  RuleCheckContext(CheckingContext& analysis, const ExpandedRule& input)
      : analysis(analysis),
        input(input),
        tensor(analysis.types.concrete({TypeKind::kTensor}, input.origin)) {
    rule.id = RuleId{analysis.output.rules.size()};
    rule.name = input.name;
    rule.origin = input.origin;
  }

  CheckingContext& analysis;
  const ExpandedRule& input;
  Rule rule;
  TypeId tensor;
  RuleScope scope;
};

}  // namespace tepl::core::detail
