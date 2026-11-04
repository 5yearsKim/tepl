#pragma once

#include <functional>
#include <string>

#include "src/shared/catalog/completed.h"

namespace tepl::core::completed {
void validateAxisConditions(const RuleData&);
bool usesAxisQueries(const TypedExpr&);
struct WhereSchedule {
  std::vector<TypedExprPtr> source, replacement;
};
// Shared by validation and execution; dependent lets keep their where scope.
WhereSchedule scheduleWhere(const RuleData&);
}  // namespace tepl::core::completed
