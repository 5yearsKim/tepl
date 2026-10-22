#pragma once

#include <functional>
#include <string>

#include "src/shared/catalog/completed.h"

namespace tepl::core::completed {
// The one source of truth for the generated feasibility formula. Both the
// compiler and executable loader regenerate it from the admitted schedule.
std::vector<TypedExprPtr> axisConditions(const inference::Program&,
                                         const SourceOrigin&,
                                         const std::function<TypeId(Type)>&);
std::string axisWhere(const inference::Program&);
// Recognizes required conjuncts after expanding local bindings. Additional
// user conditions are permitted; a missing obligation is a compile error.
void validateAxisConditions(const RuleData&);
bool usesAxisQueries(const TypedExpr&);
struct WhereSchedule {
  std::vector<TypedExprPtr> source, replacement;
};
// Shared by validation and execution; dependent lets keep their where scope.
WhereSchedule scheduleWhere(const RuleData&);
}  // namespace tepl::core::completed
