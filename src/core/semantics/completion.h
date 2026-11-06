#pragma once
#include <functional>
#include <unordered_map>

#include "src/core/checking/rule_ir.h"
#include "src/shared/catalog/completed.h"
namespace tepl::core::semantics {
// Converts materialized compiler metadata to the closed runtime vocabulary.
class Completion {
 public:
  using GroupResolver =
      std::function<symbolic::SymbolPtr(const OutputAxisGroupRef&)>;
  explicit Completion(GroupResolver groups = {}) : groups_(std::move(groups)) {}
  completed::RuleData rule(RuleData);
  completed::SymbolPtr symbol(const symbolic::SymbolPtr&);

 private:
  GroupResolver groups_;
  completed::TypedExprPtr expression(const TypedExprPtr&);
  // Group resolution can return temporary DAGs; retain their owners while
  // memoized.
  std::unordered_map<const symbolic::Symbol*,
                     std::pair<symbolic::SymbolPtr, completed::SymbolPtr>>
      symbols_;
  std::unordered_map<const TypedExpr*, completed::TypedExprPtr> expressions_;
};
}  // namespace tepl::core::semantics
