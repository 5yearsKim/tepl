#pragma once
#include <string>

#include "src/core/checking/ir.h"
#include "src/core/semantics/ir.h"
namespace tepl::core::semantics {
// Bounded restriction search: each suggestion is independently proved and has
// a legal source witness. Rendering preserves user conditions and derivations.
std::string suggestRestrictions(const Plan&, const Rule&, ProgramTables&);
std::string restrictedRule(const Plan&, const Rule&, const ProgramTables&,
                           const std::vector<TypedExprPtr>&);
}  // namespace tepl::core::semantics
