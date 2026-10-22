#pragma once
#include <string>

#include "src/core/checking/ir.h"
#include "src/core/semantics/ir.h"
namespace tepl::core::semantics {
// Prints a complete source rule: labels, original conditions/derivations, and
// the exact generated obligations. Internal numeric query IDs never leak.
std::string suggestRule(const Plan&, const Rule&, const ProgramTables&);
}  // namespace tepl::core::semantics
