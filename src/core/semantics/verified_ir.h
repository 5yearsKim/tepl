#pragma once
#include "src/core/tables.h"
#include "src/shared/catalog/completed.h"
namespace tepl::core::semantics {
enum class VerificationKind { Structural, Semantic };
struct VerifiedData : ProgramTables {
  std::vector<completed::RuleData> rules;
};
}  // namespace tepl::core::semantics
