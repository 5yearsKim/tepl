#pragma once
#include "src/core/semantics/inspection.h"
#include "src/core/semantics/verified_ir.h"
namespace tepl::core::semantics {
struct Materialized {
  VerifiedData ir;
  std::vector<std::vector<completed::SymbolPtr>> axes;
  std::vector<VerificationKind> kinds;
};
Materialized materialize(InspectionData);
}  // namespace tepl::core::semantics
