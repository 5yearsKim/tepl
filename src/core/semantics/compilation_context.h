#pragma once
#include "src/core/semantics/inspection.h"
#include "src/core/semantics/options.h"
#include "src/shared/diagnostics/diagnostic.h"
namespace tepl::core::semantics {
struct CompilationContext {
  InspectionData& output;
  const InspectionOptions& options;
  std::vector<Diagnostic> diagnostics;
  void report(SourceOrigin at, std::string message) {
    diagnostics.push_back({std::move(at), std::move(message), {}});
  }
};
bool compile(CompilationContext&, InspectedRule&);
}  // namespace tepl::core::semantics
