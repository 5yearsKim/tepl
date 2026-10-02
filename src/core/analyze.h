#pragma once

#include <optional>
#include <vector>

#include "src/core/diagnostic.h"
#include "src/core/ir.h"

namespace tepl::ast {
struct Program;
}

namespace tepl::core {

struct AnalysisResult {
  std::optional<Program> program;
  std::vector<Diagnostic> diagnostics;
  bool ok() const { return program.has_value(); }
};

// Input must have its imports loaded with resolveImports; unloaded imports
// produce diagnostics. Never mutates the
// AST or its shared expression nodes. Only returns an IR on complete success.
// V1 supports declared tensor operations; tuples/projections are diagnosed.
AnalysisResult analyze(const ast::Program& input);

}  // namespace tepl::core
