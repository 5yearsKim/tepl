#pragma once

#include <optional>
#include <string>
#include <vector>

#include "src/core/diagnostic.h"
#include "src/core/ir.h"

namespace tepl::ast {
struct Program;
}

namespace tepl::core {

enum class SemanticPolicy { kStrict, kAlgebraic };
struct AnalysisOptions {
  SemanticPolicy semantic_policy = SemanticPolicy::kStrict;
  unsigned solver_timeout_ms = 1000;
  // Optional ideal-arithmetic rejection on small concrete metadata.
  bool verify_small = false;
  // Diagnostic/test inspection may retain unproved plans; it does not accept
  // them as rewrite rules. Normal analysis rejects an unproved index rule.
  bool inspect_index_rules = false;
  // Limit index proof checking to this rule (the CLI's --index-rule selector).
  std::string index_rule;
};

struct AnalysisResult {
  std::optional<Program> program;
  std::vector<Diagnostic> diagnostics;
  bool ok() const { return program.has_value(); }
};

// Input must have its imports loaded with resolveImports; unloaded imports
// produce diagnostics. Never mutates the
// AST or its shared expression nodes. Only returns an IR on complete success.
// V1 supports declared tensor operations; tuples/projections are diagnosed.
AnalysisResult analyze(const ast::Program& input,
                       const AnalysisOptions& options = {});

}  // namespace tepl::core
