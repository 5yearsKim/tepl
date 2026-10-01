#pragma once

#include <string>
#include <vector>

#include "src/ast.h"

namespace tepl {

struct HostDiagnostic {
  ast::SourceSpan span;
  std::string message;
};

struct HostTemplateResult {
  std::string source;
  std::vector<HostDiagnostic> diagnostics;

  bool ok() const { return diagnostics.empty(); }
};

// Discover host calls in where/derive and emit a Rust trait or an
// implementation template. This is deliberately separate from rule lowering.
HostTemplateResult generateHostTemplate(const ast::Program& program,
                                        bool implementation);

}  // namespace tepl
