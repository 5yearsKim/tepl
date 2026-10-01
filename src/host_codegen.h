#pragma once

#include <string>
#include <vector>

#include "src/ast/ast.h"

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

// Discover host calls in each rule's where/derive sections and emit per-rule
// Rust function traits or an implementation template. This is deliberately
// separate from rule lowering.
HostTemplateResult generateHostTemplate(const ast::Program& program,
                                        bool implementation);

}  // namespace tepl
