#pragma once
#include "src/core/checking/ir.h"
#include "src/core/checking/types/type_inference.h"
#include "src/core/resolution/program.h"
namespace tepl::core::detail {
struct CheckingContext {
  explicit CheckingContext(const resolution::ResolvedProgram& resolved)
      : resolved(resolved),
        input(*resolved.data().sources),
        types(diagnostics) {
    output.sources = resolved.data().source_names;
  }
  const resolution::ResolvedProgram& resolved;
  const core::resolution::SourceProject& input;
  ProgramData output;
  std::vector<Diagnostic> diagnostics;
  TypeInference types;
  void report(SourceOrigin, std::string, std::vector<SourceLocation> = {});
};
}  // namespace tepl::core::detail
