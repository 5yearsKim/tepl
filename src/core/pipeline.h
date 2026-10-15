#pragma once
#include "src/core/checking/check.h"
#include "src/core/resolution/resolve.h"
#include "src/core/semantics/inspect.h"
#include "src/core/semantics/verify.h"
#include "src/core/syntax/parse.h"
#include "src/executable/program.h"
namespace tepl {
class SourceLoader;
}
namespace tepl::core {
struct CompileRequest {
  SourceFile source;
  SemanticOptions semantics;
};
Result<executable::ExecutableProgram> compile(const CompileRequest&,
                                              SourceLoader&);
using InspectionOptions = semantics::InspectionOptions;
using InspectionResult = Result<semantics::InspectionData>;
InspectionResult inspect(const core::resolution::SourceProject&,
                         const InspectionOptions& = {});
InspectionResult inspect(const syntax::ParsedModule&,
                         const InspectionOptions& = {});
Result<executable::ExecutableProgram> compile(const syntax::ParsedModule&,
                                              const SemanticOptions& = {});
Result<executable::ExecutableProgram> compile(
    const core::resolution::SourceProject&, const SemanticOptions& = {});
}  // namespace tepl::core
