#include "src/core/pipeline.h"

#include "src/core/lowering/lower.h"
#include "src/core/resolution/loading.h"
namespace tepl::core {
InspectionResult inspect(const core::resolution::SourceProject& input,
                         const InspectionOptions& options) {
  auto resolved = resolution::resolve(input);
  if (!resolved.ok()) return {{}, std::move(resolved.diagnostics)};
  auto typed = checking::check(*resolved.program);
  if (!typed.ok()) return {{}, std::move(typed.diagnostics)};
  return semantics::inspect(*typed.program, options);
}
Result<executable::ExecutableProgram> compile(
    const core::resolution::SourceProject& input,
    const SemanticOptions& options) {
  auto resolved = resolution::resolve(input);
  if (!resolved.ok()) return {{}, std::move(resolved.diagnostics)};
  auto typed = checking::check(*resolved.program);
  if (!typed.ok()) return {{}, std::move(typed.diagnostics)};
  auto verified = semantics::verify(*typed.program, options);
  if (!verified.ok()) return {{}, std::move(verified.diagnostics)};
  return lowering::lower(*verified.program);
}
InspectionResult inspect(const syntax::ParsedModule& module,
                         const InspectionOptions& options) {
  auto loaded = loadImports(module);
  if (!loaded.ok()) return {{}, std::move(loaded.diagnostics)};
  return inspect(*loaded.program, options);
}
Result<executable::ExecutableProgram> compile(
    const syntax::ParsedModule& module, const SemanticOptions& options) {
  auto loaded = loadImports(module);
  if (!loaded.ok()) return {{}, std::move(loaded.diagnostics)};
  return compile(*loaded.program, options);
}

Result<executable::ExecutableProgram> compile(const CompileRequest& request,
                                              SourceLoader& loader) {
  auto parsed = parse(request.source);
  if (!parsed.ok()) return {{}, std::move(parsed.diagnostics)};
  auto loaded = loadImports(*parsed.program, loader);
  if (!loaded.ok()) return {{}, std::move(loaded.diagnostics)};
  return compile(*loaded.program, request.semantics);
}

}  // namespace tepl::core
