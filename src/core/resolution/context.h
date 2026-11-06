#pragma once
#include <map>
#include <memory>

#include "src/core/resolution/file_scope.h"
#include "src/core/resolution/loading.h"
#include "src/core/resolution/program.h"
#include "src/core/syntax/ast.h"
#include "src/shared/catalog/data.h"
#include "src/shared/diagnostics/result.h"
namespace tepl::core::detail {
struct ResolutionContext {
  explicit ResolutionContext(const core::resolution::SourceProject& input)
      : input(input) {}
  const core::resolution::SourceProject& input;
  struct {
    std::vector<resolution::ResolvedOperation> operations;
    std::vector<std::string> sources;
  } output;
  std::map<std::string, FileScope> scopes;
  std::vector<Diagnostic> diagnostics;
  void report(SourceOrigin at, std::string message,
              std::vector<SourceLocation> related = {});
  std::optional<OpId> operation(const std::string&, const std::string&,
                                const SourceOrigin&);
};
}  // namespace tepl::core::detail
