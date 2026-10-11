#pragma once

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/ast/ast.h"
#include "src/core/analyze.h"
#include "src/core/diagnostic.h"
#include "src/core/ir.h"
#include "src/core/resolution/file_scope.h"
#include "src/core/type/type_inference.h"

namespace tepl::core::detail {

// Program-wide state. Expanded trees and mutable rule symbols belong to rule/.
struct AnalysisContext {
  explicit AnalysisContext(const ast::Program& input,
                           AnalysisOptions options = {})
      : input(input), options(options), types(diagnostics) {}
  AnalysisContext(const AnalysisContext&) = delete;
  AnalysisContext& operator=(const AnalysisContext&) = delete;
  const ast::Program& input;
  AnalysisOptions options;
  Program output;
  std::vector<Diagnostic> diagnostics;
  std::map<std::string, FileScope> scopes;
  std::unordered_map<std::string, HostFunctionId> host_names;
  TypeInference types;

  void report(SourceOrigin origin, std::string message,
              std::vector<SourceLocation> related = {});
  HostFunctionId host(const std::string& name, std::size_t arity,
                      const SourceOrigin& origin);
  std::optional<OpId> operation(const std::string& name,
                                const std::string& source,
                                const SourceOrigin& origin);
};

}  // namespace tepl::core::detail
