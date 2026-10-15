#pragma once
#include <optional>
#include <string>

#include "src/core/resolution/project.h"
#include "src/shared/diagnostics/diagnostic.h"
namespace tepl {
std::string formatAst(const core::resolution::SourceProject&);
class SourceLoader {
 public:
  virtual ~SourceLoader() = default;
  virtual std::optional<std::string> read(const std::string& identity) = 0;
};
class FileSourceLoader final : public SourceLoader {
 public:
  std::optional<std::string> read(const std::string&) override;
};
struct LoadResult {
  std::optional<core::resolution::SourceProject> program;
  std::vector<Diagnostic> diagnostics;
  std::size_t rule_count = 0;
  std::string tree;
  bool ok() const { return program.has_value() && diagnostics.empty(); }
};
LoadResult loadImports(const syntax::ParsedModule&, SourceLoader&);
LoadResult loadImports(const syntax::ParsedModule&);
LoadResult loadProject(const std::string&);
LoadResult loadProject(const std::string&, SourceLoader&);
// Explicit roots support virtual projects without filesystem discovery.
LoadResult loadProject(const std::vector<SourceFile>&, SourceLoader&);
LoadResult loadRewriteSources(const std::string&, const std::string&);
LoadResult loadRewriteSources(const std::string&, const std::string&,
                              SourceLoader&);
}  // namespace tepl
