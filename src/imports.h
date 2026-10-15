#pragma once

#include <vector>

#include "src/ast/ast.h"
#include "src/parse.h"

namespace tepl {

// Load dialects and selected abstract rules relative to program.source_name.
// Appends definitions to dialects/imported_rules, without expanding
// inheritance. Reports missing, invalid, or cyclic imports at the import
// statement.
std::vector<Diagnostic> resolveImports(ast::Program& program);

// Load every .tepl file under dialects/, rules/, and graphs/, preserving
// scopes.
ParseResult loadProject(const std::string& directory);

// Load graph source and independently selected rule source (file or directory).
// A graph project directory contributes dialects/ and graphs/; --rules chooses
// its rule input explicitly. Files retain their own import/use scopes.
ParseResult loadRewriteSources(const std::string& graph,
                               const std::string& rules);
}  // namespace tepl
