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

// Load every .tepl file under dialects/ and rules/, preserving each file scope.
ParseResult loadProject(const std::string& directory);
}  // namespace tepl
