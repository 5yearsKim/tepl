#pragma once

#include <string>

#include "src/core/syntax/ast.h"

namespace tepl {

// A readable view of AST structure for the command-line parser.
std::string formatAst(const syntax::ParsedModule& program);

}  // namespace tepl
