#pragma once

#include <string>

#include "src/ast.h"

namespace tepl {

// A readable view of AST structure for the command-line parser.
std::string formatAst(const ast::Program& program);

}  // namespace tepl
