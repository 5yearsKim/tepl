#pragma once

#include <optional>

#include "src/ast/ast.h"

namespace tepl {

// Match the rule's LHS at the input root and substitute its captures into the
// RHS. Returns nullopt on a mismatch. Invalid simple rules throw
// std::invalid_argument; call validateSimpleRule for detailed diagnostics.
// The returned tree owns copies of captured subtrees.
std::optional<ast::GraphExprPtr> rewriteOnce(const ast::Rule& rule,
                                             const ast::GraphExpr& input);

}  // namespace tepl
