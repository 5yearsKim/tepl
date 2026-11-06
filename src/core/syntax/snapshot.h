#pragma once
#include "src/core/syntax/ast.h"
namespace tepl::syntax {
// Deep ownership boundary for syntax supplied by parsers or API callers.
ParsedModule snapshot(const ParsedModule&);
Rule snapshot(const Rule&);
}  // namespace tepl::syntax
