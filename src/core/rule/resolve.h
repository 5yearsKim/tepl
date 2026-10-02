#pragma once

#include <map>

#include "src/ast/ast.h"
#include "src/core/resolution/source.h"

namespace tepl::core::detail {

struct AnalysisContext;
using RuleRegistry = std::map<DeclarationKey, const ast::Rule*>;

// Validate parameter declarations immediately, including unused abstract rules.
RuleRegistry resolveRules(AnalysisContext& context);

}  // namespace tepl::core::detail
