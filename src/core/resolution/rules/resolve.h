#pragma once

#include <map>

#include "src/core/syntax/ast.h"
#include "src/shared/source/identity.h"

namespace tepl::core::detail {

struct ResolutionContext;
using RuleRegistry = std::map<DeclarationKey, const syntax::Rule*>;

// Validate parameter declarations immediately, including unused abstract rules.
RuleRegistry resolveRules(ResolutionContext& context);

}  // namespace tepl::core::detail
