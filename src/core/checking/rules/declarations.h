#pragma once

namespace tepl::core::detail {

struct RuleCheckContext;

// Requires LHS captures to have been collected.
void checkDeclarations(RuleCheckContext& context);

}  // namespace tepl::core::detail
