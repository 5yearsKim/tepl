#pragma once

#include <optional>

#include "src/ast/ast.h"
#include "src/core/ir.h"

namespace tepl::core::detail {

struct RuleCheckContext;
class ExpressionChecker;

// Register before RHS checking; evaluate after conditions so only captured
// descriptors are available in where and derivations advance in source order.
void registerDerivations(RuleCheckContext& context);
void checkDerivations(RuleCheckContext& context,
                      ExpressionChecker& expressions);
std::optional<DescriptorId> checkDescriptor(RuleCheckContext& context,
                                            const ast::Operator& op,
                                            OpId operation, bool lhs,
                                            const SourceOrigin& at);

}  // namespace tepl::core::detail
