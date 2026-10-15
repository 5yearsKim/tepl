#pragma once

#include <optional>

#include "src/core/checking/ir.h"

namespace tepl::core::detail {

struct CheckingContext;
struct ExpandedRule;

std::optional<Rule> check(CheckingContext& context, const ExpandedRule& input);

}  // namespace tepl::core::detail
