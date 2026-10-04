#pragma once

#include <optional>
#include <string_view>
#include <variant>

#include "src/core/types.h"

namespace tepl::core {

// Common policies require equal operand dtypes, without implicit promotion.
enum class CommonDTypePolicy { kSame, kSameNumeric, kSameFloat };

// A concrete DType fixes the output dtype without restricting operand dtypes.
// An absent operation policy provides no dtype inference.
using DTypePolicy = std::variant<CommonDTypePolicy, DType>;

std::optional<DTypePolicy> resolveDTypePolicy(std::string_view name);
std::string_view dtypePolicyName(const DTypePolicy& policy);

}  // namespace tepl::core
