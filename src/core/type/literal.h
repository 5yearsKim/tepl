#pragma once

#include <string_view>

#include "src/core/type/types.h"

namespace tepl::core {

// Validate parsed literals without changing their spelling. Tensor floating
// format rounding/representability remains the host's responsibility.
bool validGraphLiteral(std::string_view spelling, DType dtype);
bool validHostLiteral(std::string_view spelling, TypeKind type);

}  // namespace tepl::core
