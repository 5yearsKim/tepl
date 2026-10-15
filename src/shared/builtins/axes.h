#pragma once

#include <span>
#include <string_view>
#include <variant>

#include "src/shared/builtins/catalog.h"
#include "src/shared/builtins/intrinsics/catalog.h"

namespace tepl::core::builtins {
// Sequence is a signature variable shared by all occurrences within a call.
enum class AxisType { Integer, Boolean, Positions, Axes, Operand, Sequence };
enum class AxisQuery { AllAxes, Rank, Shape };
using AxisTarget = std::variant<Builtin, intrinsics::Intrinsic, AxisQuery>;
struct AxisSignature {
  std::string_view name;
  std::span<const AxisType> arguments;
  AxisType result;
  AxisTarget target;
  bool variadic = false;
  bool acceptsArity(std::size_t count) const {
    return variadic ? count >= arguments.size() : count == arguments.size();
  }
};
// Public functions callable only in operation axes blocks. Shared functions
// reference the same Builtin ID; specialized operations lower explicitly.
std::span<const AxisSignature> axisCatalog();
const AxisSignature* resolveAxisBuiltin(std::string_view);
}  // namespace tepl::core::builtins
