#include "src/core/builtins/axes.h"

namespace tepl::core::builtins {
namespace {
using T = AxisType;
using I = intrinsics::Intrinsic;
constexpr T kOperand[] = {T::Operand};
constexpr T kPositions[] = {T::Positions};
constexpr T kInteger[] = {T::Integer};
constexpr T kSequence[] = {T::Sequence};
constexpr T kSequences[] = {T::Sequence, T::Sequence};
constexpr T kPositionPair[] = {T::Positions, T::Positions};
constexpr T kPositionInteger[] = {T::Positions, T::Integer};
constexpr T kPermute[] = {T::Axes, T::Positions};
constexpr T kPlace[] = {T::Axes, T::Positions, T::Axes};
constexpr AxisSignature kAxes[] = {
    {"len", kSequence, T::Integer, Builtin::kLen},
    {"range", kInteger, T::Positions, Builtin::kRange},
    {"concat", kSequences, T::Sequence, Builtin::kConcat, true},
    {"gather", kPositionPair, T::Positions, Builtin::kGather},
    {"is_valid_axis_list", kPositionInteger, T::Boolean,
     Builtin::kIsValidAxisList},
    {"ordered_complement", kPositionPair, T::Positions, Builtin::kExclude},
    {"permute", kPermute, T::Axes, Builtin::kGather},
    {"all_axes", kOperand, T::Positions, AxisQuery::AllAxes},
    {"rank", kOperand, T::Integer, AxisQuery::Rank},
    {"shape", kOperand, T::Positions, AxisQuery::Shape},
    {"place", kPlace, T::Axes, I::kPlaceAxes},
    {"extent_add", kPositionPair, T::Positions, I::kExtentAdd},
    {"extent_sub", kPositionPair, T::Positions, I::kExtentSub},
    {"extent_mul", kPositionPair, T::Positions, I::kExtentMul},
    {"extent_floor_div", kPositionPair, T::Positions, I::kExtentFloorDiv},
    {"extent_le", kPositionPair, T::Boolean, I::kExtentLe},
    {"scale_extents", kPositionInteger, T::Positions, I::kScaleExtents},
    {"offset_extents", kPositionInteger, T::Positions, I::kOffsetExtents},
    {"is_valid_extents", kPositions, T::Boolean, I::kIsValidExtents},
    {"is_positive_extents", kPositions, T::Boolean, I::kIsPositiveExtents},
};
}  // namespace
std::span<const AxisSignature> axisCatalog() { return kAxes; }
const AxisSignature* resolveAxisBuiltin(std::string_view name) {
  for (const auto& entry : kAxes)
    if (entry.name == name) return &entry;
  return nullptr;
}
}  // namespace tepl::core::builtins
