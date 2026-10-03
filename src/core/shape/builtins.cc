#include "src/core/shape/builtins.h"

namespace tepl::core::shape {
namespace {
using S = SignatureType;
constexpr S kList[] = {S::kListT};
constexpr S kInteger[] = {S::kInteger};
constexpr S kLists[] = {S::kListT, S::kListT};
constexpr S kGather[] = {S::kListT, S::kIntegerList};
constexpr S kSlice[] = {S::kListT, S::kInteger, S::kInteger};
constexpr S kReplace[] = {S::kListT, S::kInteger, S::kT};
constexpr S kIntegers[] = {S::kIntegerList};
constexpr S kBooleans[] = {S::kBooleanList};
constexpr S kContains[] = {S::kListT, S::kT};
constexpr S kAxes[] = {S::kIntegerList, S::kInteger};
constexpr S kShapes[] = {S::kIntegerList, S::kIntegerList};
constexpr S kPair[] = {S::kInteger, S::kInteger};
constexpr BuiltinSignature kBuiltins[] = {
    {Builtin::kLen, "len", kList, S::kInteger},
    {Builtin::kRange, "range", kInteger, S::kIntegerList},
    {Builtin::kConcat, "concat", kLists, S::kListT, true},
    {Builtin::kGather, "gather", kGather, S::kListT},
    {Builtin::kExclude, "exclude", kLists, S::kListT},
    {Builtin::kSlice, "slice", kSlice, S::kListT},
    {Builtin::kReplace, "replace", kReplace, S::kListT},
    {Builtin::kSum, "sum", kIntegers, S::kInteger},
    {Builtin::kProduct, "product", kIntegers, S::kInteger},
    {Builtin::kAll, "all", kBooleans, S::kBoolean},
    {Builtin::kAny, "any", kBooleans, S::kBoolean},
    {Builtin::kContains, "contains", kContains, S::kBoolean},
    {Builtin::kIsValidAxisList, "is_valid_axis_list", kAxes, S::kBoolean},
    {Builtin::kIsDisjoint, "is_disjoint", kLists, S::kBoolean},
    {Builtin::kBroadcastShape, "broadcast_shape", kShapes, S::kIntegerList},
    {Builtin::kMin, "min", kPair, S::kInteger},
    {Builtin::kMax, "max", kPair, S::kInteger},
    {Builtin::kFloorDiv, "floor_div", kPair, S::kInteger},
    {Builtin::kCeilDiv, "ceil_div", kPair, S::kInteger},
};
}  // namespace

const BuiltinSignature* resolveBuiltin(std::string_view name) {
  for (const auto& builtin : kBuiltins)
    if (builtin.name == name) return &builtin;
  return nullptr;
}

std::string_view builtinName(Builtin builtin) {
  for (const auto& signature : kBuiltins)
    if (signature.builtin == builtin) return signature.name;
  return "<invalid builtin>";
}

}  // namespace tepl::core::shape
