#include "src/core/builtins/catalog.h"

namespace tepl::core::builtins {
namespace {
using S = SignatureType;
constexpr unsigned kDTypeContexts = static_cast<unsigned>(Context::kDType) |
                                    static_cast<unsigned>(Context::kWhere) |
                                    static_cast<unsigned>(Context::kDerive);
constexpr S kDType[] = {S::kDType};
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
    {Builtin::kIsFloat, "is_float", kDType, S::kBoolean, false, Domain::kDType,
     14},
    {Builtin::kIsInteger, "is_integer", kDType, S::kBoolean, false,
     Domain::kDType, kDTypeContexts},
    {Builtin::kIsSignedInteger, "is_signed_integer", kDType, S::kBoolean, false,
     Domain::kDType, kDTypeContexts},
    {Builtin::kIsUnsignedInteger, "is_unsigned_integer", kDType, S::kBoolean,
     false, Domain::kDType, kDTypeContexts},
    {Builtin::kIsNumeric, "is_numeric", kDType, S::kBoolean, false,
     Domain::kDType, kDTypeContexts},
    {Builtin::kLen, "len", kList, S::kInteger},
    {Builtin::kRange, "range", kInteger, S::kIntegerList},
    {Builtin::kConcat, "concat", kLists, S::kListT, true},
    {Builtin::kGather, "gather", kGather, S::kListT},
    {Builtin::kExclude, "exclude", kLists, S::kListT},
    {Builtin::kSlice, "slice", kSlice, S::kListT},
    {Builtin::kReplace, "replace", kReplace, S::kListT},
    {Builtin::kSum, "sum", kIntegers, S::kInteger},
    {Builtin::kProduct, "product", kIntegers, S::kInteger},
    {Builtin::kAll, "all", kBooleans, S::kBoolean, false, Domain::kCommon,
     static_cast<unsigned>(Context::kShape) |
         static_cast<unsigned>(Context::kDType)},
    {Builtin::kAny, "any", kBooleans, S::kBoolean, false, Domain::kCommon,
     static_cast<unsigned>(Context::kShape) |
         static_cast<unsigned>(Context::kDType)},
    {Builtin::kContains, "contains", kContains, S::kBoolean},
    {Builtin::kIsValidAxisList, "is_valid_axis_list", kAxes, S::kBoolean, false,
     Domain::kShape},
    {Builtin::kIsDisjoint, "is_disjoint", kLists, S::kBoolean},
    {Builtin::kBroadcastShape, "broadcast_shape", kShapes, S::kIntegerList,
     false, Domain::kShape},
    {Builtin::kMin, "min", kPair, S::kInteger, false, Domain::kCommon,
     kAllContexts, true},
    {Builtin::kMax, "max", kPair, S::kInteger, false, Domain::kCommon,
     kAllContexts, true},
    {Builtin::kFloorDiv, "floor_div", kPair, S::kInteger, false,
     Domain::kCommon, kAllContexts, true},
    {Builtin::kCeilDiv, "ceil_div", kPair, S::kInteger, false, Domain::kCommon,
     kAllContexts, true},
};
}  // namespace

std::span<const BuiltinSignature> catalog() { return kBuiltins; }

std::string_view contextName(Context context) {
  switch (context) {
    case Context::kDType:
      return "dtype";
    case Context::kShape:
      return "shape";
    case Context::kWhere:
      return "where";
    case Context::kDerive:
      return "derive";
  }
  return "<invalid context>";
}

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

}  // namespace tepl::core::builtins
