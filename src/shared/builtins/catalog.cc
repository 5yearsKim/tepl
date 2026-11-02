#include "src/shared/builtins/catalog.h"

#include <array>
#include <stdexcept>

namespace tepl::core::builtins {
namespace {
using S = SignatureType;
constexpr unsigned kDTypeContexts = static_cast<unsigned>(Context::kDType) |
                                    static_cast<unsigned>(Context::kWhere) |
                                    static_cast<unsigned>(Context::kDerive);
constexpr S kDType[] = {S::kDType};
constexpr S kList[] = {S::kListT};
constexpr S kInteger[] = {S::kInteger};
constexpr S kAnyInteger[] = {S::kAnyInteger};
constexpr S kLists[] = {S::kListT, S::kListT};
constexpr S kGather[] = {S::kListT, S::kIntegerList};
constexpr S kSlice[] = {S::kListT, S::kInteger, S::kInteger};
constexpr S kReplace[] = {S::kListT, S::kInteger, S::kT};
constexpr S kIntegers[] = {S::kIntegerList};
constexpr S kBooleans[] = {S::kBooleanList};
constexpr S kContains[] = {S::kListT, S::kT};
constexpr S kAxes[] = {S::kIntegerList, S::kInteger};
constexpr S kShapes[] = {S::kIntegerList, S::kIntegerList};
constexpr std::array<S, 0> kNoArguments{};
constexpr S kAxisFilter[] = {S::kAxisList, S::kAxisSet};
constexpr S kAxisSets[] = {S::kAxisSet, S::kAxisSet};
constexpr S kNodeGroup[] = {S::kInteger, S::kInteger};
constexpr S kAxisLists[] = {S::kAxisList, S::kAxisList};
constexpr S kPair[] = {S::kInteger, S::kInteger};
constexpr BuiltinSignature kBuiltins[] = {
#define TEPL_BUILTIN(id, name, feature, args, result, variadic, domain, \
                     contexts, pair)                                    \
  {Builtin::id, name,           Feature::feature, args, S::result,      \
   variadic,    Domain::domain, contexts,         pair},
#include "src/shared/builtins/catalog.def"
#undef TEPL_BUILTIN
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
    if (builtin.name == name && isPublicBuiltin(builtin.builtin))
      return &builtin;
  return nullptr;
}

const BuiltinSignature& signature(Builtin builtin) {
  for (const auto& entry : kBuiltins)
    if (entry.builtin == builtin) return entry;
  throw std::logic_error("invalid builtin ID");
}

std::string_view builtinName(Builtin builtin) {
  return signature(builtin).name;
}

bool isPublicBuiltin(Builtin b) { return !isAxisQuery(b); }
bool isAxisQuery(Builtin b) {
  switch (b) {
    case Builtin::kAxisCaptureOrder:
    case Builtin::kAxisSourceNodeOrder:
    case Builtin::kAxisSourceGroupOrder:
    case Builtin::kAxisSourceGroupMembers:
    case Builtin::kAxisContract:
    case Builtin::kAxisLive:
    case Builtin::kAxisFeasible:
    case Builtin::kAxisDomain:
    case Builtin::kAxisRequired:
    case Builtin::kAxisBatch:
    case Builtin::kAxisLeftFree:
    case Builtin::kAxisRightFree:
    case Builtin::kAxisInputOrder:
    case Builtin::kAxisSourcePartition:
    case Builtin::kAxisSourceOrder:
      return true;
    default:
      return isReplacementAxisQuery(b);
  }
}
bool isReplacementAxisQuery(Builtin b) {
  return b == Builtin::kAxisRhsNodeOrder || b == Builtin::kAxisRhsGroupOrder ||
         b == Builtin::kAxisRhsGroupMembers ||
         b == Builtin::kAxisRhsGroupPositions || b == Builtin::kAxisRhsLegal ||
         b == Builtin::kAxisRhsPartition || b == Builtin::kAxisRhsOrder;
}
std::string_view featureName(Feature feature) {
  switch (feature) {
    case Feature::kList:
      return "list";
    case Feature::kInteger:
      return "integer";
    case Feature::kShape:
      return "shape";
    case Feature::kDType:
      return "dtype";
  }
  throw std::logic_error("invalid builtin feature");
}

}  // namespace tepl::core::builtins
