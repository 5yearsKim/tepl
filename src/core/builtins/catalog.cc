#include "src/core/builtins/catalog.h"

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
#define TEPL_BUILTIN(id, name, feature, args, result, variadic, domain, \
                     contexts, pair)                                    \
  {Builtin::id, name,           Feature::feature, args, S::result,      \
   variadic,    Domain::domain, contexts,         pair},
#include "src/core/builtins/catalog.def"
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
    if (builtin.name == name) return &builtin;
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
