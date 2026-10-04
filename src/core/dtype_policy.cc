#include "src/core/dtype_policy.h"

namespace tepl::core {

std::optional<DTypePolicy> resolveDTypePolicy(std::string_view name) {
  if (name == "same") return CommonDTypePolicy::kSame;
  if (name == "same_numeric") return CommonDTypePolicy::kSameNumeric;
  if (name == "same_float") return CommonDTypePolicy::kSameFloat;
  if (auto dtype = resolveDType(name)) return *dtype;
  return std::nullopt;
}

std::string_view dtypePolicyName(const DTypePolicy& policy) {
  if (const auto* dtype = std::get_if<DType>(&policy)) return dtypeName(*dtype);
  switch (std::get<CommonDTypePolicy>(policy)) {
    case CommonDTypePolicy::kSame:
      return "same";
    case CommonDTypePolicy::kSameNumeric:
      return "same_numeric";
    case CommonDTypePolicy::kSameFloat:
      return "same_float";
  }
  return "<invalid dtype policy>";
}

}  // namespace tepl::core
