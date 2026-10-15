#pragma once

#include <span>
#include <string_view>

namespace tepl::core::builtins {

// Only these identities can be resolved in rule and metadata expressions.
enum class Builtin {
#define TEPL_BUILTIN(id, ...) id,
#include "src/core/builtins/catalog.def"
#undef TEPL_BUILTIN
};

// Documentation/implementation grouping is independent of runtime namespaces.
enum class Feature { kList, kInteger, kShape, kDType };
std::string_view featureName(Feature feature);

// Section permissions are independent of feature groups and runtime domains.
enum class Context : unsigned {
  kShape = 1,
  kWhere = 2,
  kDerive = 4,
  kDType = 8
};
enum class Domain { kCommon, kShape, kDType };
constexpr unsigned kAllContexts = 15;
std::string_view contextName(Context context);

// T is shared by every occurrence within one call, including nested lists.
enum class SignatureType {
  kDType,
  kInteger,
  kBoolean,
  kIntegerList,
  kBooleanList,
  kT,
  kListT
};
struct BuiltinSignature {
  Builtin builtin;
  std::string_view name;
  Feature feature;
  std::span<const SignatureType> arguments;
  SignatureType result;
  // Variadic concat repeats its final argument type and requires at least two.
  bool variadic = false;
  Domain domain = Domain::kCommon;
  unsigned contexts = kAllContexts;
  // Rules retain one shared Index/I64 type for integer pairs. Other integer
  // positions map to Index; shape expressions always use their Integer domain.
  bool rule_integer_pair = false;
  bool acceptsArity(std::size_t count) const {
    return variadic ? count >= arguments.size() : count == arguments.size();
  }
  bool availableIn(Context context) const {
    return (contexts & static_cast<unsigned>(context)) != 0;
  }
};

std::span<const BuiltinSignature> catalog();
const BuiltinSignature& signature(Builtin builtin);
const BuiltinSignature* resolveBuiltin(std::string_view name);
std::string_view builtinName(Builtin builtin);

}  // namespace tepl::core::builtins
