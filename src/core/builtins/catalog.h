#pragma once

#include <span>
#include <string_view>

namespace tepl::core::builtins {

enum class Builtin {
  kLen,
  kRange,
  kConcat,
  kGather,
  kExclude,
  kSlice,
  kReplace,
  kSum,
  kProduct,
  kAll,
  kAny,
  kContains,
  kIsValidAxisList,
  kIsDisjoint,
  kBroadcastShape,
  kMin,
  kMax,
  kFloorDiv,
  kCeilDiv,
};

// Availability is independent of a builtin's semantic domain.
enum class Context : unsigned { kShape = 1, kWhere = 2, kDerive = 4 };
enum class Domain { kCommon, kShape };
constexpr unsigned kAllContexts = 7;
std::string_view contextName(Context context);

// T is shared by every occurrence within one call, including nested lists.
enum class SignatureType {
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
  std::span<const SignatureType> arguments;
  SignatureType result;
  // Variadic concat repeats its final argument type and requires at least two.
  bool variadic = false;
  Domain domain = Domain::kCommon;
  unsigned contexts = kAllContexts;
  // Rules retain one shared Index/I64 type for integer pairs. Other integer
  // positions map to Index; shape expressions always use their Integer domain.
  bool rule_integer_pair = false;
  bool availableIn(Context context) const {
    return (contexts & static_cast<unsigned>(context)) != 0;
  }
};

std::span<const BuiltinSignature> catalog();
const BuiltinSignature* resolveBuiltin(std::string_view name);
std::string_view builtinName(Builtin builtin);

}  // namespace tepl::core::builtins
