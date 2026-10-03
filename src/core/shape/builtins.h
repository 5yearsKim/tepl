#pragma once

#include <span>
#include <string_view>

namespace tepl::core::shape {

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
};

const BuiltinSignature* resolveBuiltin(std::string_view name);
std::string_view builtinName(Builtin builtin);

}  // namespace tepl::core::shape
