#include "src/codegen/rust/builtin_emitter.h"

namespace tepl::codegen::rust {
std::string emitBuiltin(core::builtins::Builtin builtin,
                        const std::vector<std::string>& arguments,
                        const std::string& runtime, bool shape) {
  using B = core::builtins::Builtin;
  const auto* signature =
      core::builtins::resolveBuiltin(core::builtins::builtinName(builtin));
  const std::string fn =
      runtime +
      (signature->domain == core::builtins::Domain::kShape ? "::shape::"
                                                           : "::common::") +
      std::string(signature->name);
  const auto arg = [&](std::size_t i) { return arguments.at(i); };
  const auto ref = [&](std::size_t i) { return "&(" + arg(i) + ")"; };
  const std::string failure = shape ? "?" : ".ok()?";
  switch (builtin) {
    case B::kLen:
      return shape ? "(" + fn + "(" + ref(0) + ") as i128)"
                   : runtime + "::common::index_len(" + ref(0) + ")" + failure;
    case B::kRange:
      if (shape)
        return runtime + "::shape::integers(&" + fn + "(" + arg(0) + ")?)";
      return fn + "(" + arg(0) + ")" + failure;
    case B::kConcat: {
      std::string result = fn + "(&[";
      for (std::size_t i = 0; i < arguments.size(); ++i) {
        if (i) result += ", ";
        result += ref(i);
      }
      return result + "])" + failure;
    }
    case B::kGather:
    case B::kBroadcastShape:
      return fn + "(" + ref(0) + ", " + ref(1) + ")" + failure;
    case B::kExclude:
    case B::kIsDisjoint:
    case B::kContains:
      return fn + "(" + ref(0) + ", " + ref(1) + ")";
    case B::kSlice:
      return fn + "(" + ref(0) + ", " + arg(1) + ", " + arg(2) + ")" + failure;
    case B::kReplace:
      return fn + "(" + ref(0) + ", " + arg(1) + ", " + arg(2) + ")" + failure;
    case B::kSum:
    case B::kProduct:
      return fn + "(" + ref(0) + ")" + failure;
    case B::kAll:
    case B::kAny:
      return fn + "(" + ref(0) + ")";
    case B::kIsValidAxisList:
      return fn + "(" + ref(0) + ", " + arg(1) + ")";
    case B::kMin:
    case B::kMax:
      return fn + "(" + arg(0) + ", " + arg(1) + ")";
    case B::kFloorDiv:
    case B::kCeilDiv:
      return fn + "(" + arg(0) + ", " + arg(1) + ")" + failure;
  }
  return {};
}
}  // namespace tepl::codegen::rust
