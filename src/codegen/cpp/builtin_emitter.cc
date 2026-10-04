#include "src/codegen/cpp/builtin_emitter.h"

namespace tepl::codegen::cpp {
std::string emitBuiltin(core::builtins::Builtin builtin,
                        const std::vector<std::string>& args,
                        const std::string& runtime, bool shape) {
  using B = core::builtins::Builtin;
  auto signature =
      core::builtins::resolveBuiltin(core::builtins::builtinName(builtin));
  std::string out = "([&]() { ";
  for (std::size_t i = 0; i < args.size(); ++i)
    out += "auto arg_" + std::to_string(i) + " = " + args[i] + "; ";
  std::string call =
      runtime +
      (signature->domain == core::builtins::Domain::kShape ? "::shape::"
                                                           : "::common::") +
      std::string(signature->name);
  if (builtin == B::kRange && shape) call += "<" + runtime + "::Integer>";
  call += "(";
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (i) call += ",";
    call += "arg_" + std::to_string(i);
  }
  call += ")";
  switch (builtin) {
    case B::kLen:
      if (shape) call = "static_cast<" + runtime + "::Integer>(" + call + ")";
      break;
    case B::kContains:
    case B::kExclude:
    case B::kIsDisjoint:
    case B::kIsValidAxisList:
    case B::kAll:
    case B::kAny:
    case B::kMin:
    case B::kMax:
      break;
    default:
      call = runtime + "::take(" + call + ")";
      break;
  }
  return out + "return " + call + "; }())";
}
}  // namespace tepl::codegen::cpp
