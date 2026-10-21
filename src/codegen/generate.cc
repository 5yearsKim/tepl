#include "src/codegen/generate.h"

#include "src/codegen/rust/backend.h"
namespace tepl::codegen {
std::string_view targetName(Target target) {
  switch (target) {
    case Target::kRust:
      return "rust";
    case Target::kCpp:
      return "cpp";
    case Target::kPython:
      return "python";
  }
  return "unknown";
}
GenerationResult generate(const executable::ExecutableProgram& program,
                          const Options& options) {
  for (const auto& r : program.ir().rules)
    if (r.inference)
      return {{},
              {{r.origin,
                "inferred applicability programs currently support native C++ "
                "execution; code generation is deferred"}}};
  if (options.target != Target::kRust)
    return {{},
            {{{},
              std::string(targetName(options.target)) +
                  " code generation is not implemented"}}};
  return rust::generate(program, options);
}
}  // namespace tepl::codegen
