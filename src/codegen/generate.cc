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

std::unique_ptr<Generator> createGenerator(Target target) {
  switch (target) {
    case Target::kRust:
      return std::make_unique<rust::Backend>();
    case Target::kCpp:
    case Target::kPython:
      return nullptr;
  }
  return nullptr;
}

GenerationResult generate(const core::Program& program,
                          const Options& options) {
  auto backend = createGenerator(options.target);
  if (!backend) {
    GenerationResult result;
    result.diagnostics.push_back({{},
                                  std::string(targetName(options.target)) +
                                      " code generation is not implemented"});
    return result;
  }
  return backend->generate(program, options);
}

}  // namespace tepl::codegen
