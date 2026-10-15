#include "src/codegen/generate.h"

#include <stdexcept>

#include "src/codegen/cpp/backend.h"
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
      return std::make_unique<cpp::Backend>();
    case Target::kPython:
      return nullptr;
  }
  return nullptr;
}

GenerationResult generate(const program::Program& program,
                          const Options& options) {
  for (const auto& op : program.ir().operations)
    if (op.semantic_definition) {
      GenerationResult result;
      result.diagnostics.push_back(
          {op.origin, "code generation for axis semantics is not implemented"});
      return result;
    }
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

GenerationResult generate(const core::Program& checked,
                          const Options& options) {
  try {
    return generate(program::compile(checked), options);
  } catch (const std::exception& error) {
    GenerationResult result;
    result.diagnostics.push_back({{}, error.what()});
    return result;
  }
}
}  // namespace tepl::codegen
