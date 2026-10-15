#pragma once

#include <memory>

#include "src/codegen/options.h"
#include "src/codegen/output.h"
#include "src/program/program.h"

namespace tepl::codegen {

// Every backend consumes the validated portable program, including resolved
// attribute computations. JSON is an optional transport, not an execution step.
class Generator {
 public:
  virtual ~Generator() = default;
  virtual Target target() const = 0;
  virtual GenerationResult generate(const program::Program& program,
                                    const Options& options) const = 0;
};

// Returns null for an unimplemented target; generate reports a diagnostic.
std::unique_ptr<Generator> createGenerator(Target target);
GenerationResult generate(const program::Program& program,
                          const Options& options = {});

// Convenience entry point for existing frontend callers.
GenerationResult generate(const core::Program& checked,
                          const Options& options = {});

}  // namespace tepl::codegen
