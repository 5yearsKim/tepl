#pragma once

#include <memory>

#include "src/codegen/options.h"
#include "src/codegen/output.h"
#include "src/core/ir.h"

namespace tepl::codegen {

// Every language backend implements this interface. Input is a successfully
// checked program, including the tables referenced by each rule's IDs.
class Generator {
 public:
  virtual ~Generator() = default;
  virtual Target target() const = 0;
  virtual GenerationResult generate(const core::Program& program,
                                    const Options& options) const = 0;
};

// Returns null for an unimplemented target; generate reports a diagnostic.
std::unique_ptr<Generator> createGenerator(Target target);
GenerationResult generate(const core::Program& program,
                          const Options& options = {});

}  // namespace tepl::codegen
