#pragma once

#include "src/codegen/generate.h"

namespace tepl::codegen::rust {
class Backend final : public Generator {
 public:
  Target target() const override { return Target::kRust; }
  GenerationResult generate(const program::Program& program,
                            const Options& options) const override;
};
}  // namespace tepl::codegen::rust
