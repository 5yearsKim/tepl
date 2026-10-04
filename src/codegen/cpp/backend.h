#pragma once
#include "src/codegen/generate.h"
namespace tepl::codegen::cpp {
class Backend final : public Generator {
 public:
  Target target() const override { return Target::kCpp; }
  GenerationResult generate(const core::Program&,
                            const Options&) const override;
};
}  // namespace tepl::codegen::cpp
