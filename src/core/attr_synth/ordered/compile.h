#pragma once

#include "src/core/attr_synth/ordered/contract.h"
#include "src/inference/emission.h"

namespace tepl::core::attr_synth::ordered {

struct Compilation {
  std::optional<inference::Program> program;
  std::string reason;
  inference::DomainRequirements domain_requirements;
  std::optional<inference::Emission> emission;
};
Compilation compile(const Plan&, inference::Mode = inference::Mode::Synthesize);
}  // namespace tepl::core::attr_synth::ordered
