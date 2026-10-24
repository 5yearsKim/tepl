#pragma once

#include "src/core/attr_synth/ordered/contract.h"

namespace tepl::core::attr_synth::ordered {

struct Compilation {
  std::optional<inference::Program> program;
  std::string reason;
};
Compilation compile(const Plan&, inference::Mode = inference::Mode::Synthesize);
}  // namespace tepl::core::attr_synth::ordered
