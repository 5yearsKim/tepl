#pragma once

#include "src/core/semantics/synthesis/ordered/contract.h"

namespace tepl::core::semantics::ordered {
struct Compilation {
  std::optional<inference::Program> program;
  std::string reason;
};
Compilation compile(const Plan&, inference::Mode = inference::Mode::Synthesize);
}  // namespace tepl::core::semantics::ordered
