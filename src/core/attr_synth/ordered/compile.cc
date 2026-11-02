#include "src/core/attr_synth/ordered/compile.h"

namespace tepl::core::attr_synth::ordered {
Compilation compile(const Plan& plan, inference::Mode mode) {
  auto admission = admit(plan, mode);
  if (!admission.admitted()) return {{}, admission.reason};
  Compilation out;
  out.program = std::move(admission.program);
  if (mode == inference::Mode::Synthesize)
    out.emission = inference::emit(*out.program);
  return out;
}
}  // namespace tepl::core::attr_synth::ordered
