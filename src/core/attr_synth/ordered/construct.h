#pragma once

#include "src/core/semantics/symbolic/normalize.h"
#include "src/inference/emission.h"

namespace tepl::core::attr_synth::ordered {
// Compilation chooses reusable metadata expressions, without proving that the
// written guard implies feasibility. Builders are defined when feasibility
// holds; the independent verifier discharges that obligation before use.
struct Construction {
  semantics::ConstructorCandidate candidate;
  std::map<std::size_t, semantics::S> layouts;
};
// Specialize the SAME metadata IR executed by the native interpreter. This
// adapter performs no retained-axis/order construction of its own. Bounded
// specialization may fail; it never changes or invalidates the residual IR.
Construction construct(semantics::Context&, const inference::EmissionBlock&,
                       const semantics::Correspondence&,
                       const semantics::S& source_output,
                       const semantics::S& reduction_order);
}  // namespace tepl::core::attr_synth::ordered
