#pragma once

#include "src/core/semantics/symbolic/normalize.h"

namespace tepl::core::attr_synth {
// Reuse unambiguous scalar-function parameters from the normalized source.
// These are proposals; target legality and equivalence remain obligations.
void inferScalarParameters(semantics::Context&, const semantics::Terms&);
}  // namespace tepl::core::attr_synth
