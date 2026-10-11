#pragma once

#include "src/core/inference/semantic_ir.h"

namespace tepl::core::inference {
// Structural polynomial normalization. Binder names are irrelevant, factors
// commute, and unused one-point reductions are identities in algebraic mode.
// Unrecognized equivalences are left to matching and the exact verifier.
void normalizeConstraints(ConstraintPlan&);
}  // namespace tepl::core::inference
