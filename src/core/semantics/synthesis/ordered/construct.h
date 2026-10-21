#pragma once

#include "src/core/semantics/synthesis/ordered/requirements.h"

namespace tepl::core::semantics::ordered {

// Instantiates requirements bottom-up. General verification still checks the
// candidate independently; this does not claim completeness of symbolic
// correspondence or bounded proof search.
TensorMeaning construct(Context&, Lowerer&, const Admission&,
                        const Requirements&, const S& reduction_order);

}  // namespace tepl::core::semantics::ordered
