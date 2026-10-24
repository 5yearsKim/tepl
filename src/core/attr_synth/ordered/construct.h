#pragma once

#include "src/core/attr_synth/ordered/requirements.h"

namespace tepl::core::attr_synth::ordered {
using semantics::Lowerer;
using semantics::TensorMeaning;


// Instantiates requirements bottom-up. General verification still checks the
// candidate independently; this does not claim completeness of symbolic
// correspondence or bounded proof search.
TensorMeaning construct(Context&, Lowerer&, const Admission&,
                        const Requirements&, const S& reduction_order);

}  // namespace tepl::core::attr_synth::ordered
