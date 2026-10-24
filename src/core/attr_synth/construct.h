#pragma once

#include "src/core/semantics/symbolic/normalize.h"

namespace tepl::core::attr_synth {

// Internal general strategy. Installs proposed fields in the attempt's context.
semantics::TensorMeaning construct(semantics::Context&, semantics::Lowerer&,
                                   const semantics::Correspondence&,
                                   const semantics::TensorMeaning&,
                                   const semantics::S& reductions,
                                   const semantics::Terms& source_terms);

}  // namespace tepl::core::attr_synth
