#pragma once
#include "src/core/semantics/context.h"
namespace tepl::core::semantics {
S substituteMetadata(const S&, const S& variable, const S& value);
S liftMetadataMap(Context&, const S& sequence, const S& variable,
                  const S& body);
S simplifyMetadataSequence(Context&, const S&);
}  // namespace tepl::core::semantics
