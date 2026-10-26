#pragma once
#include "src/core/semantics/context.h"
namespace tepl::core::semantics {
// Project literal positions into context-sized concatenation blocks. Adjacent
// visits to one block share a selector; distinct occurrences stay distinct.
std::optional<std::vector<SequenceSelection>> projectSelection(
    Context&, const S& base, const S& positions);
S substituteMetadata(const S&, const S& variable, const S& value);
S liftMetadataMap(Context&, const S& sequence, const S& variable,
                  const S& body);
S simplifyMetadataSequence(Context&, const S&);
}  // namespace tepl::core::semantics
