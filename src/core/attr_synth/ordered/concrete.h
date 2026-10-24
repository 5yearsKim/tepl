#pragma once

#include "src/core/attr_synth/ordered/compile.h"
#include "src/inference/evaluate.h"

namespace tepl::core::attr_synth::ordered {

using inference::Attributes;
using inference::AxisId;
using inference::AxisSequence;
using inference::AxisSet;
using inference::InputPort;
using inference::Metadata;
using inference::SourceNetwork;
using ConcreteRequirements = inference::Requirements;
using inference::Obstruction;
using inference::Status;
using ConcreteResult = inference::Result;

// Convenience APIs for callers holding a compiler Plan. Runtime callers compile
// once and execute the completed inference::Program directly.
ConcreteResult solve(
    const Plan&, const Metadata&,
    std::size_t max_work = std::numeric_limits<std::size_t>::max());
bool check(const Plan&, const Metadata&, const ConcreteResult&);
}  // namespace tepl::core::attr_synth::ordered
