#pragma once

#include "src/inference/emission.h"

namespace tepl::inference {
bool insertionCandidate(const Program&);
Emission emitInsertion(const Program&);
Result executeInsertion(const Emission&, const Metadata&, const Limits&,
                        bool build, bool sound_only, ExecutionTiming*);
}  // namespace tepl::inference
