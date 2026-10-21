#pragma once

#include "src/rewrite/evaluator.h"

namespace tepl::rewrite {
// Run the completed program once; checker and constructor share its analysis.
bool inferAttributes(const executable::ProgramData&, const executable::Rule&,
                     const core::ConcreteGraph&, Bindings&, Budget&);
}  // namespace tepl::rewrite
