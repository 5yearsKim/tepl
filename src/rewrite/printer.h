#pragma once
#include <string>

#include "src/core/ir.h"
namespace tepl::rewrite {
// Uses named let bindings so sharing survives parsing. Imports refer to the
// dialect definition files; their paths are made absolute for output
// relocation.
std::string printGraph(const core::Program&, const core::ConcreteGraph&,
                       bool imports = true);
}  // namespace tepl::rewrite
