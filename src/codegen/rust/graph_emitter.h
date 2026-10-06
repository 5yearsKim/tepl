#pragma once
#include "src/codegen/output.h"
#include "src/codegen/rust/names.h"
namespace tepl::codegen::rust {
void emitGraphs(const core::Program&, const Names&, const ProjectPlan&,
                GenerationResult&);
}
