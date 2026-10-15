#pragma once
#include "src/codegen/output.h"
#include "src/codegen/rust/names.h"
namespace tepl::codegen::rust {
void emitGraphs(const executable::ProgramData&, const Names&,
                const ProjectPlan&, GenerationResult&);
}
