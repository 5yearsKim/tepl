#pragma once
#include "src/codegen/cpp/names.h"
#include "src/codegen/output.h"
namespace tepl::codegen::cpp {
void emitGraphs(const core::Program&, const Names&, const ProjectPlan&,
                GenerationResult&);
}
