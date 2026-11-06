#pragma once
#include "src/codegen/cpp/names.h"
#include "src/codegen/output.h"
namespace tepl::codegen::cpp {
void emitGraphs(const executable::ProgramData&, const Names&,
                const ProjectPlan&, GenerationResult&);
}
