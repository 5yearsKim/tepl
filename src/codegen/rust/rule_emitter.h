#pragma once
#include "src/codegen/rust/names.h"
#include "src/executable/program.h"
namespace tepl::codegen::rust {
std::string emitRules(const executable::ExecutableProgram&, const Names&,
                      const RuleModuleNames&);
}
