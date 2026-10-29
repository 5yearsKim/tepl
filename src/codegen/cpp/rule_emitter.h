#pragma once
#include "src/codegen/cpp/names.h"
#include "src/executable/program.h"
namespace tepl::codegen::cpp {
std::string emitRules(const executable::ExecutableProgram&, const Names&,
                      const RuleModuleNames&);
}
