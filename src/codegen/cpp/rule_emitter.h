#pragma once
#include "src/codegen/cpp/names.h"
namespace tepl::codegen::cpp {
std::string emitRules(const core::Program&, const Names&,
                      const std::vector<const core::Rule*>&,
                      const RuleModuleNames&);
}
