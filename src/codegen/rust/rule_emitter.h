#pragma once
#include <string>

#include "src/codegen/rust/names.h"
#include "src/core/ir.h"
namespace tepl::codegen::rust {
std::string emitRules(const core::Program& program, const Names& names,
                      const std::vector<const core::Rule*>& rules,
                      const std::string& module);
}
