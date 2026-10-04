#pragma once
#include <string>
#include <vector>

#include "src/core/builtins/catalog.h"
namespace tepl::codegen::cpp {
std::string emitBuiltin(core::builtins::Builtin,
                        const std::vector<std::string>&,
                        const std::string& runtime, bool shape);
}
