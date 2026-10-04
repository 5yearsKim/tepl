#pragma once
#include <string>

#include "src/core/ir.h"
namespace tepl::codegen::rust {
std::string emitExpression(const core::Program& program,
                           const core::TypedExpr& expression,
                           const std::string& builtins_path);
}
