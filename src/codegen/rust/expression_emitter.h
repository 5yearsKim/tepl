#pragma once
#include <string>

#include "src/codegen/rust/names.h"
#include "src/core/ir.h"
namespace tepl::codegen::rust {
std::string emitExpression(const core::Program& program, const Names& names,
                           const core::TypedExpr& expression,
                           const std::string& builtins_path);
}
