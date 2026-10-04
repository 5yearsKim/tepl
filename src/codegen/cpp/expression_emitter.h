#pragma once
#include "src/codegen/cpp/names.h"
namespace tepl::codegen::cpp {
std::string emitExpression(const core::Program&, const Names&,
                           const core::TypedExpr&);
}
