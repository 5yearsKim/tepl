#pragma once

#include <string>

#include "src/core/shape/ir.h"

namespace tepl::codegen::rust {
std::string shapeType(core::shape::Type type);
std::string shapeSymbol(core::shape::SymbolId id);
std::string shapeExpression(const core::shape::Program& program,
                            const core::shape::Expr& expression);
}  // namespace tepl::codegen::rust
