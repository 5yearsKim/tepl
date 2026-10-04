#pragma once
#include "src/core/shape/ir.h"
namespace tepl::codegen::cpp {
std::string shapeType(core::shape::Type, const std::string& root);
std::string shapeSymbol(core::shape::SymbolId);
std::string shapeExpression(const core::shape::Program&,
                            const core::shape::Expr&, const std::string& root);
}  // namespace tepl::codegen::cpp
