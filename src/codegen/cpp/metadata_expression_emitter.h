#pragma once
#include "src/core/metadata/ir.h"
namespace tepl::codegen::cpp {
std::string metadataType(core::metadata::Type, const std::string& root);
std::string metadataSymbol(core::metadata::SymbolId);
std::string metadataExpression(const core::metadata::Program&,
                               const core::metadata::Expr&,
                               const std::string& root);
}  // namespace tepl::codegen::cpp
