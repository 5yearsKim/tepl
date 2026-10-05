#pragma once

#include <string>

#include "src/core/metadata/ir.h"

namespace tepl::codegen::rust {
std::string metadataType(core::metadata::Type type);
std::string metadataSymbol(core::metadata::SymbolId id);
std::string metadataExpression(const core::metadata::Program& program,
                               const core::metadata::Expr& expression);
}  // namespace tepl::codegen::rust
