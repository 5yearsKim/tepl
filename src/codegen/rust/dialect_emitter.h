#pragma once
#include <string>

#include "src/codegen/rust/names.h"
namespace tepl::codegen::rust {
std::string emitDialect(const core::Program&, const Names&,
                        const DialectNames&);
std::string emitOpUnion(const Names&);
}  // namespace tepl::codegen::rust
