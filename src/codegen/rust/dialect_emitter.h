#pragma once
#include "src/codegen/rust/names.h"
namespace tepl::codegen::rust {
std::string emitDialect(const executable::ProgramData&, const Names&,
                        const DialectNames&);
std::string emitProgram(const executable::ProgramData&, const Names&,
                        const std::string& policy);
}  // namespace tepl::codegen::rust
