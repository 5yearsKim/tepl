#pragma once
#include "src/codegen/cpp/names.h"
namespace tepl::codegen::cpp {
std::string schemaType(const Names&, core::AttributeSchemaId);
std::string emitDialect(const executable::ProgramData&, const Names&,
                        const DialectNames&);
std::string emitProgram(const executable::ProgramData&, const Names&,
                        const std::string& policy);
}  // namespace tepl::codegen::cpp
