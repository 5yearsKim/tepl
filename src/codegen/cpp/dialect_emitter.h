#pragma once
#include "src/codegen/cpp/names.h"
namespace tepl::codegen::cpp {
std::string emitDialect(const core::Program&, const Names&,
                        const DialectNames&);
std::string emitOpUnion(const core::Program&, const Names&);
std::string schemaType(const Names&, core::AttributeSchemaId);
}  // namespace tepl::codegen::cpp
