#pragma once
#include "src/codegen/rust/code_writer.h"
#include "src/codegen/rust/names.h"
#include "src/core/ir.h"
namespace tepl::codegen::rust {
void emitVerification(CodeWriter&, const core::Program&, const core::Rule&,
                      const Names&, const std::string& root);
}
