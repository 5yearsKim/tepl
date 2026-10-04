#pragma once
#include <string>
#include <vector>

#include "src/core/builtins/catalog.h"

namespace tepl::codegen::rust {
// Arguments are emitted expressions, used once in Rust evaluation order.
// Shape uses Result propagation and i128; rules use Option and u64/i64.
std::string emitBuiltin(core::builtins::Builtin builtin,
                        const std::vector<std::string>& arguments,
                        const std::string& runtime, bool shape);
}  // namespace tepl::codegen::rust
