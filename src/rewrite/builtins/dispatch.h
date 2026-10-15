#pragma once
#include "src/core/builtins/catalog.h"
#include "src/rewrite/value.h"
namespace tepl::rewrite {
Value evaluateBuiltin(core::builtins::Builtin, const std::vector<Value>&,
                      Budget&);
}  // namespace tepl::rewrite
