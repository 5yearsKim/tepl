#pragma once
#include "src/rewrite/value.h"
#include "src/shared/builtins/catalog.h"
namespace tepl::rewrite {
Value evaluateBuiltin(core::builtins::Builtin, const std::vector<Value>&,
                      Budget&);
}  // namespace tepl::rewrite
