#pragma once
#include "src/rewrite/builtins/dispatch.h"
namespace tepl::rewrite {
namespace detail {
Value listBuiltin(core::builtins::Builtin, const std::vector<Value>&, Budget&);
Value integerBuiltin(core::builtins::Builtin, const std::vector<Value>&,
                     Budget&);
Value shapeBuiltin(core::builtins::Builtin, const std::vector<Value>&, Budget&);
Value dtypeBuiltin(core::builtins::Builtin, const std::vector<Value>&, Budget&);
}  // namespace detail
}  // namespace tepl::rewrite
