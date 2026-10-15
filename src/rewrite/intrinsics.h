#pragma once
#include <string_view>

#include "src/core/intrinsics/catalog.h"
#include "src/rewrite/value.h"
namespace tepl::rewrite {
Value evaluateIntrinsic(core::intrinsics::Intrinsic, const std::vector<Value>&,
                        Budget&);
// Explicit boundaries for checked axes source calls and textual symbolic IR.
Value evaluateAxisCall(std::string_view, const std::vector<Value>&, Budget&);
Value evaluateSymbolicCall(std::string_view, const std::vector<Value>&,
                           Budget&);
}  // namespace tepl::rewrite
