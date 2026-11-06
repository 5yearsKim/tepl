#pragma once
#include <span>
#include <string_view>

#include "src/rewrite/value.h"
namespace tepl::rewrite {
// Called after AST/operator validation. Logical short-circuiting stays in the
// evaluator.
Value evaluateOperator(std::string_view, const std::vector<Value>&, Budget&);
Value evaluateIndex(const Value&, const Value&, Budget&);
}  // namespace tepl::rewrite
