#pragma once
#include <span>
#include <string_view>

#include "src/rewrite/value.h"

namespace tepl::rewrite::detail {
void ensure(bool, const char*);
Integer arithmetic(std::string_view, Integer, Integer);
bool contains(const List&, const Value&, Budget&);
bool distinct(const List&, Budget&);
bool valid(const List&, Integer, Budget&);
void chargeArguments(std::span<const Value>, Budget&);
}  // namespace tepl::rewrite::detail
