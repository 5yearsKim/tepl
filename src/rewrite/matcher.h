#pragma once
#include "src/rewrite/evaluator.h"
namespace tepl::rewrite {
bool match(const core::Program&, const core::Rule&, const core::ConcreteGraph&,
           std::size_t, Bindings&, Budget&);
}
