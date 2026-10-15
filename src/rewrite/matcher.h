#pragma once
#include "src/rewrite/evaluator.h"
namespace tepl::rewrite {
bool match(const executable::ProgramData&, const executable::Rule&,
           const core::ConcreteGraph&, std::size_t, Bindings&, Budget&);
}
