#pragma once
#include "src/rewrite/evaluator.h"
namespace tepl::rewrite {
std::size_t build(const executable::ProgramData&, const core::BuildExpr&,
                  core::ConcreteGraph&, const Bindings&, Budget&,
                  std::size_t max_nodes);
core::ConcreteGraph replace(const core::ConcreteGraph&, std::size_t target,
                            std::size_t replacement, std::size_t old_size);
bool sameGraph(const core::ConcreteGraph&, const core::ConcreteGraph&);
}  // namespace tepl::rewrite
