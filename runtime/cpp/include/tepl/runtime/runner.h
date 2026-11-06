#pragma once
#include <eggc/runner.hpp>

#include "rewriting.h"
namespace tepl_runtime::runner {
template <Program P>
EvalResult<eggc::RunReport> run(EGraph<P>& graph,
                                const std::vector<Rewrite<P>>& rules,
                                const eggc::RunOptions& options = {}) {
  return evaluate([&] {
    graph.analysis.check().value();
    auto result = eggc::run(static_cast<BaseGraph<P>&>(graph), rules, options);
    graph.analysis.check().value();
    return result;
  });
}
}  // namespace tepl_runtime::runner
