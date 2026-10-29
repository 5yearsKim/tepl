#pragma once
#include "analysis.h"
namespace tepl_runtime::graphs {
template <Program P>
struct BuiltGraph {
  EGraph<P> graph;
  eggc::Id root;
  std::map<std::string, eggc::Id> bindings;
};
}  // namespace tepl_runtime::graphs
