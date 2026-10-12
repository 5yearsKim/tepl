#pragma once
#include "src/core/index_semantics/graph.h"
namespace tepl::core::index_semantics {
struct Term {
  S reductions;
  std::vector<E> reads;
};
using Terms = std::vector<Term>;
Terms normalize(Context&, const E&,
                std::vector<RuleProof::RewriteStep>* = nullptr);
std::string footprint(const Term&);
std::string show(const Terms&);
struct Correspondence {
  std::map<std::size_t, S> anchors;
  std::map<std::size_t, S> boundaries;
};
Correspondence correspond(Context&, const TensorMeaning&, const Terms&,
                          const Terms&);
}  // namespace tepl::core::index_semantics
