#pragma once
#include "src/core/semantics/graph.h"
namespace tepl::core::semantics {
struct Term {
  S reductions;
  std::vector<E> reads;
  std::int64_t coefficient = 1;
  // Polynomial factors may be reads or opaque scalar calls. reads retains all
  // nested tensor accesses for coordinate correspondence and attribute
  // inference.
  std::vector<E> factors;
};
using Terms = std::vector<Term>;
Terms normalize(Context&, const E&,
                std::vector<RuleProof::RewriteStep>* = nullptr);
// Coordinate correspondence uses uncollected terms; collection runs only in a
// fully lowered coordinate frame, never on the RHS arithmetic skeleton.
Terms collectTerms(Context&, const Terms&,
                   std::vector<RuleProof::RewriteStep>* = nullptr);
void equivalentTerms(Context&, const Terms&, const Terms&);
std::string footprint(const Term&);
std::string show(const Terms&);
struct Correspondence {
  std::map<std::size_t, S> anchors;
  std::map<std::size_t, S> boundaries;
  // Value support boundaries before general interface propagation.
  std::map<std::size_t, S> mandatory_boundaries;
  std::map<std::size_t, S> interfaces;
};
Correspondence correspond(Context&, const TensorMeaning&, const Terms&,
                          const Terms&);
}  // namespace tepl::core::semantics
