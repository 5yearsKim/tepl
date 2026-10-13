#include "src/core/index_semantics/axis_sets.h"

#include <algorithm>
#include <limits>

namespace tepl::core::index_semantics {
AxisSets::Id AxisSets::node(std::size_t variable, Id low, Id high) {
  tick_();
  if (low == high) return low;
  auto key = std::tuple(variable, low, high);
  auto [it, added] = unique_.emplace(key, nodes_.size());
  if (added) nodes_.push_back({variable, low, high});
  return it->second;
}
AxisSets::Id AxisSets::atom(const S& s) {
  auto [it, added] = atoms_.emplace(key(s), 0);
  if (added) it->second = node(atoms_.size(), 0, 1);
  return it->second;
}
AxisSets::Id AxisSets::apply(bool is_or, Id a, Id b) {
  tick_();
  if (a > b) std::swap(a, b);
  if (a == b) return a;
  if (is_or && (a == 1 || b == 1)) return 1;
  if (is_or && a == 0) return b;
  if (!is_or && a == 0) return 0;
  if (!is_or && a == 1) return b;
  auto key = std::tuple(is_or, a, b);
  if (auto it = applications_.find(key); it != applications_.end())
    return it->second;
  auto av =
      a < 2 ? std::numeric_limits<std::size_t>::max() : nodes_[a].variable;
  auto bv =
      b < 2 ? std::numeric_limits<std::size_t>::max() : nodes_[b].variable;
  auto variable = std::min(av, bv);
  auto al = av == variable ? nodes_[a].low : a;
  auto ah = av == variable ? nodes_[a].high : a;
  auto bl = bv == variable ? nodes_[b].low : b;
  auto bh = bv == variable ? nodes_[b].high : b;
  auto low = apply(is_or, al, bl), high = apply(is_or, ah, bh);
  return applications_[key] = node(variable, low, high);
}
AxisSets::Id AxisSets::conjunction(Id a, Id b) { return apply(false, a, b); }
AxisSets::Id AxisSets::disjunction(Id a, Id b) { return apply(true, a, b); }
AxisSets::Id AxisSets::negate(Id a) {
  tick_();
  if (a < 2) return 1 - a;
  if (auto it = negations_.find(a); it != negations_.end()) return it->second;
  auto n = nodes_[a];
  auto low = negate(n.low), high = negate(n.high);
  return negations_[a] = node(n.variable, low, high);
}
AxisSets::Id AxisSets::equivalent(Id a, Id b) {
  return disjunction(conjunction(a, b), conjunction(negate(a), negate(b)));
}
void AxisSets::assume(Id f) { premises_ = conjunction(premises_, f); }
bool AxisSets::proves(Id f) {
  return premises_ != 0 && conjunction(premises_, negate(f)) == 0;
}
}  // namespace tepl::core::index_semantics
