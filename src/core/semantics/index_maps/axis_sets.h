#pragma once

#include <functional>
#include <map>
#include <tuple>
#include <vector>

#include "src/core/semantics/symbolic/expr.h"

namespace tepl::core::semantics {
// A reduced decision DAG for per-coordinate membership, not enumeration of
// axis ranks or partitions. Set identities are propositional identities.
class AxisSets {
 public:
  using Id = std::size_t;
  explicit AxisSets(std::function<void()> tick) : tick_(std::move(tick)) {}
  Id atom(const S&);
  Id conjunction(Id, Id);
  Id disjunction(Id, Id);
  Id negate(Id);
  Id equivalent(Id, Id);
  void assume(Id);
  bool proves(Id);
  bool consistent() const { return premises_ != 0; }

 private:
  struct Node {
    std::size_t variable;
    Id low, high;
  };
  symbolic::KeyArena keys_;
  std::function<void()> tick_;
  std::vector<Node> nodes_{{0, 0, 0}, {0, 1, 1}};
  std::map<std::tuple<std::size_t, Id, Id>, Id> unique_;
  std::map<std::tuple<bool, Id, Id>, Id> applications_;
  std::map<Id, Id> negations_;
  std::map<Key, Id> atoms_;
  Id premises_ = 1;
  Id node(std::size_t, Id, Id);
  Id apply(bool, Id, Id);
};
}  // namespace tepl::core::semantics
