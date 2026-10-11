#pragma once

#include <z3++.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "src/core/semantics/solver.h"

namespace tepl::core::semantics::detail {
// Engine supplies primitive metadata evaluation and Z3 vector operations.
// Coordinate labels denote identities. This is a sufficient proof condition,
// deliberately independent of operation names and the polynomial verifier.
template <class Engine>
struct ConstraintMatcher {
  Engine& e;
  using Vector = decltype(e.literal(List{}));
  using Environment = typename Engine::Environment;
  bool unused(const inference::ConstraintTerm& t, std::size_t binder) {
    return std::find(t.unused_reductions.begin(), t.unused_reductions.end(),
                     binder) != t.unused_reductions.end();
  }
  z3::expr terms(const inference::ConstraintTerm& lhs,
                 const inference::ConstraintTerm& rhs, std::size_t pair) {
    if (lhs.coefficient != rhs.coefficient ||
        lhs.reads.size() != rhs.reads.size())
      return e.ctx.bool_val(false);
    for (std::size_t i = 0; i < lhs.reads.size(); ++i)
      if (lhs.reads[i].capture != rhs.reads[i].capture)
        return e.ctx.bool_val(false);
    if (lhs.reads.size() > 128) e.constraintLimit("factor occurrence capacity");
    const auto node = e.plan.nodes.size();
    const auto output_rank = e.constants(e.shapes[e.plan.source_root]).size();
    List free(output_rank);
    std::iota(free.begin(), free.end(), 0);
    Environment left{{"@output", e.literal(free)}}, right = left;
    List extents;
    for (auto binder : lhs.reductions) {
      auto dims = e.constants(
          e.vector(e.eval(node, e.plan.constraints.domains.at(binder))));
      // The empty dimension vector has one point; any zero extent has none.
      if (unused(lhs, binder) &&
          std::all_of(dims.begin(), dims.end(), [](auto x) { return x == 1; }))
        continue;
      List labels(dims.size());
      std::iota(labels.begin(), labels.end(), output_rank + extents.size());
      left.emplace("@bound:" + std::to_string(binder), e.literal(labels));
      extents.insert(extents.end(), dims.begin(), dims.end());
    }
    auto left_dims = e.literal(extents);
    std::vector<Vector> right_dims;
    std::vector<z3::expr> drops, counts;
    auto count = e.integer(0);
    auto condition = e.ctx.bool_val(true);
    std::size_t capacity = 0;
    for (auto binder : rhs.reductions) {
      auto dims = e.vector(e.eval(node, e.plan.constraints.domains.at(binder)));
      auto drop = e.ctx.bool_val(false);
      if (unused(rhs, binder)) {
        drop = e.ctx.bool_const(
            ("drop_" + std::to_string(pair) + "_" + std::to_string(binder))
                .c_str());
        auto unit = e.ctx.bool_val(true);
        for (std::size_t j = 0; j < dims.entries.size(); ++j)
          unit = unit &&
                 z3::implies(e.integer(j) < dims.length, dims.entries[j] == 1);
        condition = condition && z3::implies(drop, unit);
      }
      auto active_count = z3::ite(drop, e.integer(0), dims.length);
      drops.push_back(drop);
      counts.push_back(active_count);
      count = count + active_count;
      capacity += dims.entries.size();
      right_dims.push_back(std::move(dims));
    }
    if (capacity > e.options.max_vector_capacity)
      e.constraintLimit("constraint reduction capacity");
    Vector permutation{count.simplify(), {}};
    for (std::size_t i = 0; i < capacity; ++i)
      permutation.entries.push_back(e.ctx.int_const(
          ("matching_" + std::to_string(pair) + "_" + std::to_string(i))
              .c_str()));
    condition = condition && e.valid(permutation, e.integer(extents.size())) &&
                permutation.length == e.integer(extents.size());
    auto offset = e.integer(0);
    for (std::size_t b = 0; b < rhs.reductions.size(); ++b) {
      auto& dims = right_dims[b];
      Vector labels{dims.length, {}};
      for (std::size_t j = 0; j < dims.entries.size(); ++j) {
        auto target = e.at(permutation, offset + e.integer(j), false);
        condition =
            condition &&
            z3::implies(!drops[b] && e.integer(j) < dims.length,
                        dims.entries[j] == e.at(left_dims, target, false));
        labels.entries.push_back(e.integer(output_rank) + target);
      }
      right.emplace("@bound:" + std::to_string(rhs.reductions[b]),
                    std::move(labels));
      offset = offset + counts[b];
    }
    for (std::size_t begin = 0; begin < lhs.reads.size();) {
      auto end = begin + 1;
      while (end < lhs.reads.size() &&
             lhs.reads[end].capture == lhs.reads[begin].capture)
        ++end;
      if (end == begin + 1) {
        auto a = e.vector(e.eval(node, lhs.reads[begin].indices, left));
        auto b = e.vector(e.eval(node, rhs.reads[begin].indices, right));
        condition = condition && e.equal(a, b);
      } else {
        // Bijection between occurrences, sharing the reduction renaming above.
        // This preserves multiplicities and does not enumerate permutations.
        std::vector<z3::expr> choices;
        for (auto i = begin; i < end; ++i) {
          e.tick();
          auto choice = e.ctx.int_const(
              ("factor_" + std::to_string(pair) + "_" + std::to_string(i))
                  .c_str());
          for (auto& previous : choices)
            condition = condition && choice != previous;
          choices.push_back(choice);
          auto row = e.ctx.bool_val(false);
          auto a = e.vector(e.eval(node, lhs.reads[i].indices, left));
          for (auto j = begin; j < end; ++j) {
            auto b = e.vector(e.eval(node, rhs.reads[j].indices, right));
            row = row || (choice == e.integer(j) && e.equal(a, b));
          }
          condition = condition && row;
        }
      }
      begin = end;
    }
    return condition.simplify();
  }
  void instantiate() {
    const auto& c = e.plan.constraints;
    if (c.lhs.size() != c.rhs.size()) {
      e.require(e.ctx.bool_val(false));
      return;
    }
    std::vector<z3::expr> choices;
    for (std::size_t i = 0; i < c.lhs.size(); ++i) {
      e.tick();
      auto choice = e.ctx.int_const(("term_" + std::to_string(i)).c_str());
      for (auto& previous : choices) e.require(choice != previous);
      choices.push_back(choice);
      auto row = e.ctx.bool_val(false);
      for (std::size_t j = 0; j < c.rhs.size(); ++j)
        row = row || (choice == e.integer(j) &&
                      terms(c.lhs[i], c.rhs[j], i * c.rhs.size() + j));
      e.require(row);
    }
  }
};
}  // namespace tepl::core::semantics::detail
