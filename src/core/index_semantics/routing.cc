#include "src/core/index_semantics/routing.h"

#include <functional>
#include <stdexcept>

namespace tepl::core::index_semantics {
namespace {
using Captures = std::set<std::size_t>;
Captures captures(const Plan& p, std::size_t id) {
  const auto& n = p.nodes[id];
  Captures result;
  if (n.capture) result.insert(n.capture->value);
  for (auto child : n.inputs) {
    auto xs = captures(p, child);
    result.insert(xs.begin(), xs.end());
  }
  return result;
}
bool subset(const Captures& a, const Captures& b) {
  for (auto x : a)
    if (!b.contains(x)) return false;
  return true;
}
std::set<std::size_t> reductions(const Value& v) {
  std::set<std::size_t> result;
  if (v.kind == Value::Kind::Reduce) result.insert(v.index);
  for (const auto& a : v.args) {
    auto xs = reductions(*a);
    result.insert(xs.begin(), xs.end());
  }
  return result;
}
bool product(const Value& v) {
  if (v.kind == Value::Kind::Product) return true;
  for (const auto& a : v.args)
    if (product(*a)) return true;
  return false;
}
bool aligned(const Definition& d) {
  if (d.groups.size() != 1 || d.output->kind != Meta::Kind::Group ||
      d.output->index != 0)
    return false;
  if (d.groups[0].bindings.size() != d.operands.size()) return false;
  for (const auto& b : d.groups[0].bindings)
    if (b.positions->kind != Meta::Kind::Call ||
        b.positions->name != "all_axes")
      return false;
  return true;
}
std::optional<std::size_t> producer(const Plan& p, std::size_t id) {
  const auto& n = p.nodes[id];
  if (!n.definition) return {};
  if (!reductions(*n.definition->value).empty() &&
      product(*n.definition->value))
    return id;
  // A coordinate-preserving pointwise combination does not change positions.
  if (aligned(*n.definition))
    for (auto child : n.inputs)
      if (auto found = producer(p, child)) return found;
  return {};
}
}  // namespace
std::vector<S> routingGuard(const Plan& p, std::size_t max_work) {
  if (!p.algebraic) return {};
  Context c{p, max_work};
  c.sourceScope(p.lhs);
  Lowerer evaluator{c};
  // Restrict shared, retained multiplicative groups to empty sequences. This
  // selects an unbatched fragment without constraining free/contraction ranks.
  for (auto id : c.source_nodes) {
    const auto& n = p.nodes[id];
    if (!n.definition || reductions(*n.definition->value).empty() ||
        !product(*n.definition->value))
      continue;
    auto bound = reductions(*n.definition->value);
    for (std::size_t g = 0; g < n.definition->groups.size(); ++g)
      if (!bound.contains(g) && n.definition->groups[g].bindings.size() > 1)
        for (const auto& b : n.definition->groups[g].bindings) {
          if (b.positions->kind != Meta::Kind::Attribute) return {};
          c.require(call("==", {c.field(n, b.positions->index),
                                make(Symbol::Kind::List)}));
        }
  }
  evaluator.lower(p.lhs);
  bool routed = false;
  for (auto id : c.source_nodes) {
    const auto& n = p.nodes[id];
    if (!n.definition) continue;
    auto bound = reductions(*n.definition->value);
    if (bound.empty() || !product(*n.definition->value)) continue;
    std::vector<TensorMeaning> inputs;
    for (std::size_t o = 0; o < n.inputs.size(); ++o)
      inputs.push_back({{}, c.input_shapes.at(id)[o], {}, {}});
    for (std::size_t o = 0; o < n.inputs.size(); ++o) {
      auto found = producer(p, n.inputs[o]);
      if (!found) continue;
      const auto& child = p.nodes[*found];
      auto child_caps = captures(p, *found);
      Captures outside;
      for (std::size_t k = 0; k < n.inputs.size(); ++k)
        if (k != o) {
          auto xs = captures(p, n.inputs[k]);
          outside.insert(xs.begin(), xs.end());
        }
      std::optional<Captures> target;
      std::function<void(std::size_t)> visit = [&](std::size_t r) {
        const auto& rhs = p.nodes[r];
        auto xs = captures(p, r);
        Captures intersection;
        for (auto x : child_caps)
          if (xs.contains(x)) intersection.insert(x);
        if (rhs.definition && subset(outside, xs) && !intersection.empty() &&
            intersection != child_caps &&
            (!target || xs.size() < target->size()))
          target = xs;
        for (auto input : rhs.inputs) visit(input);
      };
      visit(p.rhs);
      if (!target) continue;
      std::vector<S> consumed, retained;
      auto child_bound = reductions(*child.definition->value);
      for (std::size_t g = 0; g < child.definition->groups.size(); ++g) {
        if (child_bound.contains(g)) continue;
        auto axes = c.tensor_groups.at(*found)[g];
        if (callIs(axes, "empty_axes")) continue;
        Captures support;
        for (const auto& b : child.definition->groups[g].bindings) {
          auto xs = captures(p, child.inputs[b.operand]);
          support.insert(xs.begin(), xs.end());
        }
        auto positions = c.positions(c.tensor_axes.at(*found), axes);
        (subset(support, *target) ? consumed : retained).push_back(positions);
      }
      if (consumed.empty() || retained.empty()) continue;
      auto contraction = call("concat", consumed),
           free = call("concat", retained);
      bool did = false;
      for (std::size_t g = 0; g < n.definition->groups.size(); ++g)
        for (const auto& b : n.definition->groups[g].bindings)
          if (b.operand == o) {
            bool eliminate = bound.contains(g);
            if (!eliminate && n.definition->groups[g].bindings.size() != 1)
              continue;
            auto actual = evaluator.metadata(*b.positions, n, inputs,
                                             c.tensor_groups.at(id));
            c.require(call("==", {actual, eliminate ? contraction : free}));
            did = true;
          }
      routed |= did;
    }
  }
  if (!routed) return {};
  return c.guards;
}
}  // namespace tepl::core::index_semantics
