#include "src/core/index_semantics/synthesize.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "src/core/index_semantics/verify.h"

namespace tepl::core::index_semantics {
namespace {
struct Constructor {
  Context& c;
  Lowerer& lowerer;
  const Correspondence& match;
  const TensorMeaning& source;
  S reductions;
  void assign(const Meta& m, const Instance& n,
              const std::vector<TensorMeaning>& inputs,
              const std::vector<S>& groups, const S& value) {
    auto old = lowerer.metadata(m, n, inputs, groups);
    old = c.resolve(old);
    if (old->kind == Symbol::Kind::Hole) {
      c.assignAttribute(old->owner, old->index, value);
    } else if (containsHole(old))
      throw ProofFailure(
          "inverse metadata expression is outside constructive fragment");
    else
      c.require(call("==", {old, value}));
  }
  TensorMeaning build(std::size_t id) {
    c.tick();
    const auto& n = c.plan.nodes[id];
    if (n.capture) return lowerer.lower(id);
    std::vector<TensorMeaning> inputs;
    for (auto child : n.inputs) inputs.push_back(build(child));
    if (!n.automatic) return lowerer.operation(id, std::move(inputs));
    std::set<std::size_t> reduced;
    std::function<void(const Value&)> scan = [&](const Value& v) {
      if (v.kind == Value::Kind::Reduce) reduced.insert(v.index);
      for (const auto& a : v.args) scan(*a);
    };
    scan(*n.definition->value);
    std::set<std::pair<std::set<std::size_t>, bool>> signatures;
    std::vector<S> groups;
    for (std::size_t g = 0; g < n.definition->groups.size(); ++g) {
      const auto& group = n.definition->groups[g];
      std::set<std::size_t> incidence;
      for (const auto& b : group.bindings) incidence.insert(b.operand);
      if (!signatures.emplace(incidence, reduced.contains(g)).second)
        throw ProofFailure(
            "multiple binding groups share an unresolved incidence class");
      if (group.domain) {
        if (!incidence.empty())
          throw ProofFailure("explicit domain groups cannot bind operands");
        std::vector<S> operand_axes;
        for (const auto& input : inputs) operand_axes.push_back(input.axes);
        auto wanted = c.difference(match.boundaries.at(id),
                                   call("union_axes", std::move(operand_axes)));
        auto order = c.filter(match.interfaces.at(id), wanted);
        assign(*group.domain, n, inputs, groups, c.extents(order));
        groups.push_back(order);
        continue;
      }
      S wanted;
      for (auto o : incidence)
        wanted = wanted ? c.intersect(wanted, inputs[o].interface.axes)
                        : c.support(inputs[o].interface.axes);
      for (std::size_t o = 0; o < inputs.size(); ++o)
        if (!incidence.contains(o))
          wanted = c.difference(wanted, inputs[o].interface.axes);
      wanted = reduced.contains(g)
                   ? c.difference(wanted, match.boundaries.at(id))
                   : c.intersect(wanted, match.boundaries.at(id));
      S order;
      // A fixed binding determines group order. Unknown paired fields follow
      // it.
      for (const auto& b : group.bindings) {
        auto p = lowerer.metadata(*b.positions, n, inputs, groups);
        if (!containsHole(c.resolve(p))) {
          c.require(call("is_valid_axis_list",
                         {p, call("len", {inputs[b.operand].interface.axes})}));
          order = c.select(inputs[b.operand].interface.axes, p);
          c.selectionFacts(inputs[b.operand].interface.axes, p, order);
          break;
        }
      }
      if (order)
        c.require(call("same_set", {order, wanted}));
      else {
        auto preferred = reduced.contains(g) ? reductions : source.axes;
        order =
            call("concat", {c.filter(preferred, wanted),
                            c.filter(inputs[*incidence.begin()].interface.axes,
                                     c.difference(wanted, preferred))});
      }
      for (const auto& b : group.bindings) {
        // Fixed selectors are checked by semantic lowering in their original
        // form. Only unresolved selectors need inversion into attributes.
        auto fixed =
            c.resolve(lowerer.metadata(*b.positions, n, inputs, groups));
        if (!containsHole(fixed)) continue;
        auto positions = inputs[b.operand].interface.lookup(order);
        if (!positions)
          positions =
              c.mappedPositions(inputs[b.operand].interface.axes, order);
        if (!positions)
          positions = c.positions(inputs[b.operand].interface.axes, order);
        assign(*b.positions, n, inputs, groups, positions);
      }
      groups.push_back(order);
    }
    // Generic inversion of output permutations. Preserve the source order at
    // the root and at internal reindexings with the same complete interface.
    std::function<void(const Meta&)> output = [&](const Meta& m) {
      if (m.kind != Meta::Kind::Call) return;
      if (m.name == "place") {
        auto left = lowerer.metadata(*m.args[0], n, inputs, groups);
        auto right = lowerer.metadata(*m.args[2], n, inputs, groups);
        auto desired = match.interfaces.at(id);
        c.require(call("same_set", {call("concat", {left, right}), desired}));
        auto p = lowerer.metadata(*m.args[1], n, inputs, groups);
        if (containsHole(p))
          assign(*m.args[1], n, inputs, groups, c.positions(desired, left));
      }
      if (m.name == "permute") {
        auto p = lowerer.metadata(*m.args[1], n, inputs, groups);
        if (containsHole(p)) {
          auto base = lowerer.metadata(*m.args[0], n, inputs, groups);
          auto wanted = id == c.plan.rhs ||
                                c.entails(call("same_set", {base, source.axes}))
                            ? source.axes
                            : base;
          c.require(call("same_set", {base, wanted}));
          assign(*m.args[1], n, inputs, groups, c.positions(base, wanted));
        }
      }
      for (const auto& a : m.args) output(*a);
    };
    output(*n.definition->output);
    // Target lowering checks every obligation before publishing this node's
    // maps. Rechecking after publication repeats proof work in a changed frame.
    return lowerer.operation(id, std::move(inputs));
  }
};
}  // namespace
namespace {
RuleProof run(const Plan& plan, std::size_t max_work) {
  Context c{plan, max_work};
  RuleProof candidate;
  std::string phase = "source lowering";
  try {
    c.sourceScope(plan.lhs);
    Lowerer source{c};
    // A second composition uses legality learned across the complete source,
    // including shapes shared through pointwise consumers.
    source.lower(plan.lhs);
    Lowerer composed{c};
    auto lhs = composed.lower(plan.lhs);
    c.enableSetReasoning();
    auto lt = normalize(c, lhs.value, &candidate.rewrites);
    phase = "coordinate correspondence";
    auto rt = normalize(c, arithmeticSkeleton(plan, plan.rhs));
    auto match = correspond(c, lhs, lt, rt);
    phase = "attribute construction";
    std::vector<S> all;
    for (const auto& t : lt) all.push_back(t.reductions);
    Lowerer target{c, &match.anchors};
    target.interfaces = &match.interfaces;
    Constructor ctor{c, target, match, lhs,
                     call("unique_axes", {call("concat", std::move(all))})};
    auto rhs = ctor.build(plan.rhs);
    auto checked = normalize(c, rhs.value, &candidate.rewrites);
    auto simplified_lhs = collectTerms(c, lt, &candidate.rewrites);
    auto simplified_rhs = collectTerms(c, checked, &candidate.rewrites);
    equivalentTerms(c, simplified_lhs, simplified_rhs);
    c.require(call("==", {lhs.axes, rhs.axes}), GuardKind::CoordinateOrder);
    c.require(call("==", {lhs.shape, rhs.shape}));
    c.require(call("==", {lhs.dtype, rhs.dtype}));
    for (const auto& n : plan.nodes)
      if (n.automatic) {
        auto& fields = candidate.constructors[n.descriptor->value];
        for (std::size_t f = 0; f < n.definition->fields.size(); ++f) {
          auto value = c.canonical(c.field(n, f));
          c.validateSource(value);
          if (value->sort != Symbol::Sort::Positions &&
              !(n.definition->extent_fields.contains(f) &&
                value->sort == Symbol::Sort::Extents))
            throw ProofFailure(
                "attribute constructor is not a position sequence");
          fields.push_back(value);
        }
      }
    candidate.guard = c.guards();
    candidate.guard_origins = c.guardOrigins();
    auto proof = verify(plan, candidate, max_work);
    proof.rewrites = std::move(candidate.rewrites);
    return proof;
  } catch (const ProofFailure& e) {
    RuleProof unknown;
    unknown.reason = phase + ": " + e.describe();
    unknown.steps = c.steps();
    return unknown;
  }
}
}  // namespace
RuleProof synthesize(const Plan& plan, std::size_t max_work) {
  return run(plan, max_work);
}
}  // namespace tepl::core::index_semantics
