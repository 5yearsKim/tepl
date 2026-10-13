#include "src/core/index_semantics/synthesize.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "src/core/index_semantics/routing.h"
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
      S wanted;
      for (auto o : incidence)
        wanted = wanted ? c.intersect(wanted, inputs[o].axes)
                        : c.support(inputs[o].axes);
      for (std::size_t o = 0; o < inputs.size(); ++o)
        if (!incidence.contains(o))
          wanted = c.difference(wanted, inputs[o].axes);
      wanted = reduced.contains(g)
                   ? c.difference(wanted, match.boundaries.at(id))
                   : c.intersect(wanted, match.boundaries.at(id));
      S order;
      // A fixed binding determines group order. Unknown paired fields follow
      // it.
      for (const auto& b : group.bindings) {
        auto p = lowerer.metadata(*b.positions, n, inputs, groups);
        if (!containsHole(c.resolve(p))) {
          order = c.select(inputs[b.operand].axes, p);
          break;
        }
      }
      if (order)
        c.require(call("same_set", {order, wanted}));
      else {
        auto preferred = reduced.contains(g) ? reductions : source.axes;
        order = call("concat", {c.filter(preferred, wanted),
                                c.filter(inputs[*incidence.begin()].axes,
                                         c.difference(wanted, preferred))});
      }
      for (const auto& b : group.bindings)
        assign(*b.positions, n, inputs, groups,
               c.positions(inputs[b.operand].axes, order));
      groups.push_back(order);
    }
    // Generic inversion of output permutations. Preserve the source order at
    // the root and at internal reindexings with the same complete interface.
    std::function<void(const Meta&)> output = [&](const Meta& m) {
      if (m.kind != Meta::Kind::Call) return;
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
    auto begin = lowerer.obligations.size();
    auto tensor = lowerer.operation(id, std::move(inputs));
    for (std::size_t i = begin; i < lowerer.obligations.size(); ++i) {
      c.require(lowerer.obligations[i]);
    }
    return tensor;
  }
};
}  // namespace
namespace {
enum class InferenceStrategy { General, WholeGroupRouting };
RuleProof run(const Plan& plan, std::size_t max_work,
              InferenceStrategy strategy, const std::vector<S>& seeds) {
  // General coordinate inference is an initial bounded strategy. If its
  // unresolved maps consume this allowance, try the existing constructive
  // routing fragment; independent verification still gets the caller's budget.
  const auto inference_work =
      strategy == InferenceStrategy::General
          ? std::min(max_work, ProofLimits::general_inference_work)
          : max_work;
  Context c{plan, inference_work};
  RuleProof candidate;
  std::string phase = "source lowering";
  try {
    c.sourceScope(plan.lhs);
    for (const auto& seed : seeds) c.require(seed);
    Lowerer source{c};
    auto lhs = source.lower(plan.lhs);
    auto lt = normalize(c, lhs.value, &candidate.rewrites);
    phase = "coordinate correspondence";
    auto rt = normalize(c, arithmeticSkeleton(plan, plan.rhs));
    auto match = correspond(c, lhs, lt, rt);
    phase = "attribute construction";
    std::vector<S> all;
    for (const auto& t : lt) all.push_back(t.reductions);
    Lowerer target{c, &match.anchors};
    Constructor ctor{c, target, match, lhs,
                     call("unique_axes", {call("concat", std::move(all))})};
    auto rhs = ctor.build(plan.rhs);
    auto checked = normalize(c, rhs.value, &candidate.rewrites);
    std::map<std::string, const Term*> left;
    for (const auto& t : lt) left[footprint(t)] = &t;
    for (const auto& t : checked)
      c.require(
          call("same_set", {left.at(footprint(t))->reductions, t.reductions}));
    c.require(call("==", {lhs.axes, rhs.axes}));
    c.require(call("==", {lhs.shape, rhs.shape}));
    c.require(call("==", {lhs.dtype, rhs.dtype}));
    for (const auto& n : plan.nodes)
      if (n.automatic) {
        auto& fields = candidate.constructors[n.descriptor->value];
        for (std::size_t f = 0; f < n.definition->fields.size(); ++f) {
          auto value = c.canonical(c.field(n, f));
          c.validateSource(value);
          if (value->sort != Symbol::Sort::Positions)
            throw ProofFailure(
                "attribute constructor is not a position sequence");
          fields.push_back(value);
        }
      }
    candidate.guard = c.guards();
    auto proof = completeGuard(plan, candidate, max_work);
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
  auto general = run(plan, max_work, InferenceStrategy::General, {});
  if (general.status != ProofStatus::Unknown || !plan.algebraic ||
      max_work < ProofLimits::routing_min_work)
    return general;
  try {
    auto guard = routingGuard(plan, max_work);
    if (guard.empty()) return general;
    auto routed =
        run(plan, max_work, InferenceStrategy::WholeGroupRouting, guard);
    if (routed.status != ProofStatus::Unknown)
      routed.steps.insert(routed.steps.begin(),
                          "routing: whole source groups; shared retained "
                          "multiplicative groups empty");
    return routed;
  } catch (const ProofFailure& e) {
    general.reason = std::string("source routing: ") + e.describe();
    return general;
  }
}
}  // namespace tepl::core::index_semantics
