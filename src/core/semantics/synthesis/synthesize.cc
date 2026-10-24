#include "src/core/semantics/synthesis/synthesize.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/synthesis/index_inference.h"
#include "src/core/semantics/synthesis/ordered/construct.h"

namespace tepl::core::semantics {
namespace {
struct Constructor {
  Context& c;
  Lowerer& lowerer;
  const Correspondence& match;
  const TensorMeaning& source;
  S reductions;
  const Terms& source_terms;
  bool unresolvedIndexAttributes(const Instance& n) {
    bool found = false;
    auto metadata = [&](auto&& self, const Meta& m) -> void {
      if (m.kind == Meta::Kind::Attribute)
        found |= containsHole(c.field(n, m.index));
      for (const auto& a : m.args) self(self, *a);
    };
    auto indices = [&](auto&& self, const IndexValue& v) -> void {
      if (v.parameter) metadata(metadata, *v.parameter);
      for (const auto& a : v.args) self(self, *a);
    };
    auto values = [&](auto&& self, const Value& v) -> void {
      if (v.access) indices(indices, *v.access);
      for (const auto& a : v.args) self(self, *a);
    };
    values(values, *n.definition->value);
    return found;
  }
  void inferReadAttributes(const Instance& n,
                           const std::vector<TensorMeaning>& inputs,
                           const std::vector<S>& groups) {
    IndexAssignments inferred;
    auto merge = [&](IndexAssignments& into, const IndexAssignments& values) {
      for (const auto& [id, value] : values) {
        auto [old, added] = into.emplace(id, value);
        if (!added && !equalExtentArithmetic(c, old->second, value))
          throw ProofFailure(
              "ambiguous or conflicting calculated-read attribute "
              "constructors");
      }
    };
    auto read = [&](const E& pattern) {
      std::optional<IndexAssignments> solution;
      for (const auto& term : source_terms)
        for (const auto& actual : term.reads) {
          c.tick();
          if (actual->capture != pattern->capture) continue;
          auto candidate = matchIndexAttributes(
              c, pattern->access ? pattern->access : coordinates(pattern->axes),
              actual->access ? actual->access : coordinates(actual->axes));
          if (!candidate) continue;
          if (!solution)
            solution = *candidate;
          else
            merge(*solution, *candidate);
        }
      if (!solution)
        throw ProofFailure(
            "no scoped coordinate match for calculated-read attributes; "
            "composed or inverse inference may be required");
      merge(inferred, *solution);
    };
    auto captures = [&](auto&& self, const E& e) -> void {
      if (e->kind == Expr::Kind::Read) read(e);
      for (const auto& a : e->args) self(self, a);
    };
    auto values = [&](auto&& self, const Value& v) -> void {
      if (v.kind == Value::Kind::Read) {
        auto indices = lowerer.indexValue(*v.access, n, inputs, groups);
        // Compose through an already constructed input, including reindexings.
        // Only proposals are produced here; final lowering checks every domain.
        auto expression = lowerer.evaluateAt(inputs[v.index].value,
                                             inputs[v.index].axes, indices);
        captures(captures, expression);
      }
      for (const auto& a : v.args) self(self, *a);
    };
    values(values, *n.definition->value);
    if (inferred.empty())
      throw ProofFailure(
          "calculated reads provide no direct attribute constructor");
    for (const auto& [id, value] : inferred)
      c.assignAttribute(id.first, id.second, value);
    c.note("index-coefficient-matching");
  }
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
    if (n.capture || n.literal) return lowerer.lower(id);
    std::vector<TensorMeaning> inputs;
    for (auto child : n.inputs) inputs.push_back(build(child));
    if (!n.automatic) return lowerer.operation(id, std::move(inputs));
    std::set<std::size_t> reduced;
    std::function<void(const Value&)> scan = [&](const Value& v) {
      if (v.kind == Value::Kind::Reduce) reduced.insert(v.index);
      for (const auto& a : v.args) scan(*a);
    };
    scan(*n.definition->value);
    const auto trial_obligations = lowerer.obligations.size();
    const bool infer_indices = unresolvedIndexAttributes(n);
    std::vector<std::size_t> deferred_domains;
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
        // A single non-reduced domain covers the output interface even when
        // an identity coordinate map reuses axes from its input. Subtracting
        // input axes would erase that frame before calculated-read matching.
        if (infer_indices && !reduced.contains(g) &&
            n.definition->groups.size() == 1)
          wanted = match.interfaces.at(id);
        auto order = c.filter(match.interfaces.at(id), wanted);
        auto domain = lowerer.metadata(*group.domain, n, inputs, groups);
        if (infer_indices && containsHole(c.resolve(domain)) &&
            c.resolve(domain)->kind != Symbol::Kind::Hole)
          deferred_domains.push_back(g);
        else
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
    if (infer_indices) {
      inferReadAttributes(n, inputs, groups);
      for (auto g : deferred_domains)
        assign(*n.definition->groups[g].domain, n, inputs, groups,
               c.extents(groups[g]));
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
    // Trial metadata lowering can contain holes. The completed operation is
    // lowered afresh and checks its own obligations, without trial premises.
    lowerer.obligations.resize(trial_obligations);
    return lowerer.operation(id, std::move(inputs));
  }
};
}  // namespace
namespace {
RuleProof run(const Plan& plan, std::size_t max_work,
              const ordered::Admission* admitted = nullptr) {
  Context c{plan, max_work};
  ConstructorCandidate candidate;
  std::vector<RuleProof::RewriteStep> rewrites;
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
    phase = "metadata definedness";
    for (const auto& value : plan.derived_metadata) c.checkMetadata(value);
    for (const auto& [value, type] : plan.constructed_metadata)
      c.checkMetadataType(value, type);
    auto lt = normalize(c, lhs.value, &rewrites);
    phase = "coordinate correspondence";
    auto rt = normalize(c, arithmeticSkeleton(plan, plan.rhs));
    auto match = correspond(c, lhs, lt, rt);
    phase = "attribute construction";
    std::vector<S> all;
    for (const auto& t : lt) all.push_back(t.reductions);
    Lowerer target{c, &match.anchors};
    target.interfaces = &match.interfaces;
    target.expected_dtype = lhs.dtype;
    Constructor ctor{c,
                     target,
                     match,
                     lhs,
                     call("unique_axes", {call("concat", std::move(all))}),
                     lt};
    auto rhs = admitted ? ordered::construct(c, target, *admitted,
                                             ordered::deriveRequirements(
                                                 c, *admitted, match, lhs.axes),
                                             ctor.reductions)
                        : ctor.build(plan.rhs);
    auto checked = normalize(c, rhs.value, &rewrites);
    auto simplified_lhs = collectTerms(c, lt, &rewrites);
    auto simplified_rhs = collectTerms(c, checked, &rewrites);
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
    auto proof = verify(plan, candidate, max_work);
    // Construction provenance is diagnostic only. Verification above used a
    // fresh context and checked the completed accesses and domains
    // independently.
    const std::string coefficient_law = "law: index-coefficient-matching";
    if (proof.status == ProofStatus::Proved &&
        std::find(c.steps().begin(), c.steps().end(), coefficient_law) !=
            c.steps().end())
      proof.steps.push_back(coefficient_law);
    proof.rewrites = std::move(rewrites);
    if (admitted && proof.status == ProofStatus::Proved)
      proof.steps.push_back("law: ordered-requirements-construction");
    return proof;
  } catch (const ProofFailure& e) {
    RuleProof unknown;
    unknown.guard = c.guards();
    unknown.guard_origins = c.guardOrigins();
    unknown.reason = phase + ": " + e.describe();
    unknown.steps = c.steps();
    return unknown;
  }
}
}  // namespace
RuleProof synthesize(const Plan& plan, std::size_t max_work) {
  // Propose descriptor copies for structurally identical trees. Correctness is
  // still checked in a fresh verifier, including explicit RHS definitions.
  try {
    Context c{plan, max_work};
    c.sourceScope(plan.lhs);
    Lowerer source{c};
    source.lower(plan.lhs);
    ConstructorCandidate candidate;
    std::function<bool(std::size_t, std::size_t)> copy = [&](std::size_t a,
                                                             std::size_t b) {
      c.tick();
      const auto& left = plan.nodes.at(a);
      const auto& right = plan.nodes.at(b);
      if (left.capture || right.capture) return left.capture == right.capture;
      if (left.literal || right.literal)
        return left.literal == right.literal &&
               left.literal_dtype == right.literal_dtype;
      if (left.operation != right.operation ||
          left.inputs.size() != right.inputs.size())
        return false;
      for (std::size_t i = 0; i < left.inputs.size(); ++i)
        if (!copy(left.inputs[i], right.inputs[i])) return false;
      if (right.automatic) {
        std::vector<S> fields;
        for (std::size_t f = 0; f < right.definition->fields.size(); ++f)
          fields.push_back(c.field(left, f));
        auto [found, added] =
            candidate.constructors.emplace(right.descriptor->value, fields);
        if (!added) {
          if (found->second.size() != fields.size()) return false;
          for (std::size_t f = 0; f < fields.size(); ++f)
            if (!equal(found->second[f], fields[f])) return false;
        }
      }
      return true;
    };
    if (copy(plan.lhs, plan.rhs)) {
      auto proof = verify(plan, candidate, max_work);
      if (proof.status == ProofStatus::Proved) return proof;
    }
  } catch (const ProofFailure&) {
    // Failed candidate construction is not an impossibility result.
  }
  const auto admitted = ordered::admit(plan);
  if (admitted.admitted()) {
    auto proof = run(plan, max_work, &admitted);
    if (proof.status == ProofStatus::Proved) return proof;
    // A bounded symbolic attempt is not an impossibility decision. Keep the
    // generic path and its established behavior while correspondence and the
    // structural certificate checker are developed independently.
    return run(plan, max_work);
  }
  return run(plan, max_work);
}
}  // namespace tepl::core::semantics
