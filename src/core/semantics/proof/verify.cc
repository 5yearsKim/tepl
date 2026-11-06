#include "src/core/semantics/proof/verify.h"

#include <functional>

#include "src/core/semantics/symbolic/normalize.h"

namespace tepl::core::semantics {
RuleProof verify(const Plan& plan, const ConstructorCandidate& candidate,
                 std::size_t max_work,
                 std::chrono::steady_clock::time_point deadline) {
  Context c{plan, max_work, deadline};
  RuleProof proof;
  auto retainPredicateAxes = [&] {
    std::set<std::pair<std::size_t, std::size_t>> groups;
    std::set<const Symbol*> visited;
    for (const auto& axis : proof.axis_definitions)
      if (axis->kind == Symbol::Kind::AxisGroup)
        groups.insert({axis->owner, axis->index});
    std::function<void(const SymbolPtr&)> visit = [&](const SymbolPtr& value) {
      if (!visited.insert(value.get()).second) return;
      if (value->kind == Symbol::Kind::AxisGroup &&
          groups.insert({value->owner, value->index}).second)
        proof.axis_definitions.push_back(value);
      for (const auto& child : value->args) visit(child);
    };
    for (const auto& q : candidate.feasibility) visit(q);
  };
  try {
    c.sourceScope(plan.lhs);
    for (const auto& n : plan.nodes)
      if (plan.needsSynthesis(n)) {
        auto found = candidate.constructors.find(n.descriptor->value);
        if (found == candidate.constructors.end() ||
            found->second.size() != n.definition->fields.size())
          throw ProofFailure("missing or incorrectly sized constructor",
                             FailureKind::InvalidCandidate);
        for (std::size_t f = 0; f < found->second.size(); ++f) {
          c.checkConstructor(n, f, found->second[f]);
          c.assignAttribute(n.descriptor->value, f, found->second[f]);
        }
      }
    auto lhs = lowerSource(c);
    // Establish source-only predicate definedness before using q as a premise.
    // Builder definedness and semantic obligations are checked under C && q.
    for (const auto& condition : candidate.feasibility) {
      c.validateSource(condition);
      if (condition->sort != Symbol::Sort::Boolean)
        throw ProofFailure("feasibility obligation must be Boolean",
                           FailureKind::InvalidCandidate);
      c.wellFormed(condition);
      c.learn(condition);
    }
    for (const auto& value : plan.derived_metadata) c.checkMetadata(value);
    for (const auto& [value, type] : plan.constructed_metadata)
      c.checkMetadataType(value, type);
    for (const auto& n : plan.nodes)
      if (plan.needsSynthesis(n)) {
        const auto& fields = candidate.constructors.at(n.descriptor->value);
        for (std::size_t f = 0; f < fields.size(); ++f) {
          if (plan.attributes(n).isAuto(f)) {
            c.wellFormed(fields[f]);
          } else {
            // Supplied source expressions may contain comprehensions. Check
            // their evaluation and scoped structure without normalizing away
            // any undefined subexpression or unbound variable.
            c.checkMetadata(fields[f]);
            c.wellFormed(fields[f]);
          }
        }
      }
    for (const auto& g : candidate.guard) {
      c.validateSource(g);
      if (g->sort != Symbol::Sort::Boolean)
        throw ProofFailure("candidate obligation must be Boolean",
                           FailureKind::InvalidCandidate);
      c.wellFormed(g);
      c.require(g);  // An obligation, never an additional premise.
    }
    for (const auto& n : plan.nodes)
      if (plan.needsSynthesis(n)) {
        const auto& fields = candidate.constructors.at(n.descriptor->value);
        for (std::size_t f = 0; f < fields.size(); ++f)
          c.require(call("==", {c.field(n, f), fields[f]}));
      }
    // Identical operation trees need no access-correspondence search. This
    // also preserves repeated capture identity and works for singleton/empty
    // extents without invoking a completeness-domain assumption.
    auto identical =
        matchOperationTree(c, [&](const Instance& left, const Instance& right) {
          if (bool(left.descriptor) != bool(right.descriptor)) return false;
          if (left.descriptor)
            for (std::size_t f = 0; f < left.definition->fields.size(); ++f)
              if (!c.entails(call("==", {c.field(left, f), c.field(right, f)})))
                return false;
          return true;
        });
    if (identical) {
      proof.feasibility = candidate.feasibility;
      proof.constructors = candidate.constructors;
      proof.status = ProofStatus::Proved;
      proof.reason =
          "identical operations, operands, and verified attributes under "
          "declared conditions";
      proof.steps = c.steps();
      proof.steps.push_back(
          "verified: identical operation tree, including repeated captures");
      proof.lhs_axes = proof.rhs_axes = lhs.axes;
      // Keep diagnostic normal forms when available. The identity proof above
      // does not depend on arithmetic normalization succeeding.
      try {
        proof.lhs_normalized =
            show(collectTerms(c, normalize(c, lhs.value, &proof.rewrites)));
        proof.rhs_normalized = proof.lhs_normalized;
      } catch (const ProofFailure&) {
      }
      for (const auto& [id, group] : c.source().groups)
        proof.axis_definitions.push_back(group);
      retainPredicateAxes();
      return proof;
    }
    auto lt = normalize(c, lhs.value, &proof.rewrites);
    auto rt = normalize(c, arithmeticSkeleton(c, plan.rhs));
    auto correspondence = correspond(c, lhs, lt, rt);
    Lowerer target{c, &correspondence.anchors};
    target.interfaces = &correspondence.interfaces;
    target.expected_dtype = lhs.dtype;
    auto rhs = target.lower(plan.rhs);
    auto checked = normalize(c, rhs.value, &proof.rewrites);
    auto simplified_lhs = collectTerms(c, lt, &proof.rewrites);
    auto simplified_rhs = collectTerms(c, checked, &proof.rewrites);
    equivalentTerms(c, simplified_lhs, simplified_rhs);
    c.require(call("==", {lhs.axes, rhs.axes}), GuardKind::CoordinateOrder);
    c.require(call("==", {lhs.shape, rhs.shape}));
    c.require(call("==", {lhs.dtype, rhs.dtype}));
    proof.feasibility = candidate.feasibility;
    proof.constructors = candidate.constructors;
    proof.guard = c.guards();
    proof.guard_origins = c.guardOrigins();
    proof.status = ProofStatus::Proved;
    proof.reason =
        "valid for all legal LHS metadata satisfying the declared conditions "
        "under the selected arithmetic policy";
    proof.steps = c.steps();
    proof.steps.push_back(
        "verified: capture access + reduction domains + ordered axes + shape + "
        "dtype + legality");
    proof.lhs_normalized = show(simplified_lhs);
    proof.rhs_normalized = show(simplified_rhs);
    for (const auto& [id, g] : c.source().groups)
      proof.axis_definitions.push_back(g);
    for (const auto& t : lt)
      for (const auto& r : t.reads) proof.accesses[r->occurrence] = r->axes;
    for (const auto& t : checked)
      for (const auto& r : t.reads) proof.accesses[r->occurrence] = r->axes;
    retainPredicateAxes();
    proof.lhs_axes = lhs.axes;
    proof.rhs_axes = rhs.axes;
    return proof;
  } catch (const ProofFailure& e) {
    proof = RuleProof{};
    proof.guard = c.guards();
    proof.guard_origins = c.guardOrigins();
    proof.reason = e.describe();
    proof.steps = c.steps();
    return proof;
  }
}
}  // namespace tepl::core::semantics
