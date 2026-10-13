#include "src/core/index_semantics/verify.h"

#include <stdexcept>

namespace tepl::core::index_semantics {
namespace {
RuleProof check(const Plan& plan, const RuleProof& candidate,
                std::size_t max_work) {
  Context c{plan, max_work};
  RuleProof proof;
  try {
    c.sourceScope(plan.lhs);
    for (const auto& n : plan.nodes)
      if (n.automatic) {
        auto found = candidate.constructors.find(n.descriptor->value);
        if (found == candidate.constructors.end() ||
            found->second.size() != n.definition->fields.size())
          throw ProofFailure("missing or incorrectly sized constructor",
                             FailureKind::InvalidCandidate);
        for (std::size_t f = 0; f < found->second.size(); ++f) {
          c.validateSource(found->second[f]);
          if (found->second[f]->sort != Symbol::Sort::Positions)
            throw ProofFailure("constructor must be a position sequence",
                               FailureKind::InvalidCandidate);
          c.assignAttribute(n.descriptor->value, f, found->second[f]);
        }
      }
    Lowerer source{c};
    // A second composition uses legality learned across the complete source,
    // including shapes shared through pointwise consumers.
    source.lower(plan.lhs);
    Lowerer composed{c};
    auto lhs = composed.lower(plan.lhs);
    c.enableSetReasoning();
    auto lt = normalize(c, lhs.value);
    for (const auto& [id, fields] : candidate.constructors)
      for (const auto& f : fields) c.wellFormed(f);
    for (const auto& g : candidate.guard) {
      c.validateSource(g);
      if (g->sort != Symbol::Sort::Boolean)
        throw ProofFailure("candidate obligation must be Boolean",
                           FailureKind::InvalidCandidate);
      c.wellFormed(g);
      c.require(g);  // An obligation, never an additional premise.
    }
    for (const auto& n : plan.nodes)
      if (n.automatic) {
        const auto& fields = candidate.constructors.at(n.descriptor->value);
        for (std::size_t f = 0; f < fields.size(); ++f)
          c.require(call("==", {c.field(n, f), fields[f]}));
      }
    auto rt = normalize(c, arithmeticSkeleton(plan, plan.rhs));
    auto correspondence = correspond(c, lhs, lt, rt);
    Lowerer target{c, &correspondence.anchors};
    auto rhs = target.lower(plan.rhs);
    auto checked = normalize(c, rhs.value);
    auto simplified_lhs = collectTerms(c, lt);
    auto simplified_rhs = collectTerms(c, checked);
    equivalentTerms(c, simplified_lhs, simplified_rhs);
    c.require(call("==", {lhs.axes, rhs.axes}), GuardKind::CoordinateOrder);
    c.require(call("==", {lhs.shape, rhs.shape}));
    c.require(call("==", {lhs.dtype, rhs.dtype}));
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
    proof.lhs_axes = lhs.axes;
    proof.rhs_axes = rhs.axes;
    return proof;
  } catch (const ProofFailure& e) {
    proof = RuleProof{};
    proof.reason = e.describe();
    proof.steps = c.steps();
    return proof;
  }
}
}  // namespace
RuleProof verify(const Plan& plan, const RuleProof& proof, std::size_t work) {
  return check(plan, proof, work);
}
}  // namespace tepl::core::index_semantics
