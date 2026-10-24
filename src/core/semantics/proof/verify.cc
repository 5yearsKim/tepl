#include "src/core/semantics/proof/verify.h"

#include <functional>
#include <stdexcept>

#include "src/core/semantics/symbolic/normalize.h"

namespace tepl::core::semantics {
RuleProof verify(const Plan& plan, const ConstructorCandidate& candidate,
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
          if (found->second[f]->sort != Symbol::Sort::Positions &&
              !(n.definition->extent_fields.contains(f) &&
                found->second[f]->sort == Symbol::Sort::Extents))
            throw ProofFailure("constructor must be a position sequence",
                               FailureKind::InvalidCandidate);
          c.assignAttribute(n.descriptor->value, f, found->second[f]);
        }
      }
    auto lhs = lowerSource(c);
    for (const auto& value : plan.derived_metadata) c.checkMetadata(value);
    for (const auto& [value, type] : plan.constructed_metadata)
      c.checkMetadataType(value, type);
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
    // Identical operation trees need no access-correspondence search. This
    // also preserves repeated capture identity and works for singleton/empty
    // extents without invoking a completeness-domain assumption.
    std::function<bool(std::size_t, std::size_t)> identical =
        [&](std::size_t a, std::size_t b) {
          c.tick();
          const auto& left = plan.nodes.at(a);
          const auto& right = plan.nodes.at(b);
          if (left.capture || right.capture)
            return left.capture == right.capture;
          if (left.literal || right.literal)
            return left.literal == right.literal &&
                   left.literal_dtype == right.literal_dtype;
          if (left.operation != right.operation ||
              left.definition != right.definition ||
              left.inputs.size() != right.inputs.size())
            return false;
          for (std::size_t i = 0; i < left.inputs.size(); ++i)
            if (!identical(left.inputs[i], right.inputs[i])) return false;
          if (bool(left.descriptor) != bool(right.descriptor)) return false;
          if (left.descriptor)
            for (std::size_t f = 0; f < left.definition->fields.size(); ++f)
              if (!c.entails(call("==", {c.field(left, f), c.field(right, f)})))
                return false;
          return true;
        };
    if (identical(plan.lhs, plan.rhs)) {
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
        proof.lhs_normalized = show(collectTerms(c, normalize(c, lhs.value)));
        proof.rhs_normalized = proof.lhs_normalized;
      } catch (const ProofFailure&) {
      }
      for (const auto& [id, group] : c.source().groups)
        proof.axis_definitions.push_back(group);
      return proof;
    }
    auto lt = normalize(c, lhs.value);
    auto rt = normalize(c, arithmeticSkeleton(plan, plan.rhs));
    auto correspondence = correspond(c, lhs, lt, rt);
    Lowerer target{c, &correspondence.anchors};
    target.interfaces = &correspondence.interfaces;
    target.expected_dtype = lhs.dtype;
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
    proof.guard = c.guards();
    proof.guard_origins = c.guardOrigins();
    proof.reason = e.describe();
    proof.steps = c.steps();
    return proof;
  }
}
}  // namespace tepl::core::semantics
