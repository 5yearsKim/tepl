#include "src/core/index_semantics/verify.h"

#include <stdexcept>

namespace tepl::core::index_semantics {
namespace {
RuleProof check(const Plan& plan, const RuleProof& candidate,
                std::size_t max_work, bool complete) {
  Context c{plan, max_work};
  RuleProof proof;
  try {
    c.sourceScope(plan.lhs);
    for (const auto& n : plan.nodes)
      if (n.automatic) {
        auto found = candidate.constructors.find(n.descriptor->value);
        if (found == candidate.constructors.end() ||
            found->second.size() != n.definition->fields.size())
          throw std::runtime_error("missing or incorrectly sized constructor");
        for (std::size_t f = 0; f < found->second.size(); ++f) {
          c.validateSource(found->second[f]);
          if (found->second[f]->sort != Symbol::Sort::Positions)
            throw std::runtime_error("constructor must be a position sequence");
          c.solutions[{n.descriptor->value, f}] = found->second[f];
        }
      }
    for (const auto& g : candidate.guard) {
      c.validateSource(g);
      if (g->sort != Symbol::Sort::Boolean)
        throw std::runtime_error("applicability guard must be Boolean");
      c.require(g);
    }
    c.allow_guards = complete;
    Lowerer source{c};
    auto lhs = source.lower(plan.lhs);
    auto lt = normalize(c, lhs.value);
    for (const auto& [id, fields] : candidate.constructors)
      for (const auto& f : fields) c.wellFormed(f);
    for (const auto& g : candidate.guard) c.wellFormed(g);
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
    for (const auto& f : target.obligations) c.require(f);
    auto checked = normalize(c, rhs.value);
    std::map<std::string, const Term*> left;
    for (const auto& t : lt) left[footprint(t)] = &t;
    for (const auto& t : checked) {
      c.require(
          call("same_set", {left.at(footprint(t))->reductions, t.reductions}));
      std::map<std::size_t, E> reads;
      for (const auto& r : left.at(footprint(t))->reads) reads[r->capture] = r;
      for (const auto& r : t.reads)
        c.require(call("==", {reads.at(r->capture)->axes, r->axes}));
    }
    c.require(call("==", {lhs.axes, rhs.axes}));
    c.require(call("==", {lhs.shape, rhs.shape}));
    c.require(call("==", {lhs.dtype, rhs.dtype}));
    proof.constructors = candidate.constructors;
    proof.guard = c.guards;
    proof.status =
        proof.guard.empty() ? ProofStatus::Proved : ProofStatus::Conditional;
    proof.reason = proof.guard.empty()
                       ? "valid for all legal LHS metadata under the selected "
                         "arithmetic policy"
                       : "verified under a sufficient metadata-only guard; "
                         "weakest guard not claimed";
    proof.steps = c.steps;
    proof.steps.push_back(
        "verified: capture access + reduction domains + ordered axes + shape + "
        "dtype + legality");
    proof.lhs_normalized = show(lt);
    proof.rhs_normalized = show(checked);
    for (const auto& [id, g] : c.coordinate_groups)
      proof.axis_definitions.push_back(g);
    for (const auto& t : lt)
      for (const auto& r : t.reads) proof.accesses[r->occurrence] = r->axes;
    for (const auto& t : checked)
      for (const auto& r : t.reads) proof.accesses[r->occurrence] = r->axes;
    proof.lhs_axes = lhs.axes;
    proof.rhs_axes = rhs.axes;
    return proof;
  } catch (const std::exception& e) {
    proof = RuleProof{};
    proof.reason = e.what();
    proof.steps = c.steps;
    return proof;
  }
}
}  // namespace
RuleProof verify(const Plan& plan, const RuleProof& proof, std::size_t work) {
  return check(plan, proof, work, false);
}
RuleProof completeGuard(const Plan& plan, const RuleProof& proof,
                        std::size_t work) {
  auto candidate = proof;
  RuleProof checked;
  // Well-definedness of generated conditions can expose further preconditions.
  // Close that finite dependency set, then check the fixed guard independently.
  for (std::size_t round = 0; round < 6; ++round) {
    candidate = check(plan, candidate, work, true);
    if (candidate.status == ProofStatus::Unknown) return candidate;
    checked = verify(plan, candidate, work);
    if (checked.status != ProofStatus::Unknown) return checked;
  }
  checked.reason = "guard closure did not converge: " + checked.reason;
  return checked;
}
}  // namespace tepl::core::index_semantics
