#include "src/core/semantics/prove_rule.h"

#include "src/core/attr_synth/synthesize.h"
#include "src/core/semantics/proof/verify.h"

namespace tepl::core::semantics {

RuleProof proveRule(const Plan& plan, std::size_t max_work) {
  RuleProof proof;
  for (auto strategy :
       {attr_synth::Strategy::Identity, attr_synth::Strategy::Ordered,
        attr_synth::Strategy::General}) {
    auto result = attr_synth::synthesize(plan, strategy, max_work);
    if (!result.constructed) {
      proof = {};
      proof.guard = std::move(result.candidate.guard);
      proof.guard_origins = std::move(result.guard_origins);
      proof.reason = std::move(result.reason);
      proof.steps = std::move(result.steps);
      continue;
    }
    // Verification rebuilds its context from the plan and candidate. No facts
    // or attribute assignments from the synthesis attempt become premises.
    proof = verify(plan, result.candidate, max_work);
    if (proof.status == ProofStatus::Proved) {
      proof.steps.insert(proof.steps.end(), result.steps.begin(),
                         result.steps.end());
      return proof;
    }
    // Construction or verification failure is not an impossibility decision.
    // In particular, an ordered attempt must not prevent the general fallback.
  }
  return proof;
}

}  // namespace tepl::core::semantics
