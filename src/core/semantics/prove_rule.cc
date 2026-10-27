#include "src/core/semantics/prove_rule.h"

#include <chrono>

#include "src/core/attr_synth/synthesize.h"
#include "src/core/semantics/proof/verify.h"

namespace tepl::core::semantics {
namespace {
template <class F>
auto measure(F run, double* milliseconds) {
  if (!milliseconds) return run();
  const auto start = std::chrono::steady_clock::now();
  auto result = run();
  *milliseconds += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  return result;
}
}  // namespace

RuleProof proveRule(const Plan& plan, std::size_t max_work, ProofTiming* timing,
                    std::chrono::steady_clock::time_point deadline) {
  if (timing) *timing = {};
  RuleProof proof;
  const auto expired = [&] {
    if (deadline == std::chrono::steady_clock::time_point::max()) return false;
    if (std::chrono::steady_clock::now() < deadline) return false;
    proof = {};
    proof.reason = "time limit exceeded: inference pipeline deadline";
    return true;
  };
  for (auto strategy :
       {attr_synth::Strategy::Identity, attr_synth::Strategy::Ordered,
        attr_synth::Strategy::General}) {
    if (expired()) return proof;
    if (timing) ++timing->construction_attempts;
    auto result = measure(
        [&] {
          return attr_synth::synthesize(plan, strategy, max_work, deadline);
        },
        timing ? &timing->construction_ms : nullptr);
    if (expired()) return proof;
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
    if (timing) ++timing->verification_attempts;
    proof = measure(
        [&] { return verify(plan, result.candidate, max_work, deadline); },
        timing ? &timing->verification_ms : nullptr);
    if (expired()) return proof;
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
