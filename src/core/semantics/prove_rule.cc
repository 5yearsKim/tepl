#include "src/core/semantics/prove_rule.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <functional>

#include "src/core/attr_synth/ordered/contract.h"
#include "src/core/attr_synth/ordered/extensions.h"
#include "src/core/attr_synth/synthesize.h"
#include "src/core/semantics/proof/verify.h"
#include "src/inference/metadata_facts.h"
#include "src/shared/catalog/dtype_contracts.h"

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
  if (plan.core_backend) {
    auto admission = attr_synth::ordered::admit(plan);
    if (!admission.admitted())
      admission =
          attr_synth::ordered::admit(plan, inference::Mode::Synthesize, true);
    const bool extensions =
        admission.admitted() && inference::hasExtensions(admission.program);
    if (!admission.admitted() ||
        (!extensions && !plan.derived_metadata.empty())) {
      proof.reason =
          "core backend requires admitted automatic fields without derive "
          "helpers: " +
          admission.reason;
      return proof;
    }
    if (extensions) {
      try {
        inference::emitExtension(admission.program);
      } catch (const std::invalid_argument&) {
        proof.reason = "outside supported view/reindexing schemas";
        return proof;
      }
    }
    if (!completed::coreTypes(admission.program,
                              [&](std::size_t id) -> const metadata::Program* {
                                for (const auto& n : plan.nodes)
                                  if (n.operation && n.operation->value == id)
                                    return n.dtype.get();
                                return nullptr;
                              })) {
      proof.reason =
          "core backend requires canonical dtype contracts and source numeric "
          "legality";
      return proof;
    }
    inference::MetadataFacts facts;
    auto integerValue = [](const SymbolPtr& x, std::int64_t& n) {
      if (x->kind != Symbol::Kind::Integer) return false;
      auto r =
          std::from_chars(x->name.data(), x->name.data() + x->name.size(), n);
      return r.ec == std::errc{} && r.ptr == x->name.data() + x->name.size();
    };
    auto fact = [&](const SymbolPtr& a, const SymbolPtr& b) {
      if (auto x = attr_synth::ordered::metadataTerm(a))
        if (auto y = attr_synth::ordered::metadataTerm(b))
          facts.equalities.emplace_back(*x, *y);
      if (a->kind == Symbol::Kind::Call && a->name == "len" &&
          a->args.size() == 1) {
        std::int64_t n;
        if (a->args[0]->kind == Symbol::Kind::Field && integerValue(b, n) &&
            n >= 0) {
          auto x = a->args[0];
          facts.lengths.emplace(std::pair{x->owner, x->index}, n);
        }
      }

      if (a->kind == Symbol::Kind::Field && b->kind == Symbol::Kind::List) {
        facts.lengths.emplace(std::pair{a->owner, a->index}, b->args.size());
        inference::List list;
        for (const auto& x : b->args) {
          std::int64_t n;
          if (!integerValue(x, n)) return;
          list.push_back(n);
        }
        facts.fields.emplace(std::pair{a->owner, a->index}, list);
      }
      std::int64_t n;
      if (a->kind == Symbol::Kind::Call && a->name == "len" &&
          a->args.size() == 1 && a->args[0]->kind == Symbol::Kind::Shape &&
          integerValue(b, n) && n >= 0) {
        facts.ranks.emplace(a->args[0]->owner, n);
      }
    };
    std::function<void(const SymbolPtr&)> collect = [&](const auto& x) {
      if (x->kind != Symbol::Kind::Call) return;
      if (x->name == "&&")
        for (const auto& a : x->args) collect(a);
      else if (x->name == "==" && x->args.size() == 2) {
        fact(x->args[0], x->args[1]);
        fact(x->args[1], x->args[0]);
      }
    };
    for (const auto& a : plan.assumptions) collect(a);
    if (expired()) return proof;
    if (extensions && !attr_synth::ordered::suppliedExtensionFields(
                          plan, admission.program, facts)) {
      proof.reason =
          "core schema cannot validate supplied fields or derive helpers";
      return proof;
    }
    proof.status = ProofStatus::Proved;
    proof.core = std::move(admission.program);
    proof.reason =
        extensions ? "core schema and supplied metadata validated; sound for "
                     "all legal extents"
                   : "conditional DAG builder is sound for all legal extents";
    proof.steps.push_back("verified: conditional core emission theorem");
    return proof;
  }
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
