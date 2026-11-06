#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/attr_synth/synthesize.h"
#include "src/core/pipeline.h"
#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/prove_rule.h"
#include "src/core/semantics/symbolic/expr.h"
#include "src/rewrite/engine.h"

namespace {
using namespace tepl;
std::string dialect;
void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
auto parseRule(const std::string& text) {
  auto parsed = parse(dialect + text, "reasoning.tepl");
  expect(parsed.ok(), "fixture parses");
  return parsed;
}
executable::ExecutableProgram compile(const std::string& text) {
  auto parsed = parseRule(text);
  auto compiled =
      core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic});
  if (!compiled.ok())
    for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
  expect(compiled.ok(), "rule proved under its written conditions");
  return std::move(*compiled.program);
}
void rejects(const std::string& text) {
  auto parsed = parseRule(text);
  expect(
      !core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic}).ok(),
      "unproved replacement is rejected at compile time: " + text);
}
void conditionalFeasibility() {
  auto p = compile(R"(
rule conditional {A:[Dims...] (dot A B)=>(dot B A) where len(Dims)==1 }
graph skip {input A:f32[2] input B:f32[3] yield(dot[lhs_contracting_dimensions=[],rhs_contracting_dimensions=[],lhs_batching_dimensions=[],rhs_batching_dimensions=[]] A B) }
graph apply {input A:f32[2] input B:f32[2] yield(dot[lhs_contracting_dimensions=[0],rhs_contracting_dimensions=[0],lhs_batching_dimensions=[],rhs_batching_dimensions=[]] A B) }
graph guard {input A:f32[] input B:f32[] yield(dot[lhs_contracting_dimensions=[],rhs_contracting_dimensions=[],lhs_batching_dimensions=[],rhs_batching_dimensions=[]] A B) }
)");
  expect(!p.ir().rules[0].feasibility.empty(),
         "compiled q survives verification");
  auto loaded = executable::load(executable::serialize(p));
  rewrite::Options options;
  options.max_steps = 1;
  for (const auto* program : {&p, &loaded}) {
    expect(rewrite::run(*program, program->ir().graphs[0], options)
                   .rewrite_count == 0,
           "C true, q false skips normally before a partial attribute builder");
    auto out = rewrite::run(*program, program->ir().graphs[1], options);
    expect(
        out.rewrite_count == 1 && out.graph.nodes.back().shape ==
                                      std::optional<std::vector<std::uint64_t>>(
                                          std::vector<std::uint64_t>{}),
        "same compiled rule succeeds when q holds");
    expect(rewrite::run(*program, program->ir().graphs[2], options)
                   .rewrite_count == 0,
           "explicit guard remains enforced");
  }
}
void identityAndPracticalDomains() {
  auto identity = compile("rule identity {(dot A B)=>(dot A B)}");
  const auto& r = identity.ir().rules[0];
  expect(
      r.conditions.empty() && !r.inference && r.derivations.size() == 1,
      "identity needs only an attribute constructor, with no inferred guard");
  compile("rule repeated {(dot A A)=>(dot A A)}");
  auto strict_identity = parseRule(R"(
rule same {(d: dot (add A B) C)=>(dot (add A B) C)
 where d.axes.B.positions == [] }
)");
  expect(
      core::compile(*strict_identity.program, {core::SemanticPolicy::kStrict})
          .ok(),
      "source positions and identical trees do not require algebraic "
      "normalization");
  // The generic verifier is not restricted by the exact solver's extent domain.
  for (auto extent : {0, 1, 2}) {
    std::string fixture = R"(
rule compose {(transpose (transpose A))=>(transpose A)}
graph g { input A:f32[EXTENT,3]
 let x=(transpose[permutation=[1,0]] A)
 yield (transpose[permutation=[0,1]] x) }
)";
    fixture.replace(fixture.find("EXTENT"), 6, std::to_string(extent));
    auto p = compile(fixture);
    auto loaded = executable::load(executable::serialize(p));
    auto out = rewrite::run(loaded, loaded.ir().graphs[0]);
    expect(out.rewrite_count == 1 &&
               out.graph.nodes[out.graph.root].shape ==
                   std::optional<std::vector<std::uint64_t>>(
                       {3, static_cast<std::uint64_t>(extent)}),
           "verified constructors execute on zero, singleton, and ordinary "
           "extents");
  }
}
void mixedAttributes() {
  auto p = compile(R"(
rule mixed {
 (d: dot (p: transpose A) B) => (dot (q: transpose A) B)
 derive { q.attrs=p.attrs }
}
)");
  expect(!p.ir().rules[0].inference && p.ir().rules[0].derivations.size() == 2,
         "explicit and synthesized RHS descriptors coexist");
  rejects(R"(
rule invalid {
 (p: transpose A) => (q: transpose A)
 derive {q.attrs={permutation: []} }
}
)");
  compile("rule conditionally_commuting {(dot A B)=>(dot B A)}");
  rejects("rule unequal {(add A A)=>A}");
  compile(R"(
rule conditionally_identity {(p: transpose A)=>A
 where subsequence(A.axes.order,concat(A.axes.order,A.axes.order)) }
)");
}
void independentVerification() {
  using namespace core::semantics;
  auto parsed = parseRule("rule identity {(dot A B)=>(dot A B)}");
  auto inspected =
      core::inspect(*parsed.program, {core::SemanticPolicy::kAlgebraic, true});
  expect(inspected.ok(), "inspect identity proof");
  auto plan = *inspected.program->rules[0].semantic_plan;
  auto proposal =
      core::attr_synth::synthesize(plan, core::attr_synth::Strategy::Identity);
  expect(proposal.constructed &&
             verify(plan, proposal.candidate).status == ProofStatus::Proved,
         "candidate generation can be verified independently");
  expect(verify(plan, proposal.candidate, 0).status == ProofStatus::Unknown,
         "a constructed candidate does not bypass the verifier's budget");
  auto candidate = proposal.candidate;
  candidate.constructors.begin()->second[0] = make(Symbol::Kind::List);
  expect(verify(plan, candidate).status == ProofStatus::Unknown,
         "fresh verification rejects a corrupted constructor");
  auto associative = parseRule("rule r {(add (add A B) C)=>(add A (add B C))}");
  auto info_associative = core::inspect(
      *associative.program, {core::SemanticPolicy::kAlgebraic, true});
  expect(info_associative.ok(), "associativity fixture checks");
  auto checked = verify(*info_associative.program->rules[0].semantic_plan, {});
  expect(checked.status == ProofStatus::Proved && !checked.rewrites.empty(),
         "standalone verification owns arithmetic rewrite evidence");
  auto unguarded =
      parseRule("rule r {(d: dot (inner: dot A B) C)=>(dot A (dot B C))}");
  auto info = core::inspect(*unguarded.program,
                            {core::SemanticPolicy::kAlgebraic, true});
  expect(info.ok() &&
             !info.program->rules[0].semantic_plan->proof.guard_origins.empty(),
         "failed verification preserves unresolved-obligation provenance");
}
void synthesisBoundary() {
  using namespace core::semantics;
  using core::attr_synth::Strategy;
  auto inspect = [&](const std::string& text) {
    auto parsed = parseRule(text);
    auto info = core::inspect(*parsed.program,
                              {core::SemanticPolicy::kAlgebraic, true});
    expect(info.ok(), "synthesis boundary fixture checks");
    return *info.program->rules[0].semantic_plan;
  };
  auto incorrect = inspect("rule incorrect {(add A A)=>A}");
  auto unchecked = core::attr_synth::synthesize(incorrect, Strategy::General);
  expect(unchecked.constructed &&
             verify(incorrect, unchecked.candidate).status ==
                 ProofStatus::Unknown &&
             proveRule(incorrect).status == ProofStatus::Unknown,
         "general construction proposes metadata; only verification accepts "
         "equivalence");
  // Identical operation trees alone cannot justify explicit RHS attributes.
  auto invalid = inspect(R"(
rule invalid {(p: transpose A)=>(q: transpose A)
 derive {q.attrs={permutation: []} }}
)");
  auto proposal = core::attr_synth::synthesize(invalid, Strategy::Identity);
  expect(
      proposal.constructed &&
          verify(invalid, proposal.candidate).status == ProofStatus::Unknown &&
          proveRule(invalid).status == ProofStatus::Unknown,
      "successful descriptor copying is not proof of the rule");

  // Mixed explicit/automatic fields are outside ordered admission, but the
  // general strategy must remain available after identity verification fails.
  auto fallback = inspect(R"(
rule compose {(p: transpose (transpose A))=>(q: transpose (transpose A))
 derive {q.attrs={permutation: range(len(p.attrs.permutation))} }}
)");
  proposal = core::attr_synth::synthesize(fallback, Strategy::Identity);
  expect(proposal.constructed && verify(fallback, proposal.candidate).status ==
                                     ProofStatus::Unknown,
         "the identity candidate can fail independent verification");
  expect(!core::attr_synth::synthesize(fallback, Strategy::Ordered).constructed,
         "ordered synthesis reports unsupported admission");
  auto general = core::attr_synth::synthesize(fallback, Strategy::General);
  ProofTiming timing;
  auto timed = proveRule(fallback, 1000000, &timing);
  expect(timed.status == ProofStatus::Proved &&
             timing.construction_attempts == 3 &&
             timing.verification_attempts == 2,
         "optional timing includes failed identity verification and ordered "
         "admission");
  auto no_budget = proveRule(fallback, 0, &timing);
  expect(
      no_budget.status == ProofStatus::Unknown &&
          timing.construction_attempts == 3 &&
          timing.verification_attempts == 0 && timing.verification_ms == 0,
      "timing accounting resets each call and does not bypass proof budgets");
  const auto past = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  auto timed_out = proveRule(fallback, 1000000, &timing, past);
  expect(
      timed_out.status == ProofStatus::Unknown &&
          timed_out.reason.find("time limit exceeded:") != std::string::npos &&
          timing.construction_attempts == 0 &&
          timing.verification_attempts == 0,
      "an expired shared deadline stops synthesis before any strategy");
  auto expired_synthesis =
      core::attr_synth::synthesize(fallback, Strategy::General, 1000000, past);
  auto expired_verification =
      verify(fallback, general.candidate, 1000000, past);
  expect(!expired_synthesis.constructed &&
             expired_synthesis.candidate.constructors.empty() &&
             expired_synthesis.reason.find("time limit exceeded:") !=
                 std::string::npos &&
             expired_verification.status == ProofStatus::Unknown &&
             expired_verification.reason.find("time limit exceeded:") !=
                 std::string::npos,
         "synthesis and independent verification honor context deadlines");
  auto within_deadline =
      proveRule(fallback, 1000000, &timing,
                std::chrono::steady_clock::now() + std::chrono::seconds(10));
  expect(within_deadline.status == ProofStatus::Proved &&
             timing.construction_attempts == 3 &&
             timing.verification_attempts == 2,
         "a future deadline preserves synthesis fallback and verification");
  expect(
      general.constructed &&
          verify(fallback, general.candidate).status == ProofStatus::Proved &&
          proveRule(fallback).status == ProofStatus::Proved,
      "failed identity verification and ordered admission fall back to general "
      "synthesis");
  auto exhausted = core::attr_synth::synthesize(fallback, Strategy::General, 0);
  expect(
      !exhausted.constructed && exhausted.candidate.constructors.empty() &&
          exhausted.reason.find("budget exhausted") != std::string::npos,
      "exhausted synthesis preserves diagnostics without a partial candidate");
}
void explicitConditionIntegrity() {
  rejects("rule r {(d: dot A B)=>(dot A B) where axis_domain() }");
  rejects(
      "rule r {(source: transpose (dot A B))=>(result: dot A B) "
      "where source.axes.order == result.axes.order }");
}

}  // namespace
int main(int argc, char** argv) {
  try {
    std::string error;
    auto files = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    expect(files && argc == 2, "runfiles");
    std::ifstream input(files->Rlocation(argv[1]));
    dialect.assign(std::istreambuf_iterator<char>(input), {});
    expect(!dialect.empty(), "tensor dialect");
    conditionalFeasibility();
    identityAndPracticalDomains();
    mixedAttributes();
    independentVerification();
    synthesisBoundary();
    explicitConditionIntegrity();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
