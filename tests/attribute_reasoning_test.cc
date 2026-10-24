#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/pipeline.h"
#include "src/core/semantics/proof/verify.h"
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
      "unproved replacement is rejected at compile time");
}
std::string suggest(const std::string& text) {
  auto parsed = parseRule(text);
  auto inspected = core::inspect(
      *parsed.program, {core::SemanticPolicy::kAlgebraic, true, {}, true});
  expect(inspected.ok(), "suggestion search preserves an inspection result");
  const auto& plan = *inspected.program->rules.at(0).semantic_plan;
  expect(plan.proof.status == core::semantics::ProofStatus::Unknown,
         "a suggestion never accepts the original rule");
  return plan.suggested_rule;
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
 where {d.axes.B.positions == []}}
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
graph g { input A:f32[EXTENT,3];
 let x=(transpose[permutation=[1,0]] A);
 yield (transpose[permutation=[0,1]] x); }
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
 derive { q.attrs=p.attrs; }
}
)");
  expect(!p.ir().rules[0].inference && p.ir().rules[0].derivations.size() == 2,
         "explicit and synthesized RHS descriptors coexist");
  rejects(R"(
rule invalid {
 (p: transpose A) => (q: transpose A)
 derive {q.attrs={permutation: []};}
}
)");
  rejects("rule unequal {(dot A B)=>(dot B A)}");
  rejects(R"(
rule unsafe {(p: transpose A)=>A
 where {subsequence(A.axes.order,concat(A.axes.order,A.axes.order))}}
)");
}
void recommendations() {
  const std::string association = R"(
rule associate {(d: dot (inner: dot A B) C)=>(dot A (dot B C))}
)";
  rejects(association);
  auto text = suggest(association);
  expect(!text.empty(), "find a verified nonvacuous reassociation restriction");
  expect(text.find("inner.axes.B.positions") != std::string::npos &&
             text.find("axis_domain") == std::string::npos,
         "suggest source group relationships without an extent restriction");
  auto p = compile(text);
  expect(p.ir().rules[0].conditions.size() == 1 && !p.ir().rules[0].inference,
         "one written condition suffices; no hidden additional filtering");
  auto loaded = executable::load(executable::serialize(p));
  expect(loaded.ir().rules[0].conditions.size() == 1,
         "serialized rules retain exactly the declared applicability");

  auto trans = suggest(R"(
rule remove {(p: transpose (dot A B))=>(dot A B)
 where {len(p.attrs.permutation) == 2}}
)");
  expect(!trans.empty() &&
             trans.find("range(len(p.attrs.permutation))") != std::string::npos,
         "discover an identity-permutation sufficient restriction");
  expect(trans.find("len(p.attrs.permutation) == 2") != std::string::npos,
         "recommendations preserve user restrictions");
  compile(trans);
  expect(suggest(R"(
rule impossible {(p: transpose (dot A B))=>(dot A B)
 where {p.attrs.permutation == [] && p.attrs.permutation == [0]}}
)")
             .empty(),
         "never recommend a condition without a concrete legal witness");
}
void independentVerification() {
  using namespace core::semantics;
  auto parsed = parseRule("rule identity {(dot A B)=>(dot A B)}");
  auto inspected =
      core::inspect(*parsed.program, {core::SemanticPolicy::kAlgebraic, true});
  expect(inspected.ok(), "inspect identity proof");
  auto plan = *inspected.program->rules[0].semantic_plan;
  auto candidate = static_cast<ConstructorCandidate>(plan.proof);
  candidate.constructors.begin()->second[0] = make(Symbol::Kind::List);
  expect(verify(plan, candidate).status == ProofStatus::Unknown,
         "fresh verification rejects a corrupted constructor");
  auto unguarded =
      parseRule("rule r {(d: dot (inner: dot A B) C)=>(dot A (dot B C))}");
  auto info = core::inspect(*unguarded.program,
                            {core::SemanticPolicy::kAlgebraic, true});
  expect(info.ok() &&
             !info.program->rules[0].semantic_plan->proof.guard_origins.empty(),
         "failed verification preserves unresolved-obligation provenance");
}
void explicitConditionIntegrity() {
  rejects("rule r {(d: dot A B)=>(dot A B) where {axis_domain()}}");
  rejects(
      "rule r {(source: transpose (dot A B))=>(result: dot A B) "
      "where {source.axes.order == result.axes.order}}");
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
    identityAndPracticalDomains();
    mixedAttributes();
    recommendations();
    independentVerification();
    explicitConditionIntegrity();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
