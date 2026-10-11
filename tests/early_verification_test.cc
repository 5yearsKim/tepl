#include "src/core/semantics/early_verification.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/analyze.h"
#include "src/parse.h"
namespace {
using namespace tepl::core;
using namespace tepl::core::semantics;
void check(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
std::string dialect;
AnalysisResult analyze(const std::string& rule,
                       const std::string& definitions = "") {
  auto parsed = tepl::parse(
      (definitions.empty() ? dialect : definitions) + rule, "early.tepl");
  check(parsed.ok(), "parse early fixture");
  auto result =
      tepl::core::analyze(*parsed.program, {SemanticPolicy::kAlgebraic, 10000});
  for (auto& d : result.diagnostics) std::cerr << d.message << '\n';
  check(result.ok(), "analyze early fixture");
  return result;
}
auto plan(const AnalysisResult& a) {
  return *a.program->rules.at(0).inference->semantic;
}
std::vector<List> fields(const inference::SemanticPlan& p,
                         const std::vector<List>& values) {
  std::vector<List> out(p.known.size());
  for (std::size_t i = 0; i < p.known.size(); ++i)
    out[i] = values.at(p.known[i].field);
  return out;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    check(argc == 2, "dialect argument");
    std::string error;
    auto files = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    check(bool(files), error);
    std::ifstream input(files->Rlocation(argv[1]));
    dialect = {std::istreambuf_iterator<char>(input), {}};
    check(!dialect.empty(), "read dialect");
    auto drop =
        plan(analyze("rule r { (add X Y) => X where { $enabled(X); } }"));
    auto bad = verifySparse(drop, {{2, 2}, {2, 2}}, {});
    check(bad.status == VerificationStatus::Counterexample && bad.degree == 1 &&
              bad.entries.size() == 1,
          "one-entry add counterexample");
    check(bad.lhs_value != bad.rhs_value, "confirmed values differ");
    check(verifySparse(drop, {{0, 2}, {0, 2}}, {}).status ==
              VerificationStatus::NoCounterexample,
          "empty output vacuously equal");
    check(verifySparse(drop, {{2}, {3}}, {}).status ==
              VerificationStatus::Skipped,
          "illegal LHS skipped");
    check(verifySparse(drop, {{2}, {2}}, {}, {}, 1).status ==
              VerificationStatus::Unknown,
          "budget is unknown");
    auto repeated =
        plan(analyze("rule r { (add X X) => X where { $enabled(X); } }"));
    auto alias = verifySparse(repeated, {{2}}, {});
    check(alias.status == VerificationStatus::Counterexample &&
              alias.entries.size() == 1,
          "repeated capture is one tensor");
    auto trans = plan(analyze("rule r { (transpose[@p] X) => X }"));
    check(verifySparse(trans, {{2}}, {{0}}).status ==
              VerificationStatus::NoCounterexample,
          "rank one hides transpose bug");
    auto swap = verifySparse(trans, {{2, 2}}, {{1, 0}});
    check(swap.status == VerificationStatus::Counterexample &&
              swap.entries.size() == 1,
          "rank two exposes transpose bug");
    check(
        verifySparse(trans, {{2, 3}}, {{1, 0}}).reason.find("shapes differ") !=
            std::string::npos,
        "shape failure separate");
    std::vector<std::vector<List>> blocked;
    for (int i = 0; i < 2; ++i) {
      auto sample = sampleMetadata(trans, {{2, 2}}, blocked);
      check(sample.status == Status::Found, "sample legal source permutation");
      check(sample.known[0].size() == 2, "permutation full rank");
      blocked.push_back(sample.known);
    }
    check(blocked[0] != blocked[1], "samples distinct");
    check(sampleMetadata(trans, {{2, 2}}, blocked).status == Status::Invalid,
          "exact blocked domain exhausted");
    auto distribution =
        plan(analyze("rule r { (dot[@d] (add A B) C) => (add (dot[@d] A C) "
                     "(dot[@d] B C)) }"));
    auto known = fields(distribution, {{1}, {0}, {}, {}});
    auto valid = verifySparse(distribution, {{2, 3}, {2, 3}, {3, 2}}, known, {},
                              1000000);
    if (valid.status == VerificationStatus::Unknown)
      std::cerr << valid.reason << '\n';
    check(valid.status == VerificationStatus::NoCounterexample &&
              valid.degree == 2,
          "distribution covers all values at metadata");
    auto sampled = sampleMetadata(distribution, {{2, 2}, {2, 2}, {2, 2}});
    check(sampled.status == Status::Found, "source dot sample");
    for (std::size_t i = 0; i < distribution.known.size(); ++i)
      check(sampled.known[i] == sampled.known[distribution.known[i].field],
            "shared RHS attrs use source sample");
    auto inconsistent = known;
    inconsistent.back() = {0};
    check(verifySparse(distribution, {{2, 3}, {2, 3}, {3, 2}}, inconsistent)
                  .status == VerificationStatus::Unknown,
          "shared captured attributes cannot change on RHS");
    check(verifySparse(drop, {{2}, {2}}, {}, {}, 100000, 0).status ==
              VerificationStatus::Unknown,
          "degree limit applies to original captures");
    auto lora =
        plan(analyze("rule r { (dot[@o] X (add W (dot[@i] A B))) => (add "
                     "(dot[@o] X W) (dot[@o] (dot[@o] X A) B)) }"));
    auto lo = verifySparse(lora, {{2, 3}, {3, 4}, {3, 1}, {1, 4}},
                           fields(lora, {{1}, {0}, {}, {}}), {}, 1000000);
    if (lo.status == VerificationStatus::Unknown)
      std::cerr << "LoRA: " << lo.reason << "\n";
    check(lo.status == VerificationStatus::NoCounterexample && lo.degree == 3,
          "LoRA degree derived through intermediates");
    auto square =
        plan(analyze("rule r { (square X) => X where { $enabled(X); } }", R"(
      dialect D { op square(x:tensor)->tensor { dtype(t){yield t;} shape(s){yield s;}
        semantics(i){yield read(x,i) * read(x,i);} } }
    )"));
    auto sq = verifySparse(square, {{}}, {});
    check(sq.status == VerificationStatus::Counterexample && sq.degree == 2 &&
              sq.entries[0].value == 2,
          "grid needs values beyond one");
    auto constants = plan(
        analyze("rule r { (one X) => (zero X) where { $enabled(X); } }", R"(
      dialect D {
        op one(x:tensor)->tensor { dtype(t){yield t;} shape(s){yield s;} semantics(i){yield 1;} }
        op zero(x:tensor)->tensor { dtype(t){yield t;} shape(s){yield s;} semantics(i){yield 0;} }
      }
    )"));
    auto constant = verifySparse(constants, {{}}, {});
    check(constant.status == VerificationStatus::Counterexample &&
              constant.degree == 0 && constant.entries.empty(),
          "degree-zero all-zero witness");
    auto parsed_bad = tepl::parse(dialect + "rule r { (transpose[@p] X) => X }",
                                  "reject.tepl");
    auto compile_bad = tepl::core::analyze(
        *parsed_bad.program, {SemanticPolicy::kAlgebraic, 1000, true});
    check(!compile_bad.ok() &&
              compile_bad.diagnostics[0].message.find(
                  "early sparse counterexample") != std::string::npos,
          "optional compile-time rejection checks more than scalar rank");
    auto parsed_good = tepl::parse(
        dialect + "rule r { (dot[@d] A B) => (dot[@d] A B) }", "pass.tepl");
    check(tepl::core::analyze(*parsed_good.program,
                              {SemanticPolicy::kAlgebraic, 1000, true})
              .ok(),
          "bounded check accepts correct explicit rule");
    auto automatic =
        plan(analyze("rule r { (transpose[@p] X) => (transpose[auto] X) }"));
    check(verifySparse(automatic, {{2, 2}}, {{1, 0}}).status ==
              VerificationStatus::Unknown,
          "auto not silently synthesized");
    auto request = std::string("sparse-v1 10000 100000 8 ") +
                   encodePlan(square) + "1 0 0 0 ";
    check(verificationRequest(request).starts_with("0 2 "),
          "versioned sparse wire response");
    check(verificationRequest("malformed").empty(),
          "malformed wire is unknown");
    std::cout << "early sparse verification tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
