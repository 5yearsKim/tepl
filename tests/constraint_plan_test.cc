#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/codegen/generate.h"
#include "src/core/analyze.h"
#include "src/core/semantics/solver.h"
#include "src/parse.h"

namespace {
using namespace tepl::core;
using semantics::List;
using semantics::Status;
void check(bool b, const std::string& message) {
  if (!b) throw std::runtime_error(message);
}
std::string dialect;
AnalysisResult analyze(const std::string& rule,
                       const std::string& definitions = "",
                       bool algebraic = true) {
  auto parsed = tepl::parse(
      (definitions.empty() ? dialect : definitions) + rule, "constraints.tepl");
  check(parsed.ok(), "parse constraint fixture");
  auto result = tepl::core::analyze(
      *parsed.program,
      {algebraic ? SemanticPolicy::kAlgebraic : SemanticPolicy::kStrict,
       10000});
  for (auto& d : result.diagnostics) std::cerr << d.message << '\n';
  check(result.ok(), "analyze constraint fixture");
  return result;
}
const inference::SemanticPlan& plan(const AnalysisResult& a) {
  return *a.program->rules.at(0).inference->semantic;
}
auto solve(const inference::SemanticPlan& p, const std::vector<List>& shapes,
           const std::vector<List>& known = {},
           const std::vector<List>& pins = {}) {
  auto result = semantics::solveGeneric(p, shapes, known, pins, 1000000);
  if (result.status == Status::Unknown) std::cerr << result.reason << '\n';
  return result;
}
void compiled(const semantics::Solution& s, const std::string& message) {
  check(s.status == Status::Found && s.method == "constraints-z3", message);
}
std::string request(const inference::SemanticPlan& p,
                    const std::vector<List>& shapes,
                    const std::vector<List>& known) {
  auto out = semantics::encodePlan(p);
  for (auto lists : {shapes, known, std::vector<List>{}}) {
    out += std::to_string(lists.size()) + " ";
    for (auto& l : lists) {
      out += std::to_string(l.size()) + " ";
      for (auto x : l) out += std::to_string(x) + " ";
    }
  }
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 2, "dialect fixture argument");
    std::string error;
    auto runfiles = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    check(bool(runfiles), error);
    std::ifstream input(runfiles->Rlocation(argv[1]));
    dialect = {std::istreambuf_iterator<char>(input), {}};
    check(!dialect.empty(), "read dialect");
    const std::vector<List> matrix{{1}, {0}, {}, {}};
    auto distribution = analyze(
        "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B "
        "C)) }");
    auto& d = plan(distribution);
    check(d.constraints.available && d.constraints.lhs.size() == 2 &&
              d.constraints.rhs.size() == 2,
          "distribute during compilation");
    check(d.constraints.domains.size() == 3,
          "reduction templates are not unrolled");
    for (std::size_t rank : {2, 3, 6}) {
      List a(rank, 2);
      compiled(solve(d, {a, a, {2, 3}}, {{rank - 1}, {0}, {}, {}}),
               "one compiled distribution plan handles several ranks");
    }
    compiled(solve(d, {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 4, 5, 6}},
                   {{2, 3}, {1, 2}, {0}, {0}}),
             "multiple reduction and batch axes");
    auto associate = analyze(
        "rule r { (dot[@outer] (dot[@inner] A B) C) => (dot[auto] A (dot[auto] "
        "B C)) }");
    auto matrices = matrix;
    matrices.insert(matrices.end(), matrix.begin(), matrix.end());
    check(plan(associate).constraints.lhs[0].reductions.size() == 2,
          "nested reductions expanded at compile time");
    compiled(
        solve(plan(associate), {{2, 3}, {3, 4}, {4, 5}}, matrices),
        "reduction matching infers reassociation without ordered constructor");
    auto lora = analyze(
        "rule r { (dot[@outer] X (add W (dot[@inner] A B))) => "
        "(add (dot[auto] X W) (dot[auto] (dot[auto] X A) B)) }");
    check(plan(lora).constraints.lhs.size() == 2,
          "LoRA has two normalized terms");
    compiled(solve(plan(lora), {{2, 3}, {3, 4}, {3, 1}, {1, 4}}, matrices),
             "LoRA attributes inferred jointly including singleton reduction");

    auto renamed = dialect;
    for (auto [from, to] :
         {std::pair{"dot_general", "arbitrary"},
          std::pair{"alias: dot;", "alias: product;"},
          std::pair{"lhs_contracting_dimensions", "left_axes"}})
      for (std::size_t pos = 0;
           (pos = renamed.find(from, pos)) != std::string::npos;
           pos += std::string(to).size())
        renamed.replace(pos, std::string(from).size(), to);
    auto custom = analyze(
        "rule r { (product[@d] (add A B) C) => (add (product[auto] A C) "
        "(product[auto] B C)) }",
        renamed);
    compiled(
        solve(plan(custom), {{2, 3}, {2, 3}, {3, 4}}, matrix),
        "no operation or attribute names participate in constraint generation");

    auto shared = analyze(
        "rule r { (transpose[@q] X) => (transpose[auto] (transpose[auto] X)) "
        "}");
    auto square = plan(shared);
    // Share the two descriptor fields in the runtime IR to exercise a coupled
    // unknown independently of surface syntax for naming automatic descriptors.
    square.nodes.back().fields[0].output =
        square.nodes[square.source_end].fields[0].output;
    auto root = solve(square, {{2, 2, 2, 2}}, {{1, 0, 3, 2}});
    compiled(root, "shared unknown permutation square root");
    auto p = root.fields.at(square.nodes.back().fields[0].output);
    for (std::size_t i = 0; i < p.size(); ++i)
      check(p[p[i]] == List{1, 0, 3, 2}[i],
            "returned permutation squares to source");
    check(solve(square, {{2, 2}}, {{1, 0}}).status == Status::Invalid,
          "no shared permutation root after complete fallback");

    auto unit =
        analyze("rule r { (unit_sum X) => (transpose[auto] X) }", dialect + R"(
      dialect U {
        op unit_sum(x:tensor)->tensor {
          dtype(t){yield t;} shape(s){yield s;}
          semantics(i){yield sum(k in index_space([1])) { read(x,i) };}
        }
      }
      use U::{unit_sum};
    )");
    check(plan(unit).constraints.available,
          "unused reduction still has a sufficient plan");
    compiled(solve(plan(unit), {{2, 3}}),
             "unused singleton reductions normalize before matching");

    auto repeated =
        analyze("rule r { (dot[@d] A A) => (transpose[auto] (dot[@d] A A)) }");
    check(plan(repeated).constraints.available,
          "repeated factors use occurrence-bijection constraints");
    auto strict = analyze(
        "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B "
        "C)) }",
        "", false);
    check(!plan(strict).constraints.available,
          "no algebraic plan for strict arithmetic");
    check(solve(d, {{2, 3}, {2, 3}, {3, 4}}, matrix).fields.size() > 0,
          "attributes exported");
    auto wire = request(plan(lora), {{2, 3}, {3, 4}, {3, 1}, {1, 4}}, matrices);
    auto answer = semantics::solveRequest(wire);
    check(answer.starts_with("0 ") &&
              answer.find("constraints-z3") != std::string::npos,
          "serialized plan executes compiled constraints");
    check(semantics::solveRequest("0 0 0 0 0 0") == "2 0 ",
          "old/malformed plan rejected");
    auto generated =
        tepl::codegen::generate(*lora.program, {tepl::codegen::Target::kRust});
    check(generated.ok(), "Rust generation");
    std::string output;
    for (auto& f : generated.files) output += f.contents;
    check(
        output.find("const AUTO_CONSTRAINT_PLAN: &str") != std::string::npos &&
            output.find("tepl-constraints-2") != std::string::npos,
        "Rust embeds the compile-time constraint plan");
    std::cout << "Compiled semantic constraint tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
