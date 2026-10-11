#include "src/core/semantics/ordered.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/analyze.h"
#include "src/parse.h"

namespace {
using namespace tepl::core::semantics;
void check(bool good, const std::string& message) {
  if (!good) throw std::runtime_error(message);
}
std::string dialect;
auto plan(const std::string& rule, const std::string& definitions = "") {
  auto parsed = tepl::parse(
      (definitions.empty() ? dialect : definitions) + rule, "ordered.tepl");
  check(parsed.ok(), "parse ordered fixture");
  auto analyzed = tepl::core::analyze(
      *parsed.program, {tepl::core::SemanticPolicy::kAlgebraic, 10000});
  for (auto& d : analyzed.diagnostics) std::cerr << d.message << '\n';
  check(analyzed.ok(), "analyze ordered fixture");
  return *analyzed.program->rules.at(0).inference->semantic;
}
std::string replace(std::string text, const std::string& from,
                    const std::string& to) {
  std::size_t position = 0;
  while ((position = text.find(from, position)) != std::string::npos) {
    text.replace(position, from.size(), to);
    position += to.size();
  }
  return text;
}
void found(const tepl::core::inference::SemanticPlan& p,
           const std::vector<List>& shapes, const std::vector<List>& known) {
  auto result = solve(p, shapes, known);
  check(result.status == Status::Found, result.reason);
  check(result.method == "ordered", "must avoid SMT candidate search");
  check(checkOrderedWitness(p, shapes, known, result.fields),
        "independently check exact port partition and output order");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 2, "dialect fixture argument");
    std::string error;
    auto files = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    check(bool(files), error);
    std::ifstream input(files->Rlocation(argv[1]));
    dialect = {std::istreambuf_iterator<char>(input), {}};
    const auto associate =
        "rule r { (dot[@outer] (dot[@inner] A B) C) "
        "=> (dot[auto] A (dot[auto] B C)) }";
    auto p = plan(associate);
    std::size_t certified = 0;
    for (auto& n : p.nodes)
      if (n.capture < 0) {
        auto c = extractOrderedContract(n);
        check(bool(c), "extract all contraction contracts from primitive IR");
        check(c->left_contract == 0 && c->right_contract == 1 &&
                  c->left_batch == 2 && c->right_batch == 3,
              "derive the field-to-coordinate relationship");
        ++certified;
      }
    check(certified == 4, "four contractions");
    const std::vector<List> matrices{{1}, {0}, {}, {}, {1}, {0}, {}, {}};
    found(p, {{2, 3}, {3, 4}, {4, 5}}, matrices);
    found(p, {{2, 3, 4}, {2, 4, 5}, {2, 5, 6}},
          {{2}, {1}, {0}, {0}, {2}, {1}, {0}, {0}});
    found(p, {{2}, {2}, {2}}, {{}, {}, {0}, {0}, {0}, {0}, {}, {}});
    found(p, {{}, {}, {}}, {{}, {}, {}, {}, {}, {}, {}, {}});

    // Nonuniform physical ordering and 98 batch axes. Source-field copying
    // cannot supply the RHS attributes; a symbolic list search would exceed
    // the generic engine's allocation cap.
    List a(98, 2), b(99, 2), c(99, 2), batches(98);
    std::iota(batches.begin(), batches.end(), 0);
    a.insert(a.end(), {3, 4});
    b[0] = 5;
    b.push_back(4);
    c[0] = 5;
    c.push_back(6);
    List shifted = batches;
    for (auto& axis : shifted) ++axis;
    std::vector<List> large_known{{99}, {99}, batches, shifted,
                                  {99}, {0},  batches, shifted};
    p.timeout_ms = 1000;
    found(p, {a, b, c}, large_known);
    auto large = solve(p, {a, b, c}, large_known);
    auto wrong = large.fields;
    for (auto& n : p.nodes)
      if (n.capture < 0 && n.fields[0].known == -1) {
        wrong[n.fields[0].output] = {};
        break;
      }
    check(!checkOrderedWitness(p, {a, b, c}, large_known, wrong),
          "reject a tampered witness");
    auto exhausted = solve(p, {a, b, c}, large_known, {}, 1);
    check(exhausted.status == Status::Unknown && exhausted.method == "ordered",
          "resource exhaustion is inconclusive");

    // A[i] B[j] C[j] must output [j,i], but the RHS forces [i,j].
    const std::vector<List> obstruction_known{{}, {}, {}, {}, {}, {}, {1}, {0}};
    auto impossible = solve(p, {{2}, {2}, {2}}, obstruction_known);
    check(impossible.status == Status::Invalid &&
              impossible.method == "ordered" &&
              impossible.reason.find("two-axis") != std::string::npos,
          "reject all attributes with a small obstruction");
    OrderedObstruction cert{p.rhs_root, 1, 0};
    check(checkOrderedObstruction(p, {{2}, {2}, {2}}, obstruction_known, cert),
          "independent obstruction check");
    std::swap(cert.first, cert.second);
    check(!checkOrderedObstruction(p, {{2}, {2}, {2}}, obstruction_known, cert),
          "reject a reversed certificate");
    auto repair = plan(
        "rule r { (dot[@outer] (dot[@inner] A B) C) "
        "=> (transpose[auto] (dot[auto] A (dot[auto] B C))) }");
    found(repair, {{2}, {2}, {2}}, obstruction_known);
    auto leaf = plan("rule r { (transpose[@p] A) => A }");
    auto leaf_result = solve(leaf, {{2, 2}}, {{1, 0}});
    check(leaf_result.status == Status::Invalid &&
              leaf_result.method == "ordered",
          "physical leaf order obstruction");
    check(checkOrderedObstruction(leaf, {{2, 2}}, {{1, 0}},
                                  {leaf.rhs_root, 1, 0}),
          "check leaf inversion certificate");

    // Every legal source pairing/batch/contract choice for three one-axis
    // inputs is checked against the independent SMT/polynomial engine.
    auto tiny = p;
    tiny.timeout_ms = 10000;
    for (int inner = 0; inner < 3; ++inner) {
      std::vector<List> source(8);
      if (inner == 1) source[2] = source[3] = {0};
      if (inner == 2) source[0] = source[1] = {0};
      const int rank = inner == 0 ? 2 : inner == 1 ? 1 : 0;
      for (int pairing = -1; pairing < rank; ++pairing)
        for (int batch = 0; batch < (pairing < 0 ? 1 : 2); ++batch) {
          auto attributes = source;
          if (pairing >= 0) {
            attributes[batch ? 6 : 4] = {std::uint64_t(pairing)};
            attributes[batch ? 7 : 5] = {0};
          }
          auto structural = solve(tiny, {{2}, {2}, {2}}, attributes);
          auto baseline = tiny;
          baseline.constraints = {};
          auto generic =
              solveGeneric(baseline, {{2}, {2}, {2}}, attributes, {}, 1000000);
          check(structural.method == "ordered" && generic.method == "z3",
                "distinct structural and generic reasoning paths");
          check(structural.status == generic.status &&
                    generic.status != Status::Unknown,
                "exhaustive tiny-source differential agreement");
        }
    }

    // Renamed fields, commuted products, reversed assert operands, and
    // changed scatter group order normalize to the same routing contract.
    auto changed = replace(dialect, "dot_general", "join");
    changed = replace(changed, "alias: dot;", "alias: arbitrary;");
    changed = replace(changed, "lhs_contracting_dimensions", "left_sum");
    changed = replace(changed, "rhs_contracting_dimensions", "right_sum");
    changed = replace(changed, "read(lhs, li) * read(rhs, ri)",
                      "read(rhs, ri) * read(lhs, li)");
    changed = replace(changed, "len(lb) == len(rb)", "len(rb) == len(lb)");
    changed = replace(changed, "concat(lb, lf, lc)", "concat(lc, lb, lf)");
    changed = replace(changed, "concat(b, i, k)", "concat(k, b, i)");
    auto generic = plan(replace(associate, "dot", "arbitrary"), changed);
    found(generic, {{2, 3}, {3, 4}, {4, 5}}, matrices);
    auto swapped = replace(dialect, "op dot_general(lhs: tensor, rhs: tensor)",
                           "op dot_general(rhs: tensor, lhs: tensor)");
    swapped = replace(swapped, "shape(l, r)", "shape(r, l)");
    auto swapped_plan = plan(associate, swapped);
    for (auto& n : swapped_plan.nodes)
      if (n.capture < 0)
        check(extractOrderedContract(n)->left == 1,
              "operand roles are extracted, not assumed");
    found(swapped_plan, {{3, 2}, {4, 3}, {5, 4}}, matrices);

    // Reordered schema fields must change the extracted mappings, not behavior.
    auto reordered =
        replace(dialect,
                "lhs_contracting_dimensions: index[];\n            "
                "rhs_contracting_dimensions: index[];\n            "
                "lhs_batching_dimensions: index[];\n            "
                "rhs_batching_dimensions: index[];",
                "rhs_batching_dimensions: index[];\n            "
                "lhs_batching_dimensions: index[];\n            "
                "rhs_contracting_dimensions: index[];\n            "
                "lhs_contracting_dimensions: index[];");
    auto remapped = plan(associate, reordered);
    found(remapped, {{2, 3}, {3, 4}, {4, 5}},
          {{}, {}, {0}, {1}, {}, {}, {0}, {1}});

    // An extra assertion restricts the operation. It is not admitted simply
    // because its computation resembles a contraction.
    auto restricted =
        replace(dialect, "assert len(lc) == len(rc);",
                "assert len(lc) == len(rc); assert len(lc) == 1;");
    auto constrained = plan(associate, restricted);
    for (auto& n : constrained.nodes)
      if (n.capture < 0)
        check(!extractOrderedContract(n),
              "extra legality restriction excluded");
    check(solve(constrained, {{2, 3}, {3, 4}, {4, 5}}, matrices).method == "constraints-z3",
          "restricted contracts use generic solver");
    auto scaled = replace(dialect, "read(lhs, li) * read(rhs, ri)",
                          "2 * read(lhs, li) * read(rhs, ri)");
    auto changed_scalar = plan(associate, scaled);
    for (auto& n : changed_scalar.nodes)
      if (n.capture < 0)
        check(!extractOrderedContract(n), "scalar factor excluded");
    auto missing = p.nodes[p.source_root];
    auto assertion = std::find_if(
        missing.assertions.begin(), missing.assertions.end(),
        [&](auto id) { return missing.expressions[id].op == "is_disjoint"; });
    missing.assertions.erase(assertion);
    check(!extractOrderedContract(missing),
          "missing legality assertion excluded");
    auto partial =
        replace(dialect, "shape(l, r) {", "shape(l, r) { let unused = 1 / 0;");
    auto partial_plan = plan(associate, partial);
    for (auto& n : partial_plan.nodes)
      if (n.capture < 0)
        check(!extractOrderedContract(n), "unused partial let is not ignored");
    check(solve(partial_plan, {{2, 3}, {3, 4}, {4, 5}}, matrices).status ==
              Status::Unknown,
          "illegal source due to unused shape computation declines");
    auto invalid_source = solve(p, {{2, 3}, {3, 4}, {4, 5}},
                                {{7}, {0}, {}, {}, {1}, {0}, {}, {}});
    check(invalid_source.status == Status::Unknown,
          "illegal source is not an impossibility certificate");

    auto pinned = plan(
        "rule r { (dot[@d] A B) => "
        "(dot[auto] (transpose[auto,permutation=[1,0]] A) B) }");
    check(!solveOrdered(pinned, {{2, 3}, {3, 4}}, {{1}, {0}, {}, {}}, {{1, 0}}),
          "pins outside theorem");
    auto repeated = plan("rule r { (dot[@d] A A) => (dot[auto] A A) }");
    check(!solveOrdered(repeated, {{2}}, {{}, {}, {}, {}}, {}),
          "repeated inputs outside theorem");
    auto shared = plan("rule r { (dot[@d] A B) => (dot[@d] A B) }");
    check(!solveOrdered(shared, {{2, 3}, {3, 4}}, {{1}, {0}, {}, {}}, {}),
          "known RHS attributes outside theorem");
    check(!solveOrdered(p, {{1, 3}, {3, 4}, {4, 5}}, matrices, {}),
          "singleton dimensions outside theorem");
    check(!solveOrdered(p, {{0, 3}, {3, 4}, {4, 5}}, matrices, {}),
          "empty dimensions outside theorem");
    auto strict = p;
    strict.algebraic = false;
    check(!solveOrdered(strict, {{2, 3}, {3, 4}, {4, 5}}, matrices, {}),
          "strict arithmetic outside theorem");

    auto wire = encodePlan(p);
    wire += "3 2 2 3 2 3 4 2 4 5 8 1 1 1 0 0 0 1 1 1 0 0 0 0 ";
    check(solveRequest(wire).find(" ordered ") != std::string::npos,
          "native ABI exposes structural method");
    std::cout << "Ordered contraction tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
