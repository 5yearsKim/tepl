#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/analyze.h"
#include "src/core/index_semantics/ir.h"
#include "src/core/index_semantics/synthesize.h"
#include "src/core/index_semantics/verify.h"
#include "src/core/print.h"
#include "src/parse.h"
#include "tests/index_constructor_eval.h"
#include "tests/index_reference.h"

namespace {
using namespace tepl::core::index_semantics;
std::string dialect;
void expect(bool b, const std::string& s) {
  if (!b) throw std::runtime_error(s);
}
tepl::core::AnalysisResult analyze(const std::string& rule,
                                   bool algebraic = true,
                                   std::string definitions = {}) {
  auto parsed = tepl::parse(
      (definitions.empty() ? dialect : definitions) + rule, "index.tepl");
  for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  expect(parsed.ok(), "fixture parses");
  return tepl::core::analyze(*parsed.program,
                             {algebraic ? tepl::core::SemanticPolicy::kAlgebraic
                                        : tepl::core::SemanticPolicy::kStrict,
                              true});
}
Result run(const std::string& rule, Metadata m, Limits limits = {},
           std::string definitions = {}) {
  auto checked = analyze(rule, true, definitions);
  for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  expect(checked.ok(), "fixture checks");
  expect(bool(checked.program->rules[0].semantic_plan), "new inference plan");
  expect(!checked.program->rules[0].inference, "legacy inference bypassed");
  return infer(*checked.program->rules[0].semantic_plan, m, limits);
}
void found(const Result& r) {
  if (r.status != Status::Found) {
    for (const auto& t : r.trace) std::cerr << t << '\n';
    std::cerr << r.reason << '\n';
  }
  expect(r.status == Status::Found, "expected a checked witness");
}
void error(const std::string& definitions, const std::string& message) {
  auto result = analyze("", true, definitions);
  expect(!result.ok(), "invalid definition rejected");
  for (const auto& d : result.diagnostics)
    if (d.message.find(message) != std::string::npos) return;
  throw std::runtime_error("missing diagnostic: " + message);
}
std::string replace(std::string s, const std::string& a, const std::string& b) {
  for (std::size_t p = 0; (p = s.find(a, p)) != std::string::npos;
       p += b.size())
    s.replace(p, a.size(), b);
  return s;
}

// An independent dense reference evaluator uses physical dimension lists, not
// logical-axis identities or the inference engine's normalization routines.
struct Dense {
  List shape, values;
  std::int64_t at(const List& index) const {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < shape.size(); ++i)
      offset = offset * shape[i] + index[i];
    return values.at(offset);
  }
};
void indices(const List& shape, const std::function<void(const List&)>& f) {
  List index(shape.size());
  std::function<void(std::size_t)> loop = [&](std::size_t axis) {
    if (axis == shape.size()) {
      f(index);
      return;
    }
    for (index[axis] = 0; index[axis] < shape[axis]; ++index[axis])
      loop(axis + 1);
  };
  loop(0);
}
Dense data(List shape, int seed) {
  Dense d{shape, {}};
  indices(shape, [&](const auto&) {
    d.values.push_back((std::int64_t(d.values.size()) * 3 + seed) % 7 - 3);
  });
  return d;
}
List remaining(std::size_t rank, const List& a, const List& b = {}) {
  List result;
  for (std::size_t i = 0; i < rank; ++i)
    if (std::find(a.begin(), a.end(), i) == a.end() &&
        std::find(b.begin(), b.end(), i) == b.end())
      result.push_back(i);
  return result;
}
Dense add(const Dense& a, const Dense& b) {
  expect(a.shape == b.shape, "reference add shapes");
  auto out = a;
  for (std::size_t i = 0; i < out.values.size(); ++i)
    out.values[i] += b.values[i];
  return out;
}
Dense multiply(const Dense& a, const Dense& b) {
  expect(a.shape == b.shape, "reference multiply shapes");
  auto out = a;
  for (std::size_t i = 0; i < out.values.size(); ++i)
    out.values[i] *= b.values[i];
  return out;
}
Dense reduceSum(const Dense& input, const List& reduced) {
  auto retained = remaining(input.shape.size(), reduced);
  Dense output;
  List domains;
  for (auto axis : retained) output.shape.push_back(input.shape[axis]);
  for (auto axis : reduced) domains.push_back(input.shape[axis]);
  indices(output.shape, [&](const List& out) {
    List in(input.shape.size());
    for (std::size_t i = 0; i < retained.size(); ++i) in[retained[i]] = out[i];
    std::int64_t value = 0;
    indices(domains, [&](const List& k) {
      for (std::size_t i = 0; i < reduced.size(); ++i) in[reduced[i]] = k[i];
      value += input.at(in);
    });
    output.values.push_back(value);
  });
  return output;
}
Dense dot(const Dense& a, const Dense& b, const std::vector<List>& fields) {
  const auto& lc = fields[0];
  const auto& rc = fields[1];
  const auto& lb = fields[2];
  const auto& rb = fields[3];
  auto lf = remaining(a.shape.size(), lb, lc),
       rf = remaining(b.shape.size(), rb, rc);
  Dense out;
  for (auto axis : lb) out.shape.push_back(a.shape[axis]);
  for (auto axis : lf) out.shape.push_back(a.shape[axis]);
  for (auto axis : rf) out.shape.push_back(b.shape[axis]);
  List domains;
  for (auto axis : lc) domains.push_back(a.shape[axis]);
  indices(out.shape, [&](const List& o) {
    List ai(a.shape.size()), bi(b.shape.size());
    std::size_t cursor = 0;
    for (std::size_t t = 0; t < lb.size(); ++t)
      ai[lb[t]] = bi[rb[t]] = o[cursor++];
    for (auto axis : lf) ai[axis] = o[cursor++];
    for (auto axis : rf) bi[axis] = o[cursor++];
    std::int64_t sum = 0;
    indices(domains, [&](const List& k) {
      for (std::size_t t = 0; t < lc.size(); ++t) ai[lc[t]] = bi[rc[t]] = k[t];
      sum += a.at(ai) * b.at(bi);
    });
    out.values.push_back(sum);
  });
  return out;
}
Dense transpose(const Dense& in, const List& p) {
  Dense out;
  for (auto axis : p) out.shape.push_back(in.shape[axis]);
  indices(out.shape, [&](const List& o) {
    List index(in.shape.size());
    for (std::size_t i = 0; i < p.size(); ++i) index[p[i]] = o[i];
    out.values.push_back(in.at(index));
  });
  return out;
}
void equal(const Dense& a, const Dense& b) {
  expect(a.shape == b.shape && a.values == b.values,
         "independent dense equivalence");
}
Plan symbolicPlan(const std::string& rule, bool algebraic = true,
                  const std::string& definitions = {}) {
  auto checked = analyze(rule, algebraic, definitions);
  if (!checked.ok())
    for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  expect(checked.ok(), "symbolic fixture checks without metadata");
  return *checked.program->rules.at(0).semantic_plan;
}
void proved(const Plan& p) {
  if (p.proof.status != ProofStatus::Proved) {
    std::cerr << p.proof.reason << '\n';
    for (const auto& g : p.proof.guard) std::cerr << format(*g) << '\n';
  }
  expect(p.proof.status == ProofStatus::Proved,
         "expected proof under declared conditions");
  expect(p.proof.guard.empty(), "no extra metadata restrictions");
}
// Test-only specialization of the constructor expressions, independent of the
// concrete inference engine. The compiler proof itself never calls this.
List evaluate(const Symbol& s, const Metadata& m) {
  if (s.kind == Symbol::Kind::Field)
    return m.descriptors.at(s.owner).at(s.index);
  if (s.kind == Symbol::Kind::Shape) return m.shapes.at(s.owner);
  if (s.kind == Symbol::Kind::List) {
    List out;
    for (const auto& a : s.args) out.push_back(std::stoll(a->name));
    return out;
  }
  if (s.kind == Symbol::Kind::Call && s.name == "gather") {
    auto source = evaluate(*s.args[0], m), positions = evaluate(*s.args[1], m);
    List out;
    for (auto p : positions) out.push_back(source.at(p));
    return out;
  }
  throw std::runtime_error("unsupported test constructor expression");
}
void symbolicTests() {
  const std::string distribute =
      "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B C)) "
      "}";
  auto p = symbolicPlan(distribute);
  proved(p);
  expect(p.proof.constructors.size() == 2, "two symbolic descriptors");
  for (const auto& [id, fields] : p.proof.constructors) {
    expect(fields.size() == 4, "all symbolic dot fields");
    for (std::size_t f = 0; f < fields.size(); ++f)
      expect(fields[f]->kind == Symbol::Kind::Field && fields[f]->owner == 0 &&
                 fields[f]->index == f,
             "distribution derives source field references");
  }
  expect(p.proof.lhs_normalized.find("Reduce(") != std::string::npos &&
             p.proof.lhs_normalized.find(" + ") != std::string::npos,
         "visible symbolic algebraic expansion");
  for (const std::vector<List>& fields :
       {std::vector<List>{{1}, {0}, {}, {}},
        std::vector<List>{{1, 3}, {2, 0}, {2}, {3}},
        std::vector<List>{{}, {}, {}, {}}}) {
    Metadata m{{}, {{0, fields}}};
    for (const auto& [id, constructors] : p.proof.constructors)
      for (std::size_t f = 0; f < constructors.size(); ++f)
        expect(evaluate(*constructors[f], m) == fields[f],
               "symbolic specialization");
  }
  auto corrupted = p.proof;
  corrupted.constructors.begin()->second[0] =
      std::make_shared<Symbol>(Symbol{Symbol::Kind::List, {}, 0, 0, {}});
  expect(verify(p, corrupted).status == ProofStatus::Unknown,
         "independent verifier rejects corrupted constructor");
  auto dangling = p.proof;
  dangling.constructors.begin()->second[0] = std::make_shared<Symbol>(
      Symbol{Symbol::Kind::Field, "missing", 999, 0, {}});
  expect(verify(p, dangling).status == ProofStatus::Unknown,
         "constructor must depend only on existing source metadata");
  expect(synthesize(p, 1).status == ProofStatus::Unknown,
         "symbolic work exhaustion is unknown");
  auto redundant = p.proof;
  auto numeric = call("is_numeric", {make(Symbol::Kind::DType, {}, 0)});
  redundant.guard.push_back(numeric);
  redundant.guard_origins.push_back({numeric, GuardKind::Legality});
  auto simplified = verify(p, redundant);
  expect(simplified.status == ProofStatus::Proved && simplified.guard.empty(),
         "source legality discharges an injected redundant guard");
  expect(symbolicPlan(distribute, false).proof.status == ProofStatus::Unknown,
         "strict policy prevents arithmetic expansion");

  proved(symbolicPlan("rule r {(add A (add B C)) => (add (add A B) C)}"));
  auto commutative = symbolicPlan("rule r {(add A B) => (add B A)}");
  proved(commutative);
  expect(
      std::find(commutative.proof.steps.begin(), commutative.proof.steps.end(),
                "law: commutative-matching") != commutative.proof.steps.end(),
      "commutative correspondence records its algebraic law");
  expect(symbolicPlan("rule r {(add A B) => A}").proof.status ==
             ProofStatus::Unknown,
         "false algebraic rewrite is not proved");
  auto sum = symbolicPlan(
      "rule r {(reduce_sum[@d] (add A B)) => (add (reduce_sum[auto] A) "
      "(reduce_sum[auto] B))}");
  proved(sum);
  for (const auto& [id, fields] : sum.proof.constructors)
    expect(fields[0]->kind == Symbol::Kind::Field && fields[0]->owner == 0,
           "symbolic reduction dimensions copied");
  auto nonlinear = replace(dialect, "yield Reduce(K, input);",
                           "yield Reduce(K, Product(input, input));");
  expect(symbolicPlan("rule r {(reduce_sum[@d] (add A B)) => (add "
                      "(reduce_sum[auto] A) (reduce_sum[auto] B))}",
                      true, nonlinear)
                 .proof.status == ProofStatus::Unknown,
         "linearity follows semantics, not operation name");

  auto composition = symbolicPlan(
      "rule r {(transpose[@outer] (transpose[@inner] A)) => (transpose[auto] "
      "A)}",
      false);
  proved(composition);
  auto constructor = composition.proof.constructors.at(2)[0];
  expect(format(*constructor) == "gather(@#1.permutation, @#0.permutation)",
         "permutation composition orientation");
  for (auto rank : {0, 1, 3, 10}) {
    List inner(rank), outer(rank);
    std::iota(inner.begin(), inner.end(), 0);
    std::iota(outer.begin(), outer.end(), 0);
    std::reverse(inner.begin(), inner.end());
    if (rank > 1) std::rotate(outer.begin(), outer.begin() + 1, outer.end());
    auto result = evaluate(*constructor, {{{}}, {{0, {outer}}, {1, {inner}}}});
    for (std::size_t a = 0; a < result.size(); ++a)
      expect(result[a] == inner[outer[a]],
             "arbitrary rank permutation constructor");
  }
  auto wrongPermutation = composition.proof;
  wrongPermutation.constructors.at(2)[0] = constructor->args[0];
  expect(verify(composition, wrongPermutation).status == ProofStatus::Unknown,
         "wrong permutation order fails independent verification");
  auto editedCopy = composition.proof;
  auto edited = std::make_shared<Symbol>(*constructor);
  edited->name = "inverse";
  editedCopy.constructors.at(2)[0] = edited;
  expect(verify(composition, editedCopy).status == ProofStatus::Unknown,
         "copied-and-edited IR cannot spoof structural cache");
  auto unguardedPin = symbolicPlan(
      "rule r {(reduce_sum[@d] A) => (reduce_sum[auto, dimensions=[]] A)}");
  expect(unguardedPin.proof.status == ProofStatus::Unknown,
         "unrestricted pin rejected");
  auto pinned = symbolicPlan(
      "rule r {(reduce_sum[@d] A) => (reduce_sum[auto, dimensions=[]] A) where "
      "{ @d.dimensions == range(0); }}");
  proved(pinned);
  auto withoutCondition = pinned;
  withoutCondition.assumptions.clear();
  auto forged = pinned.proof;
  forged.guard = pinned.assumptions;
  expect(verify(withoutCondition, forged).status == ProofStatus::Unknown,
         "candidate guards cannot supply a missing user condition");
  expect(symbolicPlan("rule r {(dot[@d] (dot[@inner] A B) C) => (dot[auto] A "
                      "(dot[auto] B C))}")
                 .proof.status == ProofStatus::Unknown,
         "unguarded reassociation rejected");
  auto association = symbolicPlan(
      "rule r {(dot[@d] (dot[@inner] A B) C) => (dot[auto] A (dot[auto] B "
      "C)) where { @d.lhs_batching_dimensions == @inner->B; "
      "@d.lhs_contracting_dimensions == @inner->J; }}");
  expect(association.proof.status == ProofStatus::Proved,
         "reassociation has a checked routing guard");
  Metadata matrix{{{2, 3}, {3, 4}, {4, 5}},
                  {{0, {{1}, {0}, {}, {}}}, {1, {{1}, {0}, {}, {}}}}};
  tepl::test::ConstructorEvaluator associationEval(matrix, association);
  expect(associationEval.applicable(),
         "ordinary matrices satisfy declared reassociation conditions");
  auto attrs = associationEval.construct();
  auto aa = data({2, 3}, 1), ab = data({3, 4}, 2), ac = data({4, 5}, 3);
  equal(
      dot(dot(aa, ab, matrix.descriptors.at(1)), ac, matrix.descriptors.at(0)),
      dot(aa, dot(ab, ac, attrs.at(3)), attrs.at(2)));
  expect(verify(association, association.proof).status == ProofStatus::Proved,
         "reassociation certificate independently verifies");
  for (bool zero : {false, true}) {
    Metadata multi{
        {{2, 2, 3, 2}, {2, zero ? 0 : 4, 2, 2}, {2, 3, zero ? 0 : 4}},
        {{0, {{2, 3}, {2, 0}, {}, {}}}, {1, {{1, 3}, {2, 0}, {}, {}}}}};
    tepl::test::ConstructorEvaluator eval(multi, association);
    expect(eval.applicable(),
           "multiple-axis and zero-extent reassociation guard");
    auto constructed = eval.construct();
    auto ma = data(multi.shapes[0], 1), mb = data(multi.shapes[1], 2),
         mc = data(multi.shapes[2], 3);
    equal(
        dot(dot(ma, mb, multi.descriptors.at(1)), mc, multi.descriptors.at(0)),
        dot(ma, dot(mb, mc, constructed.at(3)), constructed.at(2)));
  }
  auto reverse = symbolicPlan(
      "rule r {(dot[@outer] A (dot[@inner] B C)) => (dot[auto] (dot[auto] A B) "
      "C) where { @outer.rhs_batching_dimensions == @inner->B; "
      "@outer.rhs_contracting_dimensions == @inner->I; }}");
  expect(reverse.proof.status == ProofStatus::Proved,
         "reverse association supported with guard");
  tepl::test::ConstructorEvaluator reverseEval(matrix, reverse);
  expect(reverseEval.applicable(), "matrix reverse association guard");
  auto reverseAttrs = reverseEval.construct();
  equal(
      dot(aa, dot(ab, ac, matrix.descriptors.at(1)), matrix.descriptors.at(0)),
      dot(dot(aa, ab, reverseAttrs.at(3)), ac, reverseAttrs.at(2)));
  auto lora = symbolicPlan(
      "rule r {(dot[@outer] X (add W (dot[@inner] A B))) => (add (dot[auto] X "
      "W) (dot[auto] (dot[auto] X A) B)) where { "
      "@outer.rhs_batching_dimensions == @inner->B; "
      "@outer.rhs_contracting_dimensions == @inner->I; }}");
  proved(lora);
  Metadata lm{{{2, 3}, {3, 4}, {3, 2}, {2, 4}},
              {{0, {{1}, {0}, {}, {}}}, {1, {{1}, {0}, {}, {}}}}};
  tepl::test::ConstructorEvaluator loraEval(lm, lora);
  expect(loraEval.applicable(), "ordinary LoRA matrices satisfy guard");
  auto la = loraEval.construct();
  auto lx = data(lm.shapes[0], 1), lw = data(lm.shapes[1], 2),
       lla = data(lm.shapes[2], 3), lb = data(lm.shapes[3], 4);
  equal(dot(lx, add(lw, dot(lla, lb, lm.descriptors.at(1))),
            lm.descriptors.at(0)),
        add(dot(lx, lw, la.at(2)), dot(dot(lx, lla, la.at(4)), lb, la.at(3))));
  auto trans = symbolicPlan(
      "rule r {(dot[@d] A B) => (dot[auto] (transpose[auto] A) B)}");
  proved(trans);
  auto cycle = symbolicPlan(
      "rule r { A: [K, M, N] (dot[@d] A B) => (dot[auto] (transpose[auto, "
      "permutation=[1,2,0]] A) B) where { @d.lhs_batching_dimensions == "
      "range(0); @d.lhs_contracting_dimensions == range(1); }}");
  proved(cycle);
  Metadata good{{{3, 2, 4}, {3, 5}}, {{0, {{0}, {0}, {}, {}}}}};
  tepl::test::ConstructorEvaluator goodEval(good, cycle);
  expect(goodEval.applicable(),
         "fixed transpose preserving free order is applicable");
  auto ca = goodEval.construct();
  auto ga = data(good.shapes[0], 1), gb = data(good.shapes[1], 2);
  equal(dot(ga, gb, good.descriptors.at(0)),
        dot(transpose(ga, ca.at(2)[0]), gb, ca.at(1)));
  Metadata bad{{{2, 4, 3}, {3, 5}}, {{0, {{2}, {0}, {}, {}}}}};
  tepl::test::ConstructorEvaluator badEval(bad, cycle);
  expect(!badEval.applicable(),
         "fixed transpose changing free order is rejected by guard");
  Metadata batched{{{2, 2, 3}, {2, 3, 4}, {2, 4, 5}},
                   {{0, {{2}, {1}, {0}, {0}}}, {1, {{2}, {1}, {0}, {0}}}}};
  tepl::test::ConstructorEvaluator batchedEval(batched, association);
  expect(batchedEval.applicable(),
         "compatible shared batch groups remain live");
  auto ba = batchedEval.construct();
  auto batchedA = data(batched.shapes[0], 1),
       batchedB = data(batched.shapes[1], 2),
       batchedC = data(batched.shapes[2], 3);
  equal(dot(dot(batchedA, batchedB, batched.descriptors.at(1)), batchedC,
            batched.descriptors.at(0)),
        dot(batchedA, dot(batchedB, batchedC, ba.at(3)), ba.at(2)));
  for (bool zero : {false, true}) {
    auto k = zero ? 0 : 3;
    Metadata grouped{{{2, 2, 2, k, 2}, {2, 2, 2, 3, 2, k}, {3, 2, 2, 2, 2}},
                     {{0, {{3, 4}, {3, 0}, {0, 1}, {4, 2}}},
                      {1, {{3, 0}, {5, 1}, {4, 1}, {4, 2}}}}};
    tepl::test::ConstructorEvaluator eval(grouped, association);
    expect(eval.applicable(),
           "multiple batch groups and unsorted contracting axes");
    auto attributes = eval.construct();
    auto a = data(grouped.shapes[0], 1), b = data(grouped.shapes[1], 2),
         c = data(grouped.shapes[2], 3);
    equal(
        dot(dot(a, b, grouped.descriptors.at(1)), c, grouped.descriptors.at(0)),
        dot(a, dot(b, c, attributes.at(3)), attributes.at(2)));
    expect(attributes.at(2)[1] == List({3, 2}),
           "ordering permutation reconstructs paired contraction order");
  }
  Metadata incompatible{
      {{2, 3, 2, 3}, {2, 3, 3, 4}, {3, 3, 4, 5}},
      {{0, {{3}, {2}, {1}, {0}}}, {1, {{3}, {2}, {0, 1}, {0, 1}}}}};
  tepl::test::ConstructorEvaluator incompatibleEval(incompatible, association);
  expect(!incompatibleEval.applicable(),
         "partial-group batch routing remains an explicit restriction");

  auto renamed = replace(dialect, "dot_general", "contract");
  renamed = replace(renamed, "alias: dot;", "alias: arbitrary;");
  renamed = replace(renamed, "lhs_contracting_dimensions", "left_axes");
  renamed = replace(renamed, "bind K", "bind Reduction");
  renamed = replace(renamed, "Reduce(K,", "Reduce(Reduction,");
  proved(
      symbolicPlan(replace(distribute, "dot[", "arbitrary["), true, renamed));
  auto alternate = replace(dialect, "dialect TensorLang", "dialect Alternate");
  alternate = replace(alternate, "op dot_general(", "op dot_variant(");
  alternate = replace(alternate, "op add(", "op other_add(");
  alternate = replace(alternate, "op multiply(", "op other_multiply(");
  alternate = replace(alternate, "alias: mul;", "alias: other_mul;");
  alternate = replace(alternate, "op transpose(", "op other_transpose(");
  alternate = replace(alternate, "op reduce_sum(", "op other_sum(");
  alternate = replace(alternate, "alias: dot;", "alias: other;");
  alternate = replace(
      alternate,
      "bind B { lhs: lb, rhs: rb };\n            bind K { lhs: lc, rhs: rc };",
      "bind K { lhs: lc, rhs: rc };\n            bind B { lhs: lb, rhs: rb };");
  alternate = replace(alternate, "lhs_contracting_dimensions", "left_axes");
  proved(symbolicPlan("rule r {(dot[@d] A B) => (other[auto] A B)}", true,
                      dialect + alternate));
  auto declared = symbolicPlan(
      "rule r { A: [N,M,K] (transpose[@p] A) => (transpose[auto] A) where { N "
      "> 0; M == K; } }");
  proved(declared);
  expect(declared.assumptions.size() >= 3,
         "tensor declarations and pure where conditions imported");
  auto host = symbolicPlan(
      "rule r {(transpose[@p] X) => (transpose[@copy] X) derive {@copy = "
      "$copy_permutation(@p);} }");
  expect(host.proof.status == ProofStatus::Unknown,
         "host behavior requires a semantic contract");

  const std::string restricted = R"(
    dialect Restricted {
      op sum(a:tensor, b:tensor)->tensor {
        dtype(x,y) { assert x == y; assert is_float(x); yield x; }
        axes {bind S {a:all_axes(a), b:all_axes(b)}; yield S;}
        semantics {yield Add(a,b);}
      }
    }
  )";
  auto guarded = symbolicPlan("rule r {(add A B) => (Restricted.sum A B)}",
                              true, dialect + restricted);
  expect(guarded.proof.status == ProofStatus::Unknown,
         "undeclared RHS dtype requirement is rejected");
}
void explicitConditionTests() {
  auto check = [&](const std::string& source) {
    auto parsed = tepl::parse(dialect + source, "conditions.tepl");
    for (const auto& d : parsed.diagnostics)
      std::cerr << d.message << " in " << source << '\n';
    expect(parsed.ok(), "condition fixture parses");
    return tepl::core::analyze(*parsed.program,
                               {tepl::core::SemanticPolicy::kAlgebraic});
  };
  auto rejected = [&](const std::string& source, const std::string& message) {
    auto result = check(source);
    expect(!result.ok(), "normal analysis rejects unverified rules");
    for (const auto& d : result.diagnostics)
      if (d.message.find(message) != std::string::npos) return;
    throw std::runtime_error("missing rule diagnostic: " + message);
  };
  const std::string association =
      "rule r {(dot[@d] (dot[@inner] A B) C) => (dot[auto] A (dot[auto] B C))";
  rejected(association + "}", "cannot verify rule 'r'");
  auto accepted = check(association + R"( where {
    @d.lhs_batching_dimensions == @inner->B;
    @d.lhs_contracting_dimensions == @inner->J;
  }})");
  expect(accepted.ok(), "declared group routing accepted by normal analysis");
  auto wrong = check(association + R"( where {
    @d.lhs_batching_dimensions == @inner->B;
    @d.lhs_contracting_dimensions == @inner->I;
  }})");
  expect(wrong.ok(), "whole lhs-free contraction also admits reassociation");
  rejected(
      association + " where { @d.lhs_contracting_dimensions == @inner->K; }}",
      "absent from the operation output");
  rejected(association +
               " where { @d.lhs_contracting_dimensions == @inner->Missing; }}",
           "unknown output-axis group");
  rejected(
      "rule r {(transpose[@p] A) => (transpose[@copy] A) derive { @copy = "
      "@p->S; }}",
      "only supported in where");
  rejected(
      "rule r {(transpose[@p] A) => (transpose[auto] A) where { $accept(@p); "
      "}}",
      "unsupported semantic-rule condition");
  rejected(
      "rule r {(dot[@d] A B) => (dot[@new] A B) where { len(@new->B) == 0; } "
      "derive { @new = $copy(@d); }}",
      "not available here");
  rejected(
      "rule r {(dot[@d] (dot[@d] A B) C) => (dot[auto] A (dot[auto] B C)) "
      "where { len(@d->B) == 0; }}",
      "ambiguous output-axis reference");

  // Group names come from definitions, not a hard-coded B/I/J convention.
  auto renamed = replace(dialect, "bind J", "bind RightFree");
  renamed = replace(renamed, "B ++ I ++ J", "B ++ I ++ RightFree");
  auto named = symbolicPlan(association + R"( where {
    @d.lhs_batching_dimensions == @inner->B;
    @d.lhs_contracting_dimensions == @inner->RightFree;
  }})",
                            true, renamed);
  proved(named);

  // The group follows its declared coordinate order through a permutation.
  auto permutation = symbolicPlan(
      "rule r {(transpose[@p] A) => (transpose[auto] A) where { @p.permutation "
      "== @p->S; }}",
      false);
  proved(permutation);
  Metadata cycle{{{2, 3, 4}}, {{0, {{2, 0, 1}}}}};
  tepl::test::ConstructorEvaluator notInvolution(cycle, permutation);
  expect(!notInvolution.applicable(),
         "arrow denotes inverse positions, not output traversal order");
  Metadata swap{{{2, 3}}, {{0, {{1, 0}}}}};
  tepl::test::ConstructorEvaluator involution(swap, permutation);
  expect(involution.applicable(),
         "declared permutation condition evaluated independently");

  // A guard is required only when legality alone cannot establish the rule.
  expect(check("rule r {(dot[@d] (add A B) C) => (add (dot[auto] A C) "
               "(dot[auto] B C))}")
             .ok(),
         "unrestricted distribution remains accepted");
}
void partialAssociationTests() {
  const std::string rule = R"(rule r {
    (dot[@d] (dot[@inner] A B) C) => (dot[auto] A (dot[auto] B C))
    where { @d.lhs_batching_dimensions == @inner->B; }
  })";
  auto plan = symbolicPlan(rule);
  proved(plan);
  expect(synthesize(plan, 750000).status == ProofStatus::Proved,
         "partial reassociation fits a bounded proof-work budget");
  expect(verify(plan, plan.proof, 750000).status == ProofStatus::Proved,
         "partial reassociation independently verifies within budget");
  std::vector<Metadata> cases{
      // Whole J, whole I, I-only partial, and no outer contraction.
      {{{2, 3}, {3, 4}, {4, 2}},
       {{0, {{1}, {0}, {}, {}}}, {1, {{1}, {0}, {}, {}}}}},
      {{{2, 3}, {3, 4}, {2, 5}},
       {{0, {{0}, {0}, {}, {}}}, {1, {{1}, {0}, {}, {}}}}},
      {{{2, 3, 2}, {2, 4}, {3, 2}},
       {{0, {{1}, {0}, {}, {}}}, {1, {{2}, {0}, {}, {}}}}},
      {{{2, 3}, {3, 4}, {5}}, {{0, {{}, {}, {}, {}}}, {1, {{1}, {0}, {}, {}}}}},
      // Mixed I/J contraction, interleaved paired selectors, shared batch.
      {{{2, 2, 3, 2}, {2, 2, 3, 2}, {2, 3, 3, 2}},
       {{0, {{3, 2}, {2, 1}, {0}, {0}}}, {1, {{3}, {1}, {0}, {0}}}}},
      // Same routing with two batch dimensions.
      {{{2, 2, 2, 3, 2}, {2, 2, 2, 3, 2}, {2, 2, 3, 3, 2}},
       {{0, {{4, 3}, {3, 2}, {0, 1}, {0, 1}}},
        {1, {{4}, {2}, {0, 1}, {0, 1}}}}},
      // Multiple original contractions in an unsorted paired order.
      {{{2, 3, 2, 2}, {2, 2, 3, 2}, {3, 2, 2}},
       {{0, {{1, 0}, {0, 2}, {}, {}}}, {1, {{3, 2}, {0, 1}, {}, {}}}}},
      // Empty reduction domains and rank-zero inputs.
      {{{2, 0, 2}, {2, 3}, {0, 2}},
       {{0, {{1}, {0}, {}, {}}}, {1, {{2}, {0}, {}, {}}}}},
      {{{}, {}, {}}, {{0, {{}, {}, {}, {}}}, {1, {{}, {}, {}, {}}}}},
  };
  for (const auto& metadata : cases) {
    tepl::test::ConstructorEvaluator evaluator(metadata, plan);
    expect(evaluator.applicable(),
           "legal partial routing satisfies batch guard");
    auto attrs = evaluator.construct();
    auto a = data(metadata.shapes[0], 1), b = data(metadata.shapes[1], 2),
         c = data(metadata.shapes[2], 3);
    equal(dot(dot(a, b, metadata.descriptors.at(1)), c,
              metadata.descriptors.at(0)),
          dot(a, dot(b, c, attrs.at(3)), attrs.at(2)));
  }
  auto corrupt = plan.proof;
  corrupt.constructors.at(2)[0] = make(Symbol::Kind::List);
  expect(verify(plan, corrupt).status == ProofStatus::Unknown,
         "a corrupted outer contraction is rejected independently");
  auto wrongPairing = plan.proof;
  auto paired = wrongPairing.constructors.at(2)[1];
  wrongPairing.constructors.at(2)[1] =
      call("gather", {paired, call("order_positions", {paired})});
  expect(verify(plan, wrongPairing).status == ProofStatus::Unknown,
         "independent verification rejects altered contraction pair ordering");
  auto definitions = replace(dialect, "dot_general", "contract");
  definitions = replace(definitions, "alias: dot;", "alias: arbitrary;");
  definitions = replace(definitions, "bind I", "bind LeftFree");
  definitions = replace(definitions, "B ++ I ++ J", "B ++ LeftFree ++ J");
  proved(symbolicPlan(replace(rule, "dot[", "arbitrary["), true, definitions));
}
void contextStateTests() {
  Plan plan;
  Context c{plan, 100000};
  auto domain = [](std::initializer_list<std::size_t> extents) {
    std::vector<S> xs;
    for (auto n : extents) xs.push_back(integer(n));
    return make(Symbol::Kind::List, {}, 0, 0, std::move(xs));
  };
  auto frame = make(Symbol::Kind::AxisGroup, "frame", 0, 0, {domain({2, 3})});
  auto i = make(Symbol::Kind::AxisGroup, "i", 1, 0, {domain({2})});
  auto j = make(Symbol::Kind::AxisGroup, "j", 2, 0, {domain({3})});
  expect(equal(c.support(frame), frame), "cache initial frame support");
  c.cover(frame, call("concat", {i, j}));
  expect(equal(c.support(frame), c.support(call("concat", {i, j}))),
         "coverage registration invalidates cached support");
  expect(callIs(c.positions(frame, j), "positions"),
         "cache unresolved positions before binding registration");
  auto selector = make(Symbol::Kind::List, {}, 0, 0, {integer(1)});
  c.bind(frame, j, selector);
  expect(equal(c.positions(frame, j), selector) &&
             equal(c.select(frame, selector), j),
         "binding registration updates both lookup directions and caches");

  auto first = make(Symbol::Kind::Hole, {}, 0, 0);
  auto second = make(Symbol::Kind::Hole, {}, 0, 1);
  auto fields = call("concat", {first, second});
  expect(containsHole(c.resolve(fields)),
         "cache unresolved attribute expression");
  c.assignAttribute(0, 0, selector);
  expect(containsHole(c.resolve(fields)), "one attribute remains unresolved");
  auto zero = make(Symbol::Kind::List, {}, 0, 0, {integer(0)});
  c.assignAttribute(0, 1, zero);
  expect(equal(c.resolve(fields), call("concat", {selector, zero})),
         "attribute registration invalidates cached compound resolutions");
  bool conflict = false;
  try {
    c.assignAttribute(0, 0, zero);
  } catch (const ProofFailure& e) {
    conflict = e.kind() == FailureKind::InvalidCandidate;
  }
  expect(conflict && equal(c.resolve(first), selector),
         "conflicting assignments preserve the established constructor");

  // Equal coordinate sets imply equal lengths, but deriving that implication
  // belongs to entailment rather than evidence-only normalization.
  Context queries{plan, 100000};
  auto a = call("replace_axes", {frame, i, j});
  auto b = call("replace_axes", {frame, j, i});
  queries.learn(call("distinct_axes", {a}));
  queries.learn(call("distinct_axes", {b}));
  queries.learn(call("same_set", {a, b}));
  auto lengths = call("==", {call("len", {a}), call("len", {b})});
  expect(!queries.evidence(lengths),
         "evidence lookup does not launch proof search");
  auto selected = call("gather", {a, call("range", {call("len", {b})})});
  expect(callIs(queries.canonical(selected), "gather"),
         "normalization does not search for a missing length premise");
  expect(queries.entails(lengths) && queries.evidence(lengths),
         "successful entailment is visible after an earlier evidence miss");
  expect(queries.guards().empty(), "proof queries do not assume missing facts");

  auto exhausted = synthesize(plan, 0);
  expect(exhausted.status == ProofStatus::Unknown &&
             exhausted.reason.find("budget exhausted") != std::string::npos,
         "budget exhaustion has a specific diagnostic");
  bool internal_error = false;
  try {
    synthesize(plan,
               1000);  // No source node: deliberately malformed internal IR.
  } catch (const std::out_of_range&) {
    internal_error = true;
  }
  expect(internal_error,
         "internal errors are not reported as unsupported inference");
}
void axisAndMultiplyTests() {
  auto merge = symbolicPlan(
      "rule r {(reduce_sum[@outer] (reduce_sum[@inner] A)) => "
      "(reduce_sum[auto] A)}");
  proved(merge);
  auto move = symbolicPlan(
      "rule r {(reduce_sum[@r] (transpose[@p] A)) => "
      "(transpose[auto] (reduce_sum[auto] A))}");
  proved(move);
  expect(synthesize(move, 50000).status == ProofStatus::Proved,
         "transpose/reduction inference fits a bounded metadata budget");
  expect(verify(move, move.proof, 50000).status == ProofStatus::Proved,
         "independent map verification fits the same budget");
  Context theory{move, 1000000};
  theory.sourceScope(move.lhs);
  Lowerer source{theory};
  source.lower(move.lhs);
  auto p = make(Symbol::Kind::Field, "permutation", 1, 0);
  auto r = make(Symbol::Kind::Field, "dimensions", 0, 0);
  auto twice = [](const S& x) {
    return call("inverse", {call("inverse", {x})});
  };
  expect(equal(theory.canonical(twice(p)), p),
         "source legality permits full permutation cancellation");
  auto partial = theory.canonical(twice(r));
  expect(!equal(partial, r),
         "a partial axis list does not justify inverse cancellation");
  Metadata partialMetadata{{{2, 3, 4}}, {{0, {{2}}}, {1, {{2, 0, 1}}}}};
  tepl::test::ConstructorEvaluator partialEval(partialMetadata, move.proof);
  bool undefined = false;
  try {
    partialEval.eval(partial);
  } catch (const std::exception&) {
    undefined = true;
  }
  expect(undefined, "independent interpreter preserves undefined inverse");
  bool unproved = false;
  try {
    theory.wellFormed(twice(r));
  } catch (const ProofFailure&) {
    unproved = true;
  }
  expect(unproved && !theory.guards().empty(),
         "partial inverse is rejected without an explicit condition");
  expect(!theory.entails(call("axis_partition", {r, call("len", {r})})),
         "failed obligation did not become an assumption");
  auto frame = make(Symbol::Kind::AxisGroup, "input0", 0, 0,
                    {make(Symbol::Kind::Shape, {}, 0)});
  // A fresh context must agree with the failed context on legality.
  Context lookup{move, 1000000};
  lookup.sourceScope(move.lhs);
  Lowerer lookupSource{lookup};
  lookupSource.lower(move.lhs);
  auto incomplete =
      lookup.positions(call("gather", {frame, r}), call("gather", {frame, p}));
  expect(callIs(incomplete, "positions"),
         "partial interfaces retain an explicit position lookup");
  Context aliases{move, 100000};
  auto domain = make(Symbol::Kind::Shape, {}, 0);
  auto x = make(Symbol::Kind::AxisGroup, "x", 0, 0, {domain});
  auto y = make(Symbol::Kind::AxisGroup, "y", 1, 0, {domain});
  aliases.learn(call("==", {x, y}));
  expect(!aliases.entails(call("disjoint", {x, y})),
         "coordinate aliases are not independent reduction identities");
  proved(symbolicPlan("rule r {(mul (mul A B) C) => (mul A (mul B C))}"));
  proved(
      symbolicPlan("rule r {(mul (add A B) C) => (add (mul A C) (mul B C))}"));
  proved(
      symbolicPlan("rule r {(add (mul A C) (mul B C)) => (mul (add A B) C)}"));
  auto trans = symbolicPlan(
      "rule r {(transpose[@p] (mul A B)) => "
      "(mul (transpose[auto] A) (transpose[auto] B))}");
  proved(trans);
  expect(symbolicPlan("rule r {(mul A B) => (add A B)}").proof.status ==
             ProofStatus::Unknown,
         "different scalar primitives are rejected");
  expect(symbolicPlan("rule r {(mul (add A B) C) => (add (mul A C) B)}")
                 .proof.status == ProofStatus::Unknown,
         "incorrect distribution is rejected");
  expect(symbolicPlan("rule r {(mul A B) => (mul B A)}", false).proof.status ==
             ProofStatus::Unknown,
         "multiplication commutativity requires algebraic policy");
  auto renamed = replace(dialect, "op multiply(", "op scalar_product(");
  renamed = replace(renamed, "alias: mul;", "alias: times;");
  proved(
      symbolicPlan("rule r {(times (add A B) C) => "
                   "(add (times A C) (times B C))}",
                   true, renamed));
  // Specialize the same compiled constructors at different ranks, axis orders,
  // and domains. No concrete metadata is used by the compiler.
  for (const List& shape :
       {List{}, List{2}, List{2, 3}, List{2, 1, 3, 2}, List{2, 0, 3, 2}}) {
    auto rank = shape.size();
    auto input = data(shape, 1), b = data(shape, 2), c = data(shape, 3);
    equal(multiply(add(input, b), c), add(multiply(input, c), multiply(b, c)));
    for (std::size_t mask = 0; mask < (std::size_t{1} << rank); ++mask) {
      List inner, outer;
      for (std::size_t axis = 0; axis < rank; ++axis)
        if (mask & (std::size_t{1} << axis)) inner.push_back(axis);
      std::reverse(inner.begin(), inner.end());
      auto retained = remaining(rank, inner);
      for (std::size_t axis = 0; axis < retained.size(); axis += 2)
        outer.push_back(axis);
      std::reverse(outer.begin(), outer.end());
      Metadata m{{shape}, {{0, {outer}}, {1, {inner}}}};
      tepl::test::ConstructorEvaluator eval(m, merge.proof);
      expect(eval.applicable(), "merged reduction needs no extra guard");
      auto fields = eval.construct();
      equal(reduceSum(reduceSum(input, inner), outer),
            reduceSum(input, fields.at(2)[0]));
    }
    List permutation(rank);
    std::iota(permutation.begin(), permutation.end(), 0);
    do {
      Metadata tm{{shape, shape}, {{0, {permutation}}}};
      tepl::test::ConstructorEvaluator transEval(tm, trans.proof);
      auto tf = transEval.construct();
      equal(transpose(multiply(input, b), permutation),
            multiply(transpose(input, tf.at(1)[0]), transpose(b, tf.at(2)[0])));
      for (std::size_t mask = 0; mask < (std::size_t{1} << rank); ++mask) {
        List dims;
        for (std::size_t axis = 0; axis < rank; ++axis)
          if (mask & (std::size_t{1} << axis)) dims.push_back(axis);
        std::reverse(dims.begin(), dims.end());
        Metadata rm{{shape}, {{0, {dims}}, {1, {permutation}}}};
        tepl::test::ConstructorEvaluator moveEval(rm, move.proof);
        expect(moveEval.applicable(),
               "transpose/reduction needs no extra guard");
        auto f = moveEval.construct();
        equal(reduceSum(transpose(input, permutation), dims),
              transpose(reduceSum(input, f.at(3)[0]), f.at(2)[0]));
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }
  auto bad = move.proof;
  bad.constructors.at(3)[0] = make(Symbol::Kind::List);
  expect(verify(move, bad).status == ProofStatus::Unknown,
         "wrong reduction attribute is rejected");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    expect(argc == 2, "dialect argument");
    std::string err;
    auto files = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &err));
    expect(bool(files), err);
    std::ifstream input(files->Rlocation(argv[1]));
    dialect = {std::istreambuf_iterator<char>(input), {}};
    expect(!dialect.empty(), "read dialect");
    symbolicTests();
    explicitConditionTests();
    partialAssociationTests();
    contextStateTests();
    axisAndMultiplyTests();
    const std::string distribute =
        "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B "
        "C)) }";
    const std::vector<List> matrix{{1}, {0}, {}, {}};
    auto result = run(distribute, {{{2, 3}, {2, 3}, {3, 4}}, {{0, matrix}}});
    found(result);
    expect(result.attributes.size() == 2, "two auto descriptors");
    found(run("rule r { (dot[@d] (let T = (add A B)) C) => (dot[auto] T C) }",
              {{{2, 3}, {2, 3}, {}, {3, 4}}, {{0, matrix}}}));
    for (const auto& [id, fields] : result.attributes)
      expect(fields == matrix, "distribution mappings reconstructed");
    auto a = data({2, 3}, 1), b = data({2, 3}, 2), c = data({3, 4}, 3);
    equal(dot(add(a, b), c, matrix), add(dot(a, c, result.attributes.at(1)),
                                         dot(b, c, result.attributes.at(2))));
    expect(std::find(result.trace.begin(), result.trace.end(),
                     "law: product-distributivity") != result.trace.end(),
           "distribution trace");
    expect(std::find(result.trace.begin(), result.trace.end(),
                     "law: reduction-linearity") != result.trace.end(),
           "reduction trace");

    std::vector<List> mixed{{1, 3}, {2, 0}, {2}, {3}};
    auto multi =
        run(distribute,
            {{{7, 3, 5, 11}, {7, 3, 5, 11}, {11, 13, 3, 5}}, {{0, mixed}}});
    found(multi);
    auto ma = data({7, 3, 5, 11}, 1), mb = data({7, 3, 5, 11}, 2),
         mc = data({11, 13, 3, 5}, 3);
    equal(dot(add(ma, mb), mc, mixed),
          add(dot(ma, mc, multi.attributes.at(1)),
              dot(mb, mc, multi.attributes.at(2))));
    for (auto rank : {0, 1, 20, 100}) {
      List s(rank, 2), axes(rank);
      std::iota(axes.begin(), axes.end(), 0);
      found(run(distribute, {{s, s, s}, {{0, {{}, {}, axes, axes}}}}));
    }
    found(run(distribute, {{{2, 0}, {2, 0}, {0, 4}}, {{0, matrix}}}));
    expect(run(distribute, {{{2, 3}, {2, 4}, {3, 4}}, {{0, matrix}}}).status ==
               Status::IllegalMetadata,
           "source addition shape mismatch");
    expect(run(distribute, {{{2, 3}, {2, 3}, {3, 4}}, {{0, matrix}}}, {1, 1000})
                   .status == Status::Unknown,
           "term budget is unknown");
    expect(run(distribute, {{{2, 3}, {2, 3}, {3, 4}}, {{0, matrix}}}, {256, 1})
                   .status == Status::Unknown,
           "work budget is unknown");

    auto sum =
        run("rule r { (reduce_sum[@d] (add A B)) => (add (reduce_sum[auto] A) "
            "(reduce_sum[auto] B)) }",
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, {{0, {{1, 3}}}}});
    found(sum);
    for (const auto& [id, fields] : sum.attributes)
      expect(fields == std::vector<List>{{1, 3}}, "reduction attributes");
    found(run("rule r { (reduce_sum[@d] A) => (reduce_sum[auto] A) }",
              {{{2, 3}}, {{0, {{}}}}}));
    found(run("rule r { (reduce_sum[@d] A) => (reduce_sum[auto] A) }",
              {{{2, 3}}, {{0, {{0, 1}}}}}));

    found(run("rule r { (transpose[@p] A) => (transpose[auto] A) }",
              {{{2, 3, 4}}, {{0, {{2, 0, 1}}}}}));
    auto composed =
        run("rule r { (transpose[@outer] (transpose[@inner] A)) => "
            "(transpose[auto] A) }",
            {{{2, 3, 4}}, {{0, {{1, 0, 2}}}, {1, {{2, 0, 1}}}}});
    found(composed);
    auto input_tensor = data({2, 3, 4}, 1);
    equal(transpose(transpose(input_tensor, {2, 0, 1}), {1, 0, 2}),
          transpose(input_tensor, composed.attributes.at(2)[0]));
    auto trans =
        run("rule r { (dot[@d] A B) => (dot[auto] (transpose[auto, "
            "permutation=[1,2,0]] A) B) }",
            {{{3, 2, 4}, {3, 5}}, {{0, {{0}, {0}, {}, {}}}}});
    found(trans);
    expect(trans.attributes.at(1) == std::vector<List>{{2}, {0}, {}, {}},
           "inverse contraction positions after transpose");
    auto ta = data({3, 2, 4}, 1), tb = data({3, 5}, 2);
    equal(dot(ta, tb, {{0}, {0}, {}, {}}),
          dot(transpose(ta, trans.attributes.at(2)[0]), tb,
              trans.attributes.at(1)));
    expect(run("rule r { (dot[@d] A B) => (dot[auto] (transpose[auto, "
               "permutation=[1,2,0]] A) B) }",
               {{{2, 4, 3}, {3, 5}}, {{0, {{2}, {0}, {}, {}}}}})
                   .status == Status::Unknown,
           "output-order obstruction is not a general impossibility claim");
    found(run("rule r { (dot[@d] A B) => (dot[auto] (transpose[auto] A) B) }",
              {{{2, 3}, {3, 4}}, {{0, matrix}}}));
    found(
        run("rule r { (transpose[@p] (dot[@d] A B)) => (dot[auto] "
            "(transpose[auto] B) (transpose[auto] A)) }",
            {{{2, 3}, {3, 4}}, {{0, {{1, 0}}}, {1, matrix}}}));
    expect(run("rule r { (transpose[@p] A) => (transpose[auto] A) }",
               {{{2, 3}}, {{0, {{0, 0}}}}})
                   .status == Status::IllegalMetadata,
           "bad source permutation");
    expect(run("rule r { (transpose[@p] A) => (transpose[auto, "
               "permutation=[0,0]] A) }",
               {{{2, 3}}, {{0, {{1, 0}}}}})
                   .status == Status::Unknown,
           "bad pin rejected");
    auto rhs_pin =
        run("rule r { (dot[@d] A B) => (dot[auto, "
            "rhs_contracting_dimensions=[0,1]] A B) }",
            {{{2, 3, 5}, {5, 3, 7}}, {{0, {{1, 2}, {1, 0}, {}, {}}}}});
    found(rhs_pin);
    expect(rhs_pin.attributes.at(1)[0] == List({2, 1}),
           "rhs pin controls shared contraction-group ordering");

    auto association =
        run("rule r { (dot[@d] (dot[@inner] A B) C) => (dot[auto] A (dot[auto] "
            "B C)) }",
            {{{2, 3}, {3, 4}, {4, 5}}, {{0, matrix}, {1, matrix}}});
    found(association);
    auto aa = data({2, 3}, 1), ab = data({3, 4}, 2), ac = data({4, 5}, 3);
    equal(dot(dot(aa, ab, matrix), ac, matrix),
          dot(aa, dot(ab, ac, association.attributes.at(3)),
              association.attributes.at(2)));
    auto lora =
        run("rule r { (dot[@outer] X (add W (dot[@inner] A B))) => (add "
            "(dot[auto] X W) (dot[auto] (dot[auto] X A) B)) }",
            {{{3, 4}, {4, 5}, {4, 2}, {2, 5}}, {{0, matrix}, {1, matrix}}});
    found(lora);
    auto lx = data({3, 4}, 1), lw = data({4, 5}, 2), la = data({4, 2}, 3),
         lr = data({2, 5}, 4);
    equal(dot(lx, add(lw, dot(la, lr, matrix)), matrix),
          add(dot(lx, lw, lora.attributes.at(2)),
              dot(dot(lx, la, lora.attributes.at(4)), lr,
                  lora.attributes.at(3))));
    found(run("rule r { (add A (add B C)) => (add (add A B) C) }",
              {{{2, 3}, {2, 3}, {2, 3}}, {}}));
    expect(run("rule r { (add A B) => A }", {{{2, 3}, {2, 3}}, {}}).status ==
               Status::Unknown,
           "unequal terms never produce a witness");
    expect(run("rule r { (dot[@d] A A) => (dot[auto] A A) }",
               {{{2, 2}}, {{0, matrix}}})
                   .status == Status::Unknown,
           "repeated-factor matching is explicitly unsupported");
    auto strict = analyze(distribute, false);
    expect(strict.ok(), "strict plan compiles");
    expect(infer(*strict.program->rules[0].semantic_plan,
                 {{{2, 3}, {2, 3}, {3, 4}}, {{0, matrix}}})
                   .status == Status::Unknown,
           "strict policy does not silently apply algebraic laws");

    auto renamed = replace(dialect, "dot_general", "contract");
    renamed = replace(renamed, "alias: dot;", "alias: arbitrary;");
    renamed = replace(renamed, "lhs_contracting_dimensions", "left_axes");
    renamed = replace(renamed, "bind K", "bind Reduction");
    renamed = replace(renamed, "Reduce(K,", "Reduce(Reduction,");
    found(
        run("rule r { (arbitrary[@d] (add A B) C) => (add (arbitrary[auto] A "
            "C) (arbitrary[auto] B C)) }",
            {{{2, 3}, {2, 3}, {3, 4}}, {{0, matrix}}}, {}, renamed));
    error(
        "dialect D { op f(x:tensor)->tensor { axes { bind S {missing: "
        "all_axes(x)}; yield S; } semantics {yield x;} } }",
        "unknown binding operand");
    error(
        "dialect D { op f(x:tensor)->tensor { axes { let K=all_axes(x); bind S "
        "{x:K}; yield S; } semantics {yield Reduce(K,x);} } }",
        "exported bind group");
    error(
        "dialect D { op f(x:tensor)->tensor { axes {bind S {x:all_axes(x)}; "
        "yield S;} semantics {yield Reduce(S,x);} } }",
        "exactly the free groups");
    auto checked = analyze(distribute);
    auto dump = tepl::core::formatProgram(*checked.program);
    expect(dump.find("lhs-graph:") != std::string::npos &&
               dump.find("constructor auto#") != std::string::npos &&
               dump.find("status=proved") != std::string::npos,
           "compiler graph and symbolic constructor proof dump");
    expect(laws().size() >= 6, "explicit algebraic law registry");
    std::cout << "semantic tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
