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
                                        : tepl::core::SemanticPolicy::kStrict});
}
Result run(const std::string& rule, Metadata m, Limits limits = {},
           std::string definitions = {}) {
  auto checked = analyze(rule, true, definitions);
  for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  expect(checked.ok(), "fixture checks");
  expect(bool(checked.program->rules[0].index_inference), "new inference plan");
  expect(!checked.program->rules[0].inference, "legacy inference bypassed");
  return infer(*checked.program->rules[0].index_inference, m, limits);
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
  return *checked.program->rules.at(0).index_inference;
}
void proved(const Plan& p) {
  if (p.proof.status != ProofStatus::Proved) {
    std::cerr << p.proof.reason << '\n';
    for (const auto& g : p.proof.guard) std::cerr << format(*g) << '\n';
  }
  expect(p.proof.status == ProofStatus::Proved, "expected metadata-free proof");
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
  expect(symbolicPlan(distribute, false).proof.status == ProofStatus::Unknown,
         "strict policy prevents arithmetic expansion");

  proved(symbolicPlan("rule r {(add A (add B C)) => (add (add A B) C)}"));
  auto commutative = symbolicPlan("rule r {(add A B) => (add B A)}");
  proved(commutative);
  expect(std::find(commutative.proof.steps.begin(), commutative.proof.steps.end(),
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
  auto pinned = symbolicPlan(
      "rule r {(reduce_sum[@d] A) => (reduce_sum[auto, dimensions=[]] A)}");
  expect(pinned.proof.status == ProofStatus::Conditional &&
             !pinned.proof.guard.empty(),
         "pinned field yields explicit sufficient guard");
  expect(verify(pinned, pinned.proof).status == ProofStatus::Conditional,
         "guarded proof replays independently");
  auto association = symbolicPlan(
      "rule r {(dot[@d] (dot[@inner] A B) C) => (dot[auto] A (dot[auto] B "
      "C))}");
  expect(association.proof.status == ProofStatus::Conditional,
         "reassociation has a checked routing guard");
  Metadata matrix{{{2, 3}, {3, 4}, {4, 5}},
                  {{0, {{1}, {0}, {}, {}}}, {1, {{1}, {0}, {}, {}}}}};
  tepl::test::ConstructorEvaluator associationEval(matrix, association.proof);
  expect(associationEval.applicable(),
         "ordinary matrices satisfy generated reassociation guard");
  auto attrs = associationEval.construct();
  auto aa = data({2, 3}, 1), ab = data({3, 4}, 2), ac = data({4, 5}, 3);
  equal(
      dot(dot(aa, ab, matrix.descriptors.at(1)), ac, matrix.descriptors.at(0)),
      dot(aa, dot(ab, ac, attrs.at(3)), attrs.at(2)));
  expect(
      verify(association, association.proof).status == ProofStatus::Conditional,
      "reassociation certificate independently verifies");
  for (bool zero : {false, true}) {
    Metadata multi{
        {{2, 2, 3, 2}, {2, zero ? 0 : 4, 2, 2}, {2, 3, zero ? 0 : 4}},
        {{0, {{2, 3}, {2, 0}, {}, {}}}, {1, {{1, 3}, {2, 0}, {}, {}}}}};
    tepl::test::ConstructorEvaluator eval(multi, association.proof);
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
      "C)}");
  expect(reverse.proof.status == ProofStatus::Conditional,
         "reverse association supported with guard");
  tepl::test::ConstructorEvaluator reverseEval(matrix, reverse.proof);
  expect(reverseEval.applicable(), "matrix reverse association guard");
  auto reverseAttrs = reverseEval.construct();
  equal(
      dot(aa, dot(ab, ac, matrix.descriptors.at(1)), matrix.descriptors.at(0)),
      dot(dot(aa, ab, reverseAttrs.at(3)), ac, reverseAttrs.at(2)));
  auto lora = symbolicPlan(
      "rule r {(dot[@outer] X (add W (dot[@inner] A B))) => (add (dot[auto] X "
      "W) (dot[auto] (dot[auto] X A) B))}");
  expect(lora.proof.status == ProofStatus::Conditional,
         "LoRA composition supported with guard");
  Metadata lm{{{2, 3}, {3, 4}, {3, 2}, {2, 4}},
              {{0, {{1}, {0}, {}, {}}}, {1, {{1}, {0}, {}, {}}}}};
  tepl::test::ConstructorEvaluator loraEval(lm, lora.proof);
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
      "rule r {(dot[@d] A B) => (dot[auto] (transpose[auto, "
      "permutation=[1,2,0]] A) B)}");
  expect(cycle.proof.status == ProofStatus::Conditional,
         "fixed transpose produces rank/order guard");
  Metadata good{{{3, 2, 4}, {3, 5}}, {{0, {{0}, {0}, {}, {}}}}};
  tepl::test::ConstructorEvaluator goodEval(good, cycle.proof);
  expect(goodEval.applicable(),
         "fixed transpose preserving free order is applicable");
  auto ca = goodEval.construct();
  auto ga = data(good.shapes[0], 1), gb = data(good.shapes[1], 2);
  equal(dot(ga, gb, good.descriptors.at(0)),
        dot(transpose(ga, ca.at(2)[0]), gb, ca.at(1)));
  Metadata bad{{{2, 4, 3}, {3, 5}}, {{0, {{2}, {0}, {}, {}}}}};
  tepl::test::ConstructorEvaluator badEval(bad, cycle.proof);
  expect(!badEval.applicable(),
         "fixed transpose changing free order is rejected by guard");
  Metadata batched{{{2, 2, 3}, {2, 3, 4}, {2, 4, 5}},
                   {{0, {{2}, {1}, {0}, {0}}}, {1, {{2}, {1}, {0}, {0}}}}};
  tepl::test::ConstructorEvaluator batchedEval(batched, association.proof);
  expect(!batchedEval.applicable(),
         "unbatched routing restriction is explicit");

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
  expect(guarded.proof.status == ProofStatus::Conditional,
         "new RHS dtype requirement is an explicit guard");
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
    expect(infer(*strict.program->rules[0].index_inference,
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
    std::cout << "index semantics tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
