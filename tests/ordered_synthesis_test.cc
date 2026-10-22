#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/pipeline.h"
#include "src/core/semantics/context.h"
#include "src/core/semantics/symbolic/expr.h"
#include "src/core/semantics/synthesis/ordered/concrete.h"
#include "src/core/syntax/parse.h"
#include "tests/index_constructor_eval.h"

namespace {
using namespace tepl::core::semantics;
namespace ord = tepl::core::semantics::ordered;
using ord::AxisSequence;
std::string dialect;
void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
Plan parse(const std::string& rule) {
  auto parsed = tepl::parse(dialect + rule, "ordered.tepl");
  expect(parsed.ok(), "ordered fixture parses");
  auto inspected = tepl::core::inspect(
      *parsed.program, {tepl::core::SemanticPolicy::kAlgebraic, true});
  expect(inspected.ok(), "ordered fixture checks");
  return *inspected.program->rules.at(0).semantic_plan;
}
std::vector<List> dotFields(const Definition& d, List lb = {}, List rb = {},
                            List lk = {}, List rk = {}) {
  const auto c = *ord::recognize(d);
  std::vector<List> fields(4);
  fields[c.left_batch] = std::move(lb);
  fields[c.right_batch] = std::move(rb);
  fields[c.left_contract] = std::move(lk);
  fields[c.right_contract] = std::move(rk);
  return fields;
}
void setDot(const Plan& p, ord::Metadata& m, std::size_t id, List lb = {},
            List rb = {}, List lk = {}, List rk = {}) {
  const auto& n = p.nodes[id];
  m.descriptors[n.descriptor->value] = dotFields(*n.definition, lb, rb, lk, rk);
}
void admission() {
  auto p =
      parse("rule r {(p: transpose (d: dot A B)) => (transpose (dot B A))}");
  expect(ord::admit(p).admitted(), "semantic dot/permutation admission");
  auto d = *p.nodes[p.nodes[p.lhs].inputs[0]].definition;
  d.fields = {"alpha", "beta", "gamma", "delta"};
  for (auto& g : d.groups) g.name = "renamed";
  expect(bool(ord::recognize(d)), "recognition is independent of names");
  auto restricted = d;
  restricted.assertions.push_back(d.output);
  expect(!ord::recognize(restricted), "additional assertions are not ignored");
  restricted = d;
  restricted.fields.push_back("hidden");
  expect(!ord::recognize(restricted), "unmodeled semantic attributes rejected");
  auto shared = p;
  shared.nodes[p.rhs].descriptor =
      shared.nodes[shared.nodes[p.rhs].inputs[0]].descriptor;
  expect(!ord::admit(shared).admitted(), "shared RHS descriptors rejected");
  p.algebraic = false;
  expect(!ord::admit(p).admitted(), "no strict-arithmetic completeness claim");
  auto repeated = parse("rule r {(d: dot A A) => (dot A A)}");
  expect(!ord::admit(repeated).admitted(), "repeated captures rejected");
}
void matricesAndDomains() {
  auto p =
      parse("rule r {(outer: dot (inner: dot A B) C) => (dot A (dot B C))}");
  ord::Metadata m{{{0, {2, 3}}, {1, {3, 4}}, {2, {4, 5}}}, {}};
  auto inner = p.nodes[p.lhs].inputs[0];
  setDot(p, m, inner, {}, {}, {1}, {0});
  setDot(p, m, p.lhs, {}, {}, {1}, {0});
  auto result = ord::solve(p, m);
  expect(result.status == ord::Status::Solved && ord::check(p, m, result),
         "matrix reassociation has an independently checked witness");
  auto corrupt = result;
  auto root_id = p.nodes[p.rhs].descriptor->value;
  auto c = *ord::recognize(*p.nodes[p.rhs].definition);
  corrupt.attributes[root_id][c.left_contract] = {0};
  expect(!ord::check(p, m, corrupt),
         "incorrect contracting positions rejected");
  corrupt = result;
  corrupt.source.classes[0] = corrupt.source.classes[1];
  expect(!ord::check(p, m, corrupt), "corrupt source provenance rejected");
  corrupt = result;
  corrupt.requirements.at(p.rhs).live.clear();
  expect(!ord::check(p, m, corrupt), "corrupt boundary certificate rejected");
  auto singleton = m;
  singleton.shapes[0][0] = 1;
  expect(ord::solve(p, singleton).status == ord::Status::OutsideDomain,
         "singleton extent is outside the necessity proof domain");
  singleton.shapes[0][0] = 0;
  expect(ord::solve(p, singleton).status == ord::Status::OutsideDomain,
         "zero extent is outside the necessity proof domain");
  auto invalid = m;
  invalid.shapes[1][0] = 7;
  expect(ord::solve(p, invalid).status == ord::Status::InvalidSource,
         "mismatched source extents are not impossibility");
  invalid = m;
  invalid.descriptors.erase(p.nodes[inner].descriptor->value);
  expect(ord::solve(p, invalid).status == ord::Status::MissingMetadata,
         "missing metadata is not impossibility");
  expect(ord::solve(p, m, 0).status == ord::Status::ResourceLimit,
         "work exhaustion is not impossibility");
}
void layouts() {
  auto p = parse("rule r {(p: transpose (d: dot A B)) => (dot A B)}");
  ord::Metadata m{{{0, {2}}, {1, {2}}}, {}};
  setDot(p, m, p.nodes[p.lhs].inputs[0]);
  m.descriptors[p.nodes[p.lhs].descriptor->value] = {{1, 0}};
  auto result = ord::solve(p, m);
  expect(result.status == ord::Status::Impossible && ord::check(p, m, result),
         "equal extents cannot hide an output-order obstruction");
  auto corrupt = result;
  std::swap(corrupt.obstruction->first, corrupt.obstruction->second);
  expect(!ord::check(p, m, corrupt), "false ordering obstruction rejected");
  auto repair =
      parse("rule r {(p: transpose (d: dot A B)) => (transpose (dot A B))}");
  auto fixed = ord::solve(repair, m);
  expect(fixed.status == ord::Status::Solved && ord::check(repair, m, fixed),
         "output transpose repairs an impossible ordering");
  auto internal =
      parse("rule r {(p: transpose (d: dot A B)) => (dot (transpose A) B)}");
  ord::Metadata im{{{0, {2, 3}}, {1, {5}}}, {}};
  setDot(internal, im, internal.nodes[internal.lhs].inputs[0]);
  im.descriptors[internal.nodes[internal.lhs].descriptor->value] = {{1, 0, 2}};
  auto reordered = ord::solve(internal, im);
  expect(reordered.status == ord::Status::Solved &&
             ord::check(internal, im, reordered),
         "parent order propagates to a smaller internal transpose");
  auto transpose_id = internal.nodes[internal.rhs].inputs[0];
  expect(reordered.attributes.at(internal.nodes[transpose_id].descriptor->value)
                 .at(0) == List({1, 0}),
         "internal permutation is constructed from its required subsequence");

  auto high = parse("rule r {(p: transpose A) => (transpose (transpose A))}");
  ord::Metadata hm{{{0, std::vector<std::uint64_t>(128, 2)}}, {}};
  List permutation(128);
  std::iota(permutation.rbegin(), permutation.rend(), 0);
  hm.descriptors[high.nodes[high.lhs].descriptor->value] = {permutation};
  auto high_result = ord::solve(high, hm);
  expect(high_result.status == ord::Status::Solved &&
             ord::check(high, hm, high_result),
         "arbitrary-length permutations and nonunique solutions");
}

void multiplePairings() {
  auto p = parse("rule r {(d: dot A B) => (transpose (dot B A))}");
  ord::Metadata m{{{0, {2, 3, 5, 7, 11}}, {1, {11, 13, 2, 5, 7}}}, {}};
  setDot(p, m, p.lhs, {2}, {3}, {4, 0, 3}, {0, 2, 4});
  auto result = ord::solve(p, m);
  expect(result.status == ord::Status::Solved && ord::check(p, m, result),
         "several unsorted contraction pairs and an independent batch");
  expect(result.attributes.at(p.nodes[p.rhs].descriptor->value).at(0) ==
             List({0, 2, 1}),
         "output permutation repairs free-axis blocks");

  m.shapes[0] = m.shapes[1] = std::vector<std::uint64_t>(5, 2);
  result = ord::solve(p, m);
  expect(result.status == ord::Status::Solved && ord::check(p, m, result),
         "equal extents preserve the full named-port partition");
  auto inner = p.nodes[p.rhs].inputs[0];
  auto c = *ord::recognize(*p.nodes[inner].definition);
  auto& right = result.attributes.at(p.nodes[inner].descriptor->value)
                    .at(c.right_contract);
  std::swap(right[0], right[1]);
  expect(!ord::check(p, m, result),
         "shape-legal but semantically different pairing is rejected");
}

void sequenceComparison() {
  Plan p;
  Context c{p, 1000000};
  auto shape_a = make(Symbol::Kind::Shape, {}, 0);
  auto shape_b = make(Symbol::Kind::Shape, {}, 1);
  auto a = make(Symbol::Kind::AxisGroup, "a", 0, 0, {shape_a});
  auto b = make(Symbol::Kind::AxisGroup, "b", 1, 0, {shape_b});
  c.learn(call("==", {call("len", {shape_a}), integer(2)}));
  c.learn(call("==", {call("len", {shape_b}), integer(1)}));
  auto list = [](std::initializer_list<std::size_t> values) {
    std::vector<S> xs;
    for (auto x : values) xs.push_back(integer(x));
    return make(Symbol::Kind::List, {}, 0, 0, std::move(xs));
  };
  auto gather = [&](const S& axes, std::initializer_list<std::size_t> indices) {
    return call("gather", {axes, list(indices)});
  };
  auto base = call("concat", {a, b});
  expect(c.entails(call("==", {gather(base, {1, 0, 2}),
                               call("concat", {gather(a, {1, 0}), b})})),
         "explicit permutations project through context-known blocks");
  expect(c.entails(
             call("==", {gather(base, {0, 2, 1}),
                         call("concat", {gather(a, {0}), b, gather(a, {1})})})),
         "projection preserves repeated visits to the same block");
  expect(!c.entails(call("==", {gather(base, {1, 0, 2}), base})),
         "projection does not erase a nontrivial within-block permutation");
  expect(!c.entails(call("==", {gather(base, {0, 2, 1}), base})),
         "projection does not reorder an interleaved selector");
  expect(!c.entails(call("==", {gather(base, {3, 0}), a})),
         "out-of-range selectors are not normalized into a valid sequence");
}

void compiledPrograms() {
  auto p = parse("rule r {(p: transpose A) => (transpose A)}");
  ord::Metadata m{{{0, {2, 2}}}, {}};
  m.descriptors[p.nodes[p.lhs].descriptor->value] = {{1, 0}};
  auto code = ord::compile(p);
  expect(code.program.has_value(),
         "compile a rank-independent inference program");
  auto guard = tepl::inference::feasibility(*code.program, m);
  expect(guard.status == ord::Status::Feasible && guard.attributes.empty() &&
             guard.layouts.empty() &&
             tepl::inference::check(*code.program, m, guard),
         "predicate is exact without running positional construction");
  auto result = tepl::inference::evaluate(*code.program, m);
  expect(result.status == ord::Status::Solved &&
             tepl::inference::check(*code.program, m, result),
         "completed program constructs and independently checks attributes");
  expect(tepl::inference::evaluate(*code.program, m, {10000, 1}).status ==
             ord::Status::ResourceLimit,
         "axis budget is not infeasibility");
  auto corrupt = *code.program;
  corrupt.replacement.back().inputs = {corrupt.replacement.size() - 1};
  expect(
      tepl::inference::evaluate(corrupt, m).status == ord::Status::Unsupported,
      "cyclic completed schedule rejected");
  corrupt = *code.program;
  corrupt.replacement.back().captures = {};
  expect(
      tepl::inference::evaluate(corrupt, m).status == ord::Status::Unsupported,
      "incorrect precomputed capture set rejected");

  // Fully explicit descriptors reuse replay, rather than existential
  // feasibility.
  p.nodes[p.rhs].automatic = false;
  auto explicit_code = ord::compile(p, tepl::inference::Mode::Validate);
  expect(explicit_code.program.has_value(), "compile explicit RHS validation");
  auto descriptor = p.nodes[p.rhs].descriptor->value;
  m.descriptors[descriptor] = {{1, 0}};
  auto checked = tepl::inference::evaluate(*explicit_code.program, m);
  expect(checked.status == ord::Status::Solved &&
             tepl::inference::check(*explicit_code.program, m, checked),
         "explicit supplied assignment passes semantic replay");
  m.descriptors[descriptor] = {{0, 1}};
  expect(
      tepl::inference::evaluate(*explicit_code.program, m).status ==
          ord::Status::NotEquivalent,
      "explicit checker rejects wrong values even when some solution exists");
  expect(!tepl::inference::check(*explicit_code.program, m, checked),
         "explicit certificate cannot substitute another feasible assignment");
  m.descriptors[descriptor] = {{0, 0}};
  expect(tepl::inference::evaluate(*explicit_code.program, m).status ==
             ord::Status::InvalidReplacement,
         "illegal explicit permutation diagnosed");
}

// Independent exhaustive positional-attribute oracle. It uses a dense
// equivalence matrix, not the solver's union-find, boundaries, or order pass.
struct Oracle {
  AxisSequence output;
  std::vector<std::vector<bool>> same;
  ord::Attributes attributes;
};
std::string signature(const AxisSequence& classes, const AxisSequence& output) {
  std::string out;
  for (auto a : classes) out += std::to_string(a) + ",";
  out += ":";
  for (auto a : output) out += std::to_string(a) + ",";
  return out;
}
std::string signature(const Oracle& n) {
  AxisSequence classes(n.same.size());
  for (std::size_t i = 0; i < n.same.size(); ++i) {
    classes[i] = i;
    for (std::size_t j = 0; j < i; ++j)
      if (n.same[i][j]) {
        classes[i] = j;
        break;
      }
  }
  auto output = n.output;
  for (auto& a : output) a = classes[a];
  return signature(classes, output);
}
std::vector<Oracle> enumerate(const Plan& p, std::size_t id,
                              const std::vector<AxisSequence>& ports,
                              std::size_t total) {
  const auto& node = p.nodes[id];
  if (node.capture) {
    Oracle leaf{ports.at(node.capture->value),
                std::vector<std::vector<bool>>(total, std::vector<bool>(total)),
                {}};
    for (std::size_t i = 0; i < total; ++i) leaf.same[i][i] = true;
    return {leaf};
  }
  const auto c = *ord::recognize(*node.definition);
  auto left = enumerate(p, node.inputs[0], ports, total);
  std::vector<Oracle> out;
  if (c.kind == ord::Contract::Kind::Permutation) {
    for (const auto& l : left) {
      List permutation(l.output.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      do {
        auto next = l;
        next.output.clear();
        for (auto i : permutation) next.output.push_back(l.output[i]);
        next.attributes[node.descriptor->value] = {permutation};
        out.push_back(std::move(next));
      } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
    return out;
  }
  for (const auto& l : left)
    for (const auto& r : enumerate(p, node.inputs[1], ports, total)) {
      struct Pair {
        std::size_t left, right;
        bool batch;
      };
      std::vector<Pair> pairs;
      std::set<std::size_t> used_right;
      std::function<void(std::size_t)> select = [&](std::size_t i) {
        if (i < l.output.size()) {
          select(i + 1);
          for (std::size_t j = 0; j < r.output.size(); ++j)
            if (!used_right.contains(j)) {
              used_right.insert(j);
              for (bool batch : {false, true}) {
                pairs.push_back({i, j, batch});
                select(i + 1);
                pairs.pop_back();
              }
              used_right.erase(j);
            }
          return;
        }
        std::vector<std::size_t> batches;
        for (std::size_t k = 0; k < pairs.size(); ++k)
          if (pairs[k].batch) batches.push_back(k);
        do {
          auto next = l;
          next.attributes.insert(r.attributes.begin(), r.attributes.end());
          for (std::size_t a = 0; a < total; ++a)
            for (std::size_t b = 0; b < total; ++b)
              next.same[a][b] = l.same[a][b] || r.same[a][b];
          List lb, rb, lk, rk;
          std::set<std::size_t> used_left, used;
          for (const auto& pair : pairs) {
            auto a = l.output[pair.left], b = r.output[pair.right];
            next.same[a][b] = next.same[b][a] = true;
            used_left.insert(pair.left);
            used.insert(pair.right);
            if (!pair.batch) {
              lk.push_back(pair.left);
              rk.push_back(pair.right);
            }
          }
          for (std::size_t k = 0; k < total; ++k)
            for (std::size_t a = 0; a < total; ++a)
              for (std::size_t b = 0; b < total; ++b)
                next.same[a][b] =
                    next.same[a][b] || (next.same[a][k] && next.same[k][b]);
          next.output.clear();
          for (auto k : batches) {
            lb.push_back(pairs[k].left);
            rb.push_back(pairs[k].right);
            next.output.push_back(l.output[pairs[k].left]);
          }
          for (std::size_t a = 0; a < l.output.size(); ++a)
            if (!used_left.contains(a)) next.output.push_back(l.output[a]);
          for (std::size_t b = 0; b < r.output.size(); ++b)
            if (!used.contains(b)) next.output.push_back(r.output[b]);
          next.attributes[node.descriptor->value] =
              dotFields(*node.definition, lb, rb, lk, rk);
          out.push_back(std::move(next));
        } while (std::next_permutation(batches.begin(), batches.end()));
      };
      select(0);
    }
  return out;
}

void exhaustive() {
  auto fixture =
      parse("rule r {(p: transpose (d: dot A B)) => (transpose (dot B A))}");
  auto transpose = fixture.nodes[fixture.lhs].definition;
  auto dot = fixture.nodes[fixture.nodes[fixture.lhs].inputs[0]].definition;
  std::size_t decisions = 0, impossible = 0;
  for (unsigned shape = 0; shape < 8; ++shape)
    for (bool reverse_tree : {false, true})
      for (bool internal_transpose : {false, true})
        for (bool output_transpose : {false, true}) {
          std::vector<std::size_t> order{0, 1, 2};
          do {
            Plan p;
            p.algebraic = true;
            for (std::size_t c = 0; c < 3; ++c) {
              Instance leaf;
              leaf.capture = tepl::core::CaptureId{c};
              p.nodes.push_back(leaf);
            }
            std::size_t descriptor = 0;
            auto op = [&](std::shared_ptr<const Definition> definition,
                          std::vector<std::size_t> children, bool automatic) {
              Instance n;
              n.definition = std::move(definition);
              n.inputs = std::move(children);
              n.automatic = automatic;
              n.descriptor = tepl::core::DescriptorId{descriptor++};
              p.nodes.push_back(n);
              return p.nodes.size() - 1;
            };
            auto source_inner = op(dot, {0, 1}, false);
            auto source_dot = op(dot, {source_inner, 2}, false);
            p.lhs = op(transpose, {source_dot}, false);
            auto inner = op(dot,
                            reverse_tree ? std::vector{order[0], order[1]}
                                         : std::vector{order[1], order[2]},
                            true);
            if (internal_transpose) inner = op(transpose, {inner}, true);
            p.rhs = op(dot,
                       reverse_tree ? std::vector{inner, order[2]}
                                    : std::vector{order[0], inner},
                       true);
            if (output_transpose) p.rhs = op(transpose, {p.rhs}, true);
            ord::Metadata m;
            std::vector<AxisSequence> ports(3);
            std::size_t total = 0;
            for (std::size_t c = 0; c < 3; ++c) {
              m.shapes[c] = {};
              if (shape & (1u << c)) {
                ports[c].push_back(total++);
                m.shapes[c].push_back(2);
              }
            }
            std::set<std::string> targets;
            auto compiled = ord::compile(p);
            expect(compiled.program.has_value(),
                   "exhaustive schema compilation succeeds");
            for (const auto& rhs : enumerate(p, p.rhs, ports, total))
              targets.insert(signature(rhs));
            for (const auto& source : enumerate(p, p.lhs, ports, total)) {
              m.descriptors = source.attributes;
              auto result = tepl::inference::evaluate(*compiled.program, m);
              bool exists = targets.contains(signature(source));
              auto predicate =
                  tepl::inference::feasibility(*compiled.program, m);
              expect(
                  predicate.status == (exists ? ord::Status::Feasible
                                              : ord::Status::Impossible) &&
                      predicate.attributes.empty() && predicate.layouts.empty(),
                  "generated predicate alone agrees with existential oracle");
              expect(result.status == (exists ? ord::Status::Solved
                                              : ord::Status::Impossible),
                     "structural decision agrees with exhaustive attributes");
              expect(signature(result.source.classes, result.source.output) ==
                         signature(source),
                     "source partition agrees with independent matrix oracle");
              expect(ord::check(p, m, result), "exhaustive witness checks");
              ++decisions;
              impossible += !exists;
            }
          } while (std::next_permutation(order.begin(), order.end()));
        }
  expect(decisions > 1000 && impossible > 100, "oracle covers both branches");
  std::cout << decisions << " exhaustive decisions, " << impossible
            << " checked obstructions\n";
}

void compilerIntegration() {
  auto p = parse(R"(
rule internal_order {
  A: [M,N] B: [K]
  (p: transpose (d: dot A B))
  => (dot (transpose A) B)
  where {
    d.attrs.lhs_contracting_dimensions == range(0)
    && d.attrs.rhs_contracting_dimensions == range(0)
    && d.attrs.lhs_batching_dimensions == range(0)
    && d.attrs.rhs_batching_dimensions == range(0)
    && p.attrs.permutation == [1,0,2]
  }
})");
  expect(p.proof.status == ProofStatus::Proved,
         "internal transpose order verifies under original conditions: " +
             p.proof.reason);
  expect(std::find(p.proof.steps.begin(), p.proof.steps.end(),
                   "law: ordered-requirements-construction") !=
             p.proof.steps.end(),
         "compiler uses the separated requirements/construction path");
  expect(p.proof.guard.empty(), "compiler adds no hidden applicability guard");
  Metadata m{{{2, 3}, {5}}, {{0, {{1, 0, 2}}}, {1, {{}, {}, {}, {}}}}};
  tepl::test::ConstructorEvaluator eval(m, p);
  expect(eval.applicable(), "declared source metadata is applicable");
  auto result = eval.construct();
  auto transpose_id = p.nodes[p.nodes[p.rhs].inputs[0]].descriptor->value;
  expect(result.at(transpose_id).at(0) == List({1, 0}),
         "generated constructor implements internal order requirement");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string error;
    auto runfiles = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    expect(bool(runfiles), error);
    std::ifstream input(runfiles->Rlocation(argv[1]));
    dialect.assign(std::istreambuf_iterator<char>(input), {});
    expect(!dialect.empty(), "load tensor dialect");
    admission();
    matricesAndDomains();
    layouts();
    multiplePairings();
    sequenceComparison();
    compiledPrograms();
    exhaustive();
    compilerIntegration();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
