#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/attr_synth/ordered/compile.h"
#include "src/core/attr_synth/synthesize.h"
#include "src/core/pipeline.h"
#include "src/core/semantics/context.h"
#include "src/core/semantics/graph.h"
#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/symbolic/expr.h"
#include "src/core/semantics/symbolic/metadata_sequences.h"
#include "src/core/syntax/parse.h"
#include "src/executable/codec.h"
#include "src/inference/evaluate.h"
#include "src/shared/catalog/axis_conditions.h"
#include "tests/index_constructor_eval.h"

namespace {
using namespace tepl::core::semantics;
namespace ord = tepl::core::attr_synth::ordered;
using tepl::inference::AxisSequence;
tepl::inference::Result solve(
    const Plan& plan, const tepl::inference::Metadata& metadata,
    std::size_t max_work = std::numeric_limits<std::size_t>::max()) {
  auto compiled = ord::compile(plan);
  if (!compiled.program) {
    tepl::inference::Result out;
    out.reason = compiled.reason;
    return out;
  }
  return tepl::inference::evaluate(*compiled.program, metadata, {max_work});
}
bool check(const Plan& plan, const tepl::inference::Metadata& metadata,
           const tepl::inference::Result& result) {
  auto compiled = ord::compile(plan);
  return compiled.program &&
         tepl::inference::check(*compiled.program, metadata, result);
}

std::size_t emission_checks = 0;
void checkEmission(const tepl::inference::Program& code,
                   const tepl::inference::Metadata& m,
                   const tepl::inference::Result& reference) {
  using namespace tepl::inference;
  using Status = tepl::inference::Status;
  using Result = tepl::inference::Result;
  auto emitted = emit(code);
  auto actual = execute(emitted, m);
  auto decision = predicate(emitted,m);
  if ((actual.status==Status::Solved && decision.status!=Status::Feasible) ||
      (actual.status!=Status::Solved && actual.status!=decision.status) || !decision.attributes.empty())
    throw std::runtime_error("emitted predicate is independent of builders");
  if (actual.status != reference.status)
    throw std::runtime_error("emission disagrees with concrete constructor");
  if (actual.status == Status::Solved) {
    auto completed = m;
    completed.descriptors.insert(actual.attributes.begin(), actual.attributes.end());
    auto replay = replayReplacement(code, completed);
    if (replay.status != Status::Analyzed)
      throw std::runtime_error("emitted attributes are illegal");
    auto same = [](const Result& source, const Result& target) {
      return source.source.classes == target.source.classes &&
             source.source.output == target.source.output;
    };
    if (code.components.empty()) {
      if (!same(actual,replay))throw std::runtime_error("emission changes network");
    } else {
      if(actual.components.size()!=code.components.size() || replay.components.size()!=code.components.size())
        throw std::runtime_error("emission lost additive component");
      for(std::size_t i=0;i<code.components.size();++i)
        if(!same(actual.components[i],replay.components[i]))throw std::runtime_error("emission merged component axes");
    }
  }
  ++emission_checks;
}

std::string dialect;
void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
Plan parse(const std::string& rule, bool defer = false) {
  auto parsed = tepl::parse(dialect + rule, "ordered.tepl");
  expect(parsed.ok(), "ordered fixture parses");
  tepl::core::InspectionOptions options{tepl::core::SemanticPolicy::kAlgebraic, true};
  options.defer_semantic_proofs = defer;
  auto inspected = tepl::core::inspect(*parsed.program, options);
  if (!inspected.ok())
    for (const auto& diagnostic : inspected.diagnostics)
      std::cerr << diagnostic.message << " in " << rule << "\n";
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
void setDot(const Plan& p, tepl::inference::Metadata& m, std::size_t id,
            List lb = {}, List rb = {}, List lk = {}, List rk = {}) {
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
  tepl::inference::Metadata m{{{0, {2, 3}}, {1, {3, 4}}, {2, {4, 5}}}, {}};
  auto inner = p.nodes[p.lhs].inputs[0];
  setDot(p, m, inner, {}, {}, {1}, {0});
  setDot(p, m, p.lhs, {}, {}, {1}, {0});
  auto result = solve(p, m);
  expect(
      result.status == tepl::inference::Status::Solved && check(p, m, result),
      "matrix reassociation has an independently checked witness");
  auto corrupt = result;
  auto root_id = p.nodes[p.rhs].descriptor->value;
  auto c = *ord::recognize(*p.nodes[p.rhs].definition);
  corrupt.attributes[root_id][c.left_contract] = {0};
  expect(!check(p, m, corrupt), "incorrect contracting positions rejected");
  corrupt = result;
  corrupt.source.classes[0] = corrupt.source.classes[1];
  expect(!check(p, m, corrupt), "corrupt source provenance rejected");
  corrupt = result;
  corrupt.requirements.at(p.rhs).mandatory_axes.clear();
  expect(!check(p, m, corrupt), "corrupt boundary certificate rejected");
  auto singleton = m;
  singleton.shapes[0][0] = 1;
  expect(solve(p, singleton).status == tepl::inference::Status::OutsideDomain,
         "singleton extent is outside the necessity proof domain");
  singleton.shapes[0][0] = 0;
  expect(solve(p, singleton).status == tepl::inference::Status::OutsideDomain,
         "zero extent is outside the necessity proof domain");
  auto invalid = m;
  invalid.shapes[1][0] = 7;
  expect(solve(p, invalid).status == tepl::inference::Status::InvalidSource,
         "mismatched source extents are not impossibility");
  invalid = m;
  invalid.descriptors.erase(p.nodes[inner].descriptor->value);
  expect(solve(p, invalid).status == tepl::inference::Status::MissingMetadata,
         "missing metadata is not impossibility");
  expect(solve(p, m, 0).status == tepl::inference::Status::ResourceLimit,
         "work exhaustion is not impossibility");
}
void layouts() {
  auto p = parse("rule r {(p: transpose (d: dot A B)) => (dot A B)}");
  tepl::inference::Metadata m{{{0, {2}}, {1, {2}}}, {}};
  setDot(p, m, p.nodes[p.lhs].inputs[0]);
  m.descriptors[p.nodes[p.lhs].descriptor->value] = {{1, 0}};
  auto result = solve(p, m);
  expect(result.status == tepl::inference::Status::Impossible &&
             check(p, m, result),
         "equal extents cannot hide an output-order obstruction");
  auto corrupt = result;
  std::swap(corrupt.obstruction->first, corrupt.obstruction->second);
  expect(!check(p, m, corrupt), "false ordering obstruction rejected");
  auto repair =
      parse("rule r {(p: transpose (d: dot A B)) => (transpose (dot A B))}");
  auto fixed = solve(repair, m);
  expect(fixed.status == tepl::inference::Status::Solved &&
             check(repair, m, fixed),
         "output transpose repairs an impossible ordering");
  auto internal =
      parse("rule r {(p: transpose (d: dot A B)) => (dot (transpose A) B)}");
  tepl::inference::Metadata im{{{0, {2, 3}}, {1, {5}}}, {}};
  setDot(internal, im, internal.nodes[internal.lhs].inputs[0]);
  im.descriptors[internal.nodes[internal.lhs].descriptor->value] = {{1, 0, 2}};
  auto reordered = solve(internal, im);
  expect(reordered.status == tepl::inference::Status::Solved &&
             check(internal, im, reordered),
         "parent order propagates to a smaller internal transpose");
  auto transpose_id = internal.nodes[internal.rhs].inputs[0];
  expect(reordered.attributes.at(internal.nodes[transpose_id].descriptor->value)
                 .at(0) == List({1, 0}),
         "internal permutation is constructed from its required subsequence");

  auto high = parse("rule r {(p: transpose A) => (transpose (transpose A))}");
  tepl::inference::Metadata hm{{{0, std::vector<std::uint64_t>(128, 2)}}, {}};
  List permutation(128);
  std::iota(permutation.rbegin(), permutation.rend(), 0);
  hm.descriptors[high.nodes[high.lhs].descriptor->value] = {permutation};
  auto high_result = solve(high, hm);
  expect(high_result.status == tepl::inference::Status::Solved &&
             check(high, hm, high_result),
         "arbitrary-length permutations and nonunique solutions");
}

void multiplePairings() {
  auto p = parse("rule r {(d: dot A B) => (transpose (dot B A))}");
  tepl::inference::Metadata m{{{0, {2, 3, 5, 7, 11}}, {1, {11, 13, 2, 5, 7}}},
                              {}};
  setDot(p, m, p.lhs, {2}, {3}, {4, 0, 3}, {0, 2, 4});
  auto result = solve(p, m);
  expect(
      result.status == tepl::inference::Status::Solved && check(p, m, result),
      "several unsorted contraction pairs and an independent batch");
  expect(result.attributes.at(p.nodes[p.rhs].descriptor->value).at(0) ==
             List({0, 2, 1}),
         "output permutation repairs free-axis blocks");

  m.shapes[0] = m.shapes[1] = std::vector<std::uint64_t>(5, 2);
  result = solve(p, m);
  expect(
      result.status == tepl::inference::Status::Solved && check(p, m, result),
      "equal extents preserve the full named-port partition");
  auto inner = p.nodes[p.rhs].inputs[0];
  auto c = *ord::recognize(*p.nodes[inner].definition);
  auto& right = result.attributes.at(p.nodes[inner].descriptor->value)
                    .at(c.right_contract);
  std::swap(right[0], right[1]);
  expect(!check(p, m, result),
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

void permutationComprehensions() {
  Plan p;
  Context c{p, 1000000};
  auto values = make(Symbol::Kind::Field, "values", 0, 0);
  auto permutation = make(Symbol::Kind::Field, "permutation", 1, 0);
  auto seq = call("range", {call("len", {permutation})});
  auto i = call("sequence_variable", {integer(1)});
  auto j = call("sequence_variable", {integer(2)});
  bool rejected = false;
  try {
    c.wellFormed(i);
  } catch (const ProofFailure&) {
    rejected = true;
  }
  expect(rejected, "free metadata comprehension variables are rejected");
  c.wellFormed(call("sequence_map", {call("range", {integer(2)}), i,
                                     call("+", {i, integer(1)})}));
  rejected = false;
  try {
    c.wellFormed(call("+", {i, integer(1)}));
  } catch (const ProofFailure&) {
    rejected = true;
  }
  expect(rejected, "a bound variable cannot escape its comprehension");
  rejected = false;
  try {
    c.wellFormed(
        call("sequence_map", {call("range", {integer(2)}), i,
                              call("unknown_metadata_function", {i})}));
  } catch (const ProofFailure&) {
    rejected = true;
  }
  expect(rejected, "unknown operations inside comprehensions are rejected");
  auto inner =
      call("sequence_map",
           {seq, j,
            call("if", {call("==", {call("index", {permutation, j}), i}), j,
                        integer(0)})});
  auto inverse_body = call("sum", {inner});
  expect(!liftMetadataMap(c, seq, i, inverse_body),
         "inverse sum is not lifted without a permutation proof");
  c.learn(call("axis_partition", {permutation, call("len", {permutation})}));
  auto inverse = liftMetadataMap(c, seq, i, inverse_body);
  expect(inverse && equal(inverse, call("inverse", {permutation})),
         "unique-match sum represents a proven inverse permutation");
  auto selected =
      call("sum",
           {call("sequence_map",
                 {seq, j,
                  call("if", {call("==", {call("index", {permutation, j}), i}),
                              call("index", {values, j}), integer(0)})})});
  expect(!liftMetadataMap(c, seq, i, selected),
         "selection is not lifted without a parallel-list length proof");
  c.learn(call("==", {call("len", {values}), call("len", {permutation})}));
  expect(bool(liftMetadataMap(c, seq, i, selected)),
         "parallel-list selection lifts when both lengths agree");
  auto wrong =
      call("sum",
           {call("sequence_map",
                 {seq, j,
                  call("if", {call("==", {call("index", {permutation, j}), i}),
                              j, integer(1)})})});
  expect(!liftMetadataMap(c, seq, i, wrong),
         "nonzero unmatched elements are not an inverse permutation");
  auto shadowed =
      call("sum",
           {call("sequence_map",
                 {seq, i,
                  call("if", {call("==", {call("index", {permutation, i}), i}),
                              i, integer(0)})})});
  expect(!liftMetadataMap(c, seq, i, shadowed),
         "shadowing the outer variable cannot be treated as inversion");
}

void compiledPrograms() {
  auto p = parse("rule r {(p: transpose A) => (transpose A)}");
  tepl::inference::Metadata m{{{0, {2, 2}}}, {}};
  m.descriptors[p.nodes[p.lhs].descriptor->value] = {{1, 0}};
  auto code = ord::compile(p);
  expect(code.program.has_value(),
         "compile a rank-independent inference program");
  auto guard = tepl::inference::feasibility(*code.program, m);
  expect(guard.status == tepl::inference::Status::Feasible &&
             guard.attributes.empty() && guard.layouts.empty() &&
             tepl::inference::check(*code.program, m, guard),
         "predicate is exact without running positional construction");
  auto result = tepl::inference::evaluate(*code.program, m);
  expect(result.status == tepl::inference::Status::Solved &&
             tepl::inference::check(*code.program, m, result),
         "completed program constructs and independently checks attributes");
  expect(tepl::inference::evaluate(*code.program, m, {10000, 1}).status ==
             tepl::inference::Status::ResourceLimit,
         "axis budget is not infeasibility");
  auto corrupt = *code.program;
  corrupt.replacement.back().inputs = {corrupt.replacement.size() - 1};
  expect(tepl::inference::evaluate(corrupt, m).status ==
             tepl::inference::Status::Unsupported,
         "cyclic completed schedule rejected");
  corrupt = *code.program;
  corrupt.replacement.back().captures = {};
  expect(tepl::inference::evaluate(corrupt, m).status ==
             tepl::inference::Status::Unsupported,
         "incorrect precomputed capture set rejected");

  // Fully explicit descriptors reuse replay, rather than existential
  // feasibility.
  p.descriptors.at(p.nodes[p.rhs].descriptor->value) = {};
  auto explicit_code = ord::compile(p, tepl::inference::Mode::Validate);
  expect(explicit_code.program.has_value(), "compile explicit RHS validation");
  auto descriptor = p.nodes[p.rhs].descriptor->value;
  m.descriptors[descriptor] = {{1, 0}};
  auto checked = tepl::inference::evaluate(*explicit_code.program, m);
  expect(checked.status == tepl::inference::Status::Solved &&
             tepl::inference::check(*explicit_code.program, m, checked),
         "explicit supplied assignment passes semantic replay");
  m.descriptors[descriptor] = {{0, 1}};
  expect(
      tepl::inference::evaluate(*explicit_code.program, m).status ==
          tepl::inference::Status::NotEquivalent,
      "explicit checker rejects wrong values even when some solution exists");
  expect(!tepl::inference::check(*explicit_code.program, m, checked),
         "explicit certificate cannot substitute another feasible assignment");
  m.descriptors[descriptor] = {{0, 0}};
  expect(tepl::inference::evaluate(*explicit_code.program, m).status ==
             tepl::inference::Status::InvalidReplacement,
         "illegal explicit permutation diagnosed");
}

// Independent exhaustive positional-attribute oracle. It uses a dense
// equivalence matrix, not the solver's union-find, boundaries, or order pass.
struct Oracle {
  AxisSequence output;
  std::vector<std::vector<bool>> same;
  tepl::inference::Attributes attributes;
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
  if (c.kind == ord::Contract::Kind::Summation) {
    for (const auto& l : left)
      for (std::size_t mask = 0; mask < (std::size_t{1} << l.output.size());
           ++mask) {
        auto next = l;
        next.output.clear();
        List dimensions;
        for (std::size_t i = 0; i < l.output.size(); ++i)
          if (mask & (std::size_t{1} << i))
            dimensions.push_back(i);
          else
            next.output.push_back(l.output[i]);
        next.attributes[node.descriptor->value] = {dimensions};
        out.push_back(std::move(next));
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
              n.descriptor = tepl::core::DescriptorId{descriptor++};
              DescriptorAttributes attributes;
              if (automatic) {
                attributes.origin = DescriptorAttributes::Origin::Constructed;
                for (std::size_t f = 0; f < n.definition->fields.size(); ++f)
                  attributes.fields.push_back(make(Symbol::Kind::Hole,
                                                   n.definition->fields[f],
                                                   n.descriptor->value, f));
              }
              p.descriptors.emplace(n.descriptor->value, std::move(attributes));
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
            tepl::inference::Metadata m;
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
              checkEmission(*compiled.program,m,result);
              bool exists = targets.contains(signature(source));
              auto predicate =
                  tepl::inference::feasibility(*compiled.program, m);
              expect(
                  predicate.status ==
                          (exists ? tepl::inference::Status::Feasible
                                  : tepl::inference::Status::Impossible) &&
                      predicate.attributes.empty() && predicate.layouts.empty(),
                  "generated predicate alone agrees with existential oracle");
              expect(result.status ==
                         (exists ? tepl::inference::Status::Solved
                                 : tepl::inference::Status::Impossible),
                     "structural decision agrees with exhaustive attributes");
              expect(signature(result.source.classes, result.source.output) ==
                         signature(source),
                     "source partition agrees with independent matrix oracle");
              expect(check(p, m, result), "exhaustive witness checks");
              ++decisions;
              impossible += !exists;
            }
          } while (std::next_permutation(order.begin(), order.end()));
        }
  expect(decisions > 1000 && impossible > 100, "oracle covers both branches");
  std::cout << decisions << " exhaustive decisions, " << impossible
            << " checked obstructions\n";
}

void summations() {
  using namespace tepl::inference;
  using Metadata = tepl::inference::Metadata;
  using Status = tepl::inference::Status;
  auto p =
      parse("rule r {(d: dot (s: reduce_sum A) B) => (reduce_sum (dot A B))}");
  expect(ord::admit(p).admitted(), "whole-axis summation is admitted");
  auto sum = p.nodes[p.lhs].inputs[0];
  Metadata m{{{0, {2, 3}}, {1, {3, 4}}}, {}};
  m.descriptors[p.nodes[sum].descriptor->value] = {{0}};
  setDot(p, m, p.lhs, {}, {}, {0}, {0});
  auto result = solve(p, m);
  expect(result.status == Status::Solved && check(p, m, result),
         "a dot temporarily retains a private axis for a later sum");
  auto dot = p.nodes[p.rhs].inputs[0];
  expect(result.requirements.at(dot).mandatory_axes.size() == 1 &&
             result.requirements.at(dot).retained_axes.size() == 2,
         "mandatory and retained axes are distinct");
  expect(result.attributes.at(p.nodes[p.rhs].descriptor->value) ==
             std::vector<List>{{0}},
         "sum removes the private row axis");
  expect(result.source.ports.size() == 4 && result.source.classes.size() == 4,
         "source replay preserves summed input-port identities");

  auto early =
      parse("rule r {(s: reduce_sum (d: dot A B)) => (reduce_sum (dot A B))}");
  Metadata em{{{0, {2}}, {1, {2}}}, {}};
  setDot(early, em, early.nodes[early.lhs].inputs[0], {0}, {0});
  em.descriptors[early.nodes[early.lhs].descriptor->value] = {{0}};
  auto er = solve(early, em);
  expect(er.status == Status::Solved && check(early, em, er) &&
             er.attributes.at(early.nodes[early.rhs].descriptor->value) ==
                 std::vector<List>{{}},
         "early contraction permits an empty sum");

  auto negative = parse("rule r {(s: reduce_sum A) => (transpose A)}");
  Metadata nm{{{0, {2, 3}}}, {}};
  nm.descriptors[negative.nodes[negative.lhs].descriptor->value] = {{0, 1}};
  auto nr = solve(negative, nm);
  expect(nr.status == Status::Impossible && check(negative, nm, nr),
         "a transpose cannot remove axes from a non-scalar input");
  auto code = *ord::compile(negative).program;
  auto analysis = analyze(code, nm);
  expect(
      tepl::inference::construct(code, analysis).status == Status::Impossible,
      "the builder enforces its feasibility precondition");
  auto generated = tepl::core::attr_synth::synthesize(
      negative, tepl::core::attr_synth::Strategy::Ordered);
  expect(generated.constructed && !generated.candidate.feasibility.empty(),
         "check/build expressions compile even without an implied guard");
  expect(verify(negative, generated.candidate).status == ProofStatus::Unknown,
         "unproved feasibility does not become an assumption");
  auto json = tepl::executable::detail::Codec<Program>::pack(code);
  auto restored = tepl::executable::detail::Codec<Program>::unpack(json);
  expect(restored == code &&
             feasibility(restored, nm).status == Status::Impossible,
         "summation instructions and adapters survive serialization");
  expect(tepl::core::completed::axisWhere(code).find("axis_feasible") !=
             std::string::npos,
         "generated checks include retained-set feasibility");
  nm.descriptors[negative.nodes[negative.lhs].descriptor->value] = {{0, 0}};
  expect(solve(negative, nm).status == Status::InvalidSource,
         "duplicate reduction dimensions are illegal source metadata");
  auto identity = parse(
      "rule r {A: [M,N] (s: reduce_sum A) => (reduce_sum A) where "
      "{s.attrs.dimensions == range(0)}}");
  auto identity_recipe = tepl::core::attr_synth::synthesize(
      identity, tepl::core::attr_synth::Strategy::Ordered);
  auto identity_proof = verify(identity, identity_recipe.candidate);
  expect(identity_recipe.constructed &&
             identity_proof.status == ProofStatus::Proved,
         "ordered summation candidates can pass independent verification: " +
             identity_proof.reason);
  auto renamed = *p.nodes[sum].definition;
  renamed.fields = {"axes_to_remove"};
  for (auto& group : renamed.groups) group.name = "renamed";
  expect(ord::recognize(renamed).has_value(), "sum recognition ignores names");
  renamed.assertions.push_back(renamed.output);
  expect(!ord::recognize(renamed), "restricted sum axes are outside admission");
}

void guardIndependentFeasibility() {
  auto p = parse(
      "rule r {A: [M,N] (s: reduce_sum A) => (transpose A) where "
      "{s.attrs.dimensions == range(0)}}");
  auto generated = tepl::core::attr_synth::synthesize(
      p, tepl::core::attr_synth::Strategy::Ordered);
  expect(generated.constructed, "guard-independent feasibility compiles");
  auto emitted = ord::compile(p).emission;
  expect(generated.metadata_program && emitted &&
             *generated.metadata_program == *emitted,
         "symbolic and native realizations share the exact constructed IR");
  auto limited = tepl::core::attr_synth::synthesize(
      p, tepl::core::attr_synth::Strategy::Ordered, 0);
  expect(!limited.constructed && limited.metadata_program &&
             *limited.metadata_program == *emitted,
         "bounded specialization cannot prevent or discard structural emission");
  auto domain = p;
  domain.assumptions.clear();
  Context coordinates{domain, 1000000};
  coordinates.sourceScope(domain.lhs);
  lowerSource(coordinates);
  RuleProof recipe;
  recipe.constructors = generated.candidate.constructors;
  for (const auto& [_, group] : coordinates.source().groups)
    recipe.axis_definitions.push_back(group);
  for (const auto& dimensions : {List{}, List{0}, List{0, 1}}) {
    Metadata metadata{{{2, 3}}, {}};
    metadata.descriptors[p.nodes[p.lhs].descriptor->value] = {dimensions};
    tepl::test::ConstructorEvaluator evaluator(metadata, recipe);
    bool feasible = true;
    for (const auto& condition : generated.candidate.feasibility)
      feasible &= std::get<bool>(evaluator.eval(condition));
    expect(feasible == dimensions.empty(),
           "the predicate stays exact outside the written guard");
    tepl::inference::Metadata native;
    native.shapes[0].assign(metadata.shapes.at(0).begin(), metadata.shapes.at(0).end());
    native.descriptors = metadata.descriptors;
    auto decision = tepl::inference::predicate(*generated.metadata_program, native);
    expect((decision.status == tepl::inference::Status::Feasible) == feasible,
           "symbolic specialization preserves the native IR predicate");
  }
}

void sharedIRPairedReindexing() {
  using namespace tepl::inference;
  using Status = tepl::inference::Status;
  auto p = parse("rule r {(d: dot A B) => (dot A B)}", true);
  auto generated = tepl::core::attr_synth::synthesize(
      p, tepl::core::attr_synth::Strategy::Ordered);
  expect(generated.constructed && generated.metadata_program,
         "paired-reindexing fixture emits and specializes the shared program");
  tepl::inference::Metadata metadata{{{0, {2, 3, 4}}, {1, {4, 3, 5}}}, {}};
  setDot(p, metadata, p.lhs, {}, {}, {2, 1}, {0, 1});
  Context source{p, 1000000};
  source.sourceScope(p.lhs);
  lowerSource(source);
  RuleProof recipe;
  recipe.constructors = generated.candidate.constructors;
  for (const auto& [_, group] : source.source().groups)
    recipe.axis_definitions.push_back(group);
  tepl::core::semantics::Metadata symbolic_metadata{
      {{2, 3, 4}, {4, 3, 5}}, metadata.descriptors};
  tepl::test::ConstructorEvaluator symbolic(symbolic_metadata, recipe);
  auto attributes = symbolic.construct();
  auto native = execute(*generated.metadata_program, metadata);
  expect(native.status == Status::Solved, "native residual program executes");
  auto descriptor = p.nodes[p.rhs].descriptor->value;
  auto contract = *ord::recognize(*p.nodes[p.rhs].definition);
  expect(attributes.at(descriptor).at(contract.left_contract) !=
             native.attributes.at(descriptor).at(contract.left_contract),
         "specialization exercises a genuine paired contraction permutation");
  metadata.descriptors.insert(attributes.begin(), attributes.end());
  auto replacement = replayReplacement(generated.metadata_program->original,
                                       metadata);
  expect(replacement.status == Status::Analyzed &&
             replacement.source.classes == native.source.classes &&
             replacement.source.output == native.source.output,
         "paired reindexing preserves connections and ordered output on anisotropic axes");
}

void exhaustiveSummations() {
  using Status = tepl::inference::Status;
  std::size_t decisions = 0, impossible = 0;
  for (const auto& rule :
       {"rule r {(dot (reduce_sum A) B) => (reduce_sum (dot A B))}",
        "rule r {(reduce_sum (dot A B)) => (dot (reduce_sum A) B)}",
        "rule r {(reduce_sum (transpose A)) => (transpose (reduce_sum A))}",
        "rule r {(transpose (reduce_sum A)) => (reduce_sum (transpose A))}",
        "rule r {(reduce_sum A) => (transpose A)}",
        "rule r {(reduce_sum (dot A B)) => (reduce_sum (dot B A))}"}) {
    auto p = parse(rule);
    auto code = *ord::compile(p).program;
    auto generated = tepl::core::attr_synth::synthesize(
        p, tepl::core::attr_synth::Strategy::Ordered);
    expect(generated.constructed,
           "rank-independent sum recipe compiles: " + generated.reason);
    Context coordinates{p, 1000000};
    coordinates.sourceScope(p.lhs);
    lowerSource(coordinates);
    RuleProof recipe;
    recipe.constructors = generated.candidate.constructors;
    for (const auto& [_, group] : coordinates.source().groups)
      recipe.axis_definitions.push_back(group);
    for (unsigned rank_a = 0; rank_a <= 2; ++rank_a)
      for (unsigned rank_b = 0;
           rank_b <= (code.source.back().captures.size() == 2 ? 2u : 0u);
           ++rank_b) {
        tepl::inference::Metadata metadata;
        std::vector<AxisSequence> ports(2);
        std::size_t total = 0;
        for (std::size_t c = 0; c < code.source.back().captures.size(); ++c) {
          auto rank = c == 0 ? rank_a : rank_b;
          metadata.shapes[c] = std::vector<std::uint64_t>(rank, 2);
          for (unsigned i = 0; i < rank; ++i) ports[c].push_back(total++);
        }
        std::set<std::string> targets;
        for (const auto& rhs : enumerate(p, p.rhs, ports, total))
          targets.insert(signature(rhs));
        for (const auto& source : enumerate(p, p.lhs, ports, total)) {
          metadata.descriptors = source.attributes;
          auto result = tepl::inference::evaluate(code, metadata);
          checkEmission(code,metadata,result);
          bool exists = targets.contains(signature(source));
          expect(
              (result.status == Status::Solved) == exists &&
                  tepl::inference::check(code, metadata, result),
              "sum feasibility agrees with exhaustive positional assignments");
          Metadata symbolic_metadata;
          for (const auto& [_, shape] : metadata.shapes)
            symbolic_metadata.shapes.emplace_back(shape.begin(), shape.end());
          symbolic_metadata.descriptors = metadata.descriptors;
          tepl::test::ConstructorEvaluator evaluator(symbolic_metadata, recipe);
          bool feasible = true;
          for (const auto& condition : generated.candidate.feasibility)
            feasible &= std::get<bool>(evaluator.eval(condition));
          expect(feasible == exists,
                 "symbolic and concrete sum predicates agree");
          if (feasible) {
            auto attributes = evaluator.construct();
            auto completed = metadata;
            completed.descriptors.insert(attributes.begin(), attributes.end());
            auto replay = tepl::inference::replayReplacement(code, completed);
            expect(
                replay.status == Status::Analyzed &&
                    replay.source.classes == result.source.classes &&
                    replay.source.output == result.source.output,
                "symbolic sum builders preserve the concrete source network");
          }
          ++decisions;
          impossible += !exists;
        }
      }
  }
  expect(decisions > 100 && impossible > 0,
         "sum oracle covers positive and negative cases");
  std::cout << decisions << " exhaustive sum decisions, " << impossible
            << " checked obstructions\n";
}

void additiveComponents() {
  using namespace tepl::inference;
  using Metadata = tepl::inference::Metadata;
  using Status = tepl::inference::Status;
  auto p = parse("rule r {(d: dot (add A B) C) => (add (dot A C) (dot B C))}");
  auto compiled = ord::compile(p);
  expect(
      compiled.program && compiled.program->components.size() == 2,
      "distribution compiles into two matched components: " + compiled.reason);
  Metadata m{{{0, {2, 3}}, {1, {2, 3}}, {2, {3, 4}}}, {}};
  setDot(p, m, p.lhs, {}, {}, {1}, {0});
  auto result = evaluate(*compiled.program, m);
  checkEmission(*compiled.program,m,result);
  expect(
      result.status == Status::Solved && check(p, m, result),
      "distribution constructs and independently checks component witnesses: " +
          result.reason);
  expect(result.components.size() == 2 &&
             result.components[0].source.ports.size() == 4 &&
             result.components[1].source.ports.size() == 4,
         "overlapping captures retain separate component networks");
  auto predicate = feasibility(*compiled.program, m);
  expect(predicate.status == Status::Feasible && predicate.attributes.empty() &&
             check(p, m, predicate),
         "component predicates are conjoined");
  auto analysis = analyze(*compiled.program, m);
  auto built = tepl::inference::construct(*compiled.program, analysis);
  expect(built.status == Status::Solved && check(p, m, built),
         "component analysis is reusable by the builder");
  auto generated = tepl::core::attr_synth::synthesize(
      p, tepl::core::attr_synth::Strategy::Ordered);
  expect(generated.constructed && generated.candidate.constructors.size() == 2,
         "symbolic component builders compile: " + generated.reason);
  auto proof = verify(p, generated.candidate);
  expect(
      proof.status == ProofStatus::Proved,
      "original verifier proves the distributed constructors: " + proof.reason);
  auto json = tepl::executable::detail::Codec<Program>::pack(*compiled.program);
  auto restored = tepl::executable::detail::Codec<Program>::unpack(json);
  expect(restored == *compiled.program &&
             tepl::inference::check(restored, m, evaluate(restored, m)),
         "component scopes and captured fields survive serialization");
  auto corrupt = restored;
  corrupt.components[1].source.back().fields = {0, 1, 2, 3};
  if (corrupt == restored)
    corrupt.components[1].source.back().fields = {1, 0, 2, 3};
  expect(evaluate(corrupt, m).status == Status::Unsupported,
         "serialized expansion is checked against original schedules");
  m.shapes[1] = {7, 3};
  expect(evaluate(restored, m).status == Status::InvalidSource,
         "every original addition must have matching operand shapes");
  for (const auto& rule : {"rule r {(add A A) => (add A A)}",
                           "rule r {(add (dot A B) C) => (add A (dot B C))}",
                           "rule r {(dot (add A B) C) => (dot (add A B) C)}",
                           "rule r {(add (dot A A) B) => (add (dot A A) B)}"})
    expect(!ord::admit(parse(rule)).admitted(),
           "unsupported additive component structure is rejected");
  auto overlap = parse(
      "rule r {(add (x: dot A B) (y: dot A C)) => (add (transpose (dot B A)) "
      "(transpose (dot C A)))}");
  Metadata om{{{0, {2, 2}}, {1, {2, 2}}, {2, {2, 2}}}, {}};
  setDot(overlap, om, overlap.nodes[overlap.lhs].inputs[0], {}, {}, {1}, {0});
  setDot(overlap, om, overlap.nodes[overlap.lhs].inputs[1], {}, {}, {0}, {0});
  auto different = solve(overlap, om);
  expect(different.status == Status::Solved && check(overlap, om, different) &&
             different.source.ports.empty() &&
             different.components[0].source.classes !=
                 different.components[1].source.classes,
         "shared tensors do not identify partitions in different summands");
  auto explicit_plan = overlap;
  for (const auto& [descriptor, _] : different.attributes)
    explicit_plan.descriptors.at(descriptor) = {};
  auto explicit_code = *ord::compile(explicit_plan, Mode::Validate).program;
  om.descriptors.insert(different.attributes.begin(),
                        different.attributes.end());
  auto explicit_result = evaluate(explicit_code, om);
  expect(explicit_result.status == Status::Solved &&
             tepl::inference::check(explicit_code, om, explicit_result),
         "explicit additive attributes are compared in separate networks");
  auto replayed = replayReplacement(explicit_code, om);
  expect(replayed.status == Status::Analyzed && replayed.source.ports.empty() &&
             replayed.components.size() == 2,
         "direct RHS replay preserves additive component scopes");
  const auto& changed = explicit_code.components[0].replacement;
  auto dot_instruction =
      std::find_if(changed.begin(), changed.end(),
                   [](const auto& n) { return n.kind == Kind::Contraction; });
  auto& bad = om.descriptors.at(
      *dot_instruction->descriptor)[dot_instruction->fields[2]];
  bad[0] = 1 - bad[0];
  expect(evaluate(explicit_code, om).status == Status::NotEquivalent,
         "matching output shapes do not hide a wrong component partition");
  auto sum_distribution = parse(
      "rule r {(s: reduce_sum (add A B)) => (add (reduce_sum A) (reduce_sum "
      "B))}");
  Metadata summed{{{0, {2, 3}}, {1, {2, 3}}}, {}};
  summed.descriptors[sum_distribution.nodes[sum_distribution.lhs]
                         .descriptor->value] = {{0}};
  auto sr = solve(sum_distribution, summed);
  expect(sr.status == Status::Solved && check(sum_distribution, summed, sr),
         "summation distributes through separated components");
  summed.shapes[1] = {7, 3};
  expect(
      solve(sum_distribution, summed).status == Status::InvalidSource,
      "matching reduced output shapes cannot hide an illegal inner addition");
  auto commuted = parse("rule r {(add A B) => (add B A)}");
  Metadata cm{{{0, {2, 3}}, {1, {2, 3}}}, {}};
  auto cr = solve(commuted, cm);
  expect(
      cr.status == Status::Solved && cr.attributes.empty() &&
          check(commuted, cm, cr),
      "attribute-free components match by capture set rather than sum order");
  auto product = parse(
      "rule r {(d: dot (add A B) (add C D)) => (add (add (dot A C) (dot A D)) "
      "(add (dot B C) (dot B D)))}");
  auto expanded = ord::compile(product);
  expect(expanded.program && expanded.program->components.size() == 4,
         "bilinear expansion produces four distinct components");
  for (const auto& component : expanded.program->components)
    expect(component.source.back().descriptor ==
               product.nodes[product.lhs].descriptor->value,
           "distribution preserves captured source descriptor identity");
  auto shared = p;
  auto left = shared.nodes[shared.rhs].inputs[0];
  auto right = shared.nodes[shared.rhs].inputs[1];
  shared.nodes[right].descriptor = shared.nodes[left].descriptor;
  expect(!ord::admit(shared).admitted(),
         "RHS component holes remain independent");
}

void totalEmission() {
  using namespace tepl::inference;
  using Metadata = tepl::inference::Metadata;
  using Status = tepl::inference::Status;
  auto p=parse("rule r {(transpose (transpose A)) => (transpose A)}");
  auto compiled=ord::compile(p);
  expect(bool(compiled.emission),"core compilation emits a closed DAG");
  auto without_guard=p;without_guard.assumptions.clear();
  expect(compiled.emission==ord::compile(without_guard).emission,"emission does not depend on guard proof");
  for(unsigned rank: {0u,1u,2u,8u,32u,128u}) {
    Metadata m{{{0,std::vector<std::uint64_t>(rank,2)}},{}};
    List reverse;for(unsigned i=rank;i-->0;)reverse.push_back(i);
    for(const auto& n:compiled.program->source)if(n.descriptor)m.descriptors[*n.descriptor]={reverse};
    checkEmission(*compiled.program,m,evaluate(*compiled.program,m));
    expect(execute(*compiled.emission,m).status==Status::Solved,"same emitted DAG executes at every tested rank");
    expect(execute(*compiled.emission,m,{0}).status==Status::ResourceLimit,"budget exhaustion is not infeasibility");
    if(rank) {m.shapes[0][0]=1;expect(execute(*compiled.emission,m).status==Status::OutsideDomain,"singleton outside complete domain");}
  }
  // Emission is syntactic even when the written guard prover is incomplete.
  auto sum=parse("rule r {(dot (reduce_sum A) B) => (reduce_sum (dot A B))}");
  auto emitted=ord::compile(sum);
  expect(bool(emitted.emission),"sum movement emits without symbolic set proofs");
  auto addition=parse("rule r {(add (transpose A) B) => (add A B)}",true);
  auto ac=ord::compile(addition);
  Metadata am{{{0,{2,2}},{1,{2,2}}},{}};
  for(const auto& i:ac.program->source)if(i.descriptor)am.descriptors[*i.descriptor]={{1,0}};
  auto ar=execute(*ac.emission,am);
  expect(ar.status==Status::Impossible && ar.attributes.empty(),"component checks are conjoined before returning attributes");
  checkEmission(*ac.program,am,evaluate(*ac.program,am));
  std::string left="A0",right="A11";
  for(int i=1;i<=11;++i)left="(dot "+left+" A"+std::to_string(i)+")";
  for(int i=10;i>=0;--i)right="(dot A"+std::to_string(i)+" "+right+")";
  auto chain=parse("rule r {"+left+" => "+right+"}",true);
  expect(chain.proof.status==ProofStatus::Unknown,"emission-only preparation has no acceptance proof");
  auto cc=ord::compile(chain);
  expect(bool(cc.emission),"long chain emits without theorem-prover queries");
  Metadata cm;
  for(auto cap:cc.program->source.back().captures)cm.shapes[cap]={2,2,2};
  for(const auto& i:cc.program->source)if(i.descriptor){auto& fields=cm.descriptors[*i.descriptor];fields.resize(4);fields[i.fields[0]]={0};fields[i.fields[1]]={0};fields[i.fields[2]]={2};fields[i.fields[3]]={1};}
  checkEmission(*cc.program,cm,evaluate(*cc.program,cm));
  auto bad=parse("rule r {(dot A A) => (dot A A)}");
  expect(!ord::compile(bad).emission,"repeated-input component rejected before emission");
}

void compilerIntegration() {
  auto sum_rule = parse(R"(
rule sum_rows {
  A: [M,K] B: [K,N]
  (d: dot (s: reduce_sum A) B) => (reduce_sum (dot A B))
  where {
    s.attrs.dimensions == [0]
    && d.attrs.lhs_contracting_dimensions == [0]
    && d.attrs.rhs_contracting_dimensions == [0]
    && d.attrs.lhs_batching_dimensions == range(0)
    && d.attrs.rhs_batching_dimensions == range(0)
  }
})");
  auto generated_sum = tepl::core::attr_synth::synthesize(
      sum_rule, tepl::core::attr_synth::Strategy::Ordered);
  expect(generated_sum.constructed,
         "ordered sum constructors compile: " + generated_sum.reason);
  // Evaluate the symbolic recipe independently of the concrete solver. The
  // bounded verifier may still leave a valid guarded rewrite Unknown.
  auto metadata_domain = sum_rule;
  metadata_domain.assumptions.clear();
  Context source{metadata_domain, 1000000};
  source.sourceScope(sum_rule.lhs);
  lowerSource(source);
  RuleProof recipe;
  recipe.constructors = generated_sum.candidate.constructors;
  recipe.guard = generated_sum.candidate.feasibility;
  for (const auto& [_, group] : source.source().groups)
    recipe.axis_definitions.push_back(group);
  auto source_sum = sum_rule.nodes[sum_rule.lhs].inputs[0];
  Metadata sm{{{2, 3}, {3, 4}}, {}};
  sm.descriptors[sum_rule.nodes[source_sum].descriptor->value] = {{0}};
  sm.descriptors[sum_rule.nodes[sum_rule.lhs].descriptor->value] =
      dotFields(*sum_rule.nodes[sum_rule.lhs].definition, {}, {}, {0}, {0});
  tepl::test::ConstructorEvaluator sum_eval(sm, recipe);
  expect(sum_eval.applicable(),
         "symbolic sum feasibility holds for legal metadata");
  auto fields = sum_eval.construct();
  auto target_dot = sum_rule.nodes[sum_rule.rhs].inputs[0];
  expect(
      fields.at(sum_rule.nodes[sum_rule.rhs].descriptor->value) ==
              std::vector<List>{{0}} &&
          fields.at(sum_rule.nodes[target_dot].descriptor->value) ==
              dotFields(*sum_rule.nodes[target_dot].definition, {}, {}, {1},
                        {0}),
      "symbolic builders generate the reduction and paired contraction axes");
  expect(generated_sum.candidate.guard.empty(),
         "no implicit runtime guard is added");
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
    permutationComprehensions();
    compiledPrograms();
    exhaustive();
    summations();
    guardIndependentFeasibility();
    sharedIRPairedReindexing();
    exhaustiveSummations();
    additiveComponents();
    totalEmission();
    compilerIntegration();
    std::cout << emission_checks << " emitted programs checked against concrete construction and independent replay\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
