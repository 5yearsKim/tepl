#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/pipeline.h"
#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/symbolic/normalize.h"
#include "src/core/semantics/synthesis/index_inference.h"
#include "src/core/syntax/parse.h"
#include "tests/index_graph_eval.h"

namespace {
using namespace tepl::core::semantics;
using namespace tepl::test;
std::string dialect;
void expect(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
Plan plan(const std::string& rule, std::string definitions = {},
          tepl::core::SemanticPolicy policy =
              tepl::core::SemanticPolicy::kAlgebraic) {
  auto parsed =
      tepl::parse((definitions.empty() ? dialect : definitions) + rule);
  expect(parsed.ok(), "slice fixture parses");
  auto checked = tepl::core::inspect(*parsed.program, {policy, true});
  for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  expect(checked.ok(), "slice fixture checks");
  return *checked.program->rules[0].semantic_plan;
}
void proved(const Plan& p) {
  expect(p.proof.status == ProofStatus::Proved, p.proof.reason);
  expect(p.proof.guard.empty(), "slice needs no inferred guard");
  expect(verify(p, p.proof).status == ProofStatus::Proved,
         "independent slice verification");
}
struct Slice {
  List starts, sizes, strides;
};
// Physical reference uses ordinary tensor positions and no compiler machinery.
Dense reference(const Dense& input, const Slice& config) {
  Dense out{config.sizes, {}};
  enumerate(out.shape, [&](const List& at) {
    List position;
    for (std::size_t i = 0; i < at.size(); ++i)
      position.push_back(config.starts[i] + config.strides[i] * at[i]);
    out.values.push_back(input.at(position));
  });
  return out;
}
using Attributes = std::map<std::string, std::vector<List>>;
std::vector<List> fields(const Slice& config) {
  return {config.starts, config.sizes, config.strides};
}
Metadata metadata(const Plan& p, const std::vector<Dense>& inputs,
                  const Attributes& attrs) {
  Metadata out;
  for (const auto& input : inputs) out.shapes.push_back(input.shape);
  for (const auto& n : p.nodes)
    if (n.descriptor && n.definition && !n.automatic)
      out.descriptors[n.descriptor->value] =
          attrs.at(p.descriptor_names.at(n.descriptor->value));
  return out;
}
void compare(const Plan& p, const std::vector<Dense>& inputs,
             const Attributes& attrs, const Dense& expected) {
  auto concrete = metadata(p, inputs, attrs);
  Context c{p, 1000000};
  c.sourceScope(p.lhs);
  for (const auto& [descriptor, values] : p.proof.constructors)
    for (std::size_t f = 0; f < values.size(); ++f)
      c.assignAttribute(descriptor, f, values[f]);
  Lowerer first{c};
  first.lower(p.lhs);
  Lowerer source{c};
  auto lhs = source.lower(p.lhs);
  RuleProof coordinates;
  for (const auto& [id, axes] : c.source().groups)
    coordinates.axis_definitions.push_back(axes);
  ConstructorEvaluator m(concrete, coordinates);
  GraphEvaluator interpreter{m, inputs};
  auto actual = interpreter.tensor(lhs);
  expect(actual.shape == expected.shape && actual.values == expected.values,
         "LHS slice equals physical reference");
  c.enableSetReasoning();
  auto correspondence = correspond(c, lhs, normalize(c, lhs.value),
                                   normalize(c, arithmeticSkeleton(p, p.rhs)));
  Lowerer target{c, &correspondence.anchors};
  target.interfaces = &correspondence.interfaces;
  auto rhs = target.lower(p.rhs);
  actual = interpreter.tensor(rhs);
  expect(actual.shape == expected.shape && actual.values == expected.values,
         "inferred RHS slice equals physical reference");
  // Evaluate constructors as standalone metadata programs, independently of
  // target lowering. In every fixture their size must be the actual output.
  for (const auto& [descriptor, values] : p.proof.constructors)
    expect(std::get<List>(m.eval(values[1])) == expected.shape,
           "inferred output sizes");
}
bool legal(const Plan& p, const List& shape, const Slice& config) {
  Context c{p, 1000000};
  c.sourceScope(p.lhs);
  Lowerer source{c};
  source.lower(p.lhs);
  RuleProof coordinates;
  for (const auto& [id, axes] : c.source().groups)
    coordinates.axis_definitions.push_back(axes);
  Metadata concrete{{shape}, {{0, fields(config)}}};
  ConstructorEvaluator m(concrete, coordinates);
  try {
    for (const auto& f : source.obligations)
      if (!std::get<bool>(m.eval(f))) return false;
  } catch (const std::exception&) {
    return false;
  }
  return true;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    expect(argc == 2, "dialect argument");
    std::string error;
    auto runfiles = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    std::ifstream file(runfiles->Rlocation(argv[1]));
    dialect = {std::istreambuf_iterator<char>(file), {}};
    expect(!dialect.empty(), "dialect loads");
    auto distribution =
        plan("rule r {(s: slice (add A B)) => (add (slice A) (slice B))}");
    auto composition =
        plan("rule r {(outer: slice (inner: slice A)) => (slice A)}");
    auto zero_start_composition = plan(
        "rule r {(outer: slice (inner: slice A)) => (slice A) where "
        "{inner.attrs.starts == [0] && outer.attrs.starts == [0] && "
        "inner.attrs.strides == [1] && outer.attrs.strides == [1]}}",
        {}, tepl::core::SemanticPolicy::kStrict);
    auto unit_composition = plan(
        "rule r {(outer: slice (inner: slice A)) => (slice A) where "
        "{inner.attrs.strides == [1 for d in inner.attrs.sizes] && "
        "outer.attrs.strides == inner.attrs.strides}}");
    auto rank_one_unit = plan(
        "rule r {A:[N] (outer: slice (inner: slice A)) => (slice A) where "
        "{inner.attrs.strides == [1] && outer.attrs.strides == [1]}}");
    auto permuted = plan(
        "rule r {(s: slice (p: transpose (add A B))) => (add (slice "
        "(transpose[p.attrs] A)) (slice (transpose[p.attrs] B)))}");
    auto chain = plan(
        "rule r {(outer: slice (middle: slice (inner: slice A))) => (slice "
        "A)}");
    auto identity = plan("rule r {(s: slice A) => (slice A)}");
    proved(distribution);
    proved(composition);
    proved(zero_start_composition);
    proved(unit_composition);
    proved(rank_one_unit);
    proved(permuted);
    proved(chain);
    proved(identity);
    proved(plan("rule r {(outer: slice (inner: slice A)) => (slice A)}", {},
                tepl::core::SemanticPolicy::kStrict));
    for (std::size_t rank = 0; rank <= 3; ++rank) {
      List shape, starts, sizes, strides;
      for (std::size_t i = 0; i < rank; ++i) {
        starts.push_back(1 + i % 2);
        sizes.push_back(2 + i % 2);
        strides.push_back(1 + i % 2);
        shape.push_back(starts.back() + strides.back() * (sizes.back() - 1) +
                        1);
      }
      Slice config{starts, sizes, strides};
      auto a = data(shape, 1), b = data(shape, 2);
      compare(distribution, {a, b}, {{"s", fields(config)}},
              reference(add(a, b), config));
      List permutation(rank);
      std::iota(permutation.begin(), permutation.end(), 0);
      std::reverse(permutation.begin(), permutation.end());
      auto transposed = transpose(add(a, b), permutation);
      Slice reversed = config;
      std::reverse(reversed.starts.begin(), reversed.starts.end());
      std::reverse(reversed.sizes.begin(), reversed.sizes.end());
      std::reverse(reversed.strides.begin(), reversed.strides.end());
      compare(permuted, {a, b}, {{"s", fields(reversed)}, {"p", {permutation}}},
              reference(transposed, reversed));
      Slice inner{List(rank, 1), List(rank, 5), List(rank, 2)},
          outer{List(rank, 1), List(rank, 2), List(rank, 2)};
      auto input = data(List(rank, 10), 3);
      compare(composition, {input},
              {{"inner", fields(inner)}, {"outer", fields(outer)}},
              reference(reference(input, inner), outer));
      Slice unit_inner{List(rank, 1), List(rank, 5), List(rank, 1)},
          unit_outer{List(rank, 1), List(rank, 2), List(rank, 1)};
      compare(unit_composition, {input},
              {{"inner", fields(unit_inner)}, {"outer", fields(unit_outer)}},
              reference(reference(input, unit_inner), unit_outer));
      if (rank == 1)
        compare(rank_one_unit, {input},
                {{"inner", fields(unit_inner)}, {"outer", fields(unit_outer)}},
                reference(reference(input, unit_inner), unit_outer));
      if (rank == 1) {
        Slice zero_inner{{0}, {5}, {1}}, zero_outer{{0}, {2}, {1}};
        compare(zero_start_composition, {input},
                {{"inner", fields(zero_inner)}, {"outer", fields(zero_outer)}},
                reference(reference(input, zero_inner), zero_outer));
      }
      Slice deep_inner{List(rank, 1), List(rank, 8), List(rank, 2)},
          middle{List(rank, 1), List(rank, 4), List(rank, 2)};
      auto deep = data(List(rank, 16), 4);
      compare(chain, {deep},
              {{"inner", fields(deep_inner)},
               {"middle", fields(middle)},
               {"outer", fields(outer)}},
              reference(reference(reference(deep, deep_inner), middle), outer));
    }
    expect(legal(identity, {8}, {{1}, {4}, {2}}),
           "last element at input boundary is legal");
    expect(legal(identity, {}, {{}, {}, {}}), "rank-zero slice is legal");
    expect(!legal(identity, {8}, {{-1}, {3}, {1}}), "negative start rejected");
    expect(!legal(identity, {8}, {{1}, {0}, {1}}),
           "empty slice explicitly unsupported");
    expect(!legal(identity, {8}, {{1}, {3}, {0}}), "zero stride rejected");
    expect(!legal(identity, {8}, {{1}, {3}, {-1}}), "negative stride rejected");
    expect(!legal(identity, {8}, {{2}, {4}, {2}}), "out of bounds rejected");
    expect(!legal(identity, {8}, {{1, 2}, {3}, {1}}),
           "attribute length mismatch rejected");
    auto inferred_distribution =
        plan("rule r {(s: slice (add A B)) => (add (slice A) (slice B))}");
    proved(inferred_distribution);
    auto wrong = plan(
        "rule r {(s: slice A) => (out: slice A) derive "
        "{out.attrs={..s.attrs,starts:s.attrs.strides};}}");
    expect(wrong.proof.status == ProofStatus::Unknown,
           "wrong start is not proved");
    auto wrong_sizes = plan(
        "rule r {(s: slice A) => (out: slice A) derive "
        "{out.attrs={..s.attrs,sizes:[1]};}}");
    expect(wrong_sizes.proof.status == ProofStatus::Unknown,
           "wrong output size is not proved");
    for (auto field : {0, 1, 2}) {
      auto tampered = identity.proof;
      auto& values = tampered.constructors.begin()->second;
      values[field] = call("offset_extents", {values[field], integer(1)});
      expect(verify(identity, tampered).status == ProofStatus::Unknown,
             "independent verifier rejects tampered slice attribute");
    }
    auto renamed = dialect;
    auto at = renamed.find("op slice(");
    renamed.replace(at, 9, "op crop(");
    at = renamed.find("starts: index[];");
    renamed.replace(at, 16, "begins: index[];");
    for (std::size_t position = 0;
         (position = renamed.find("attrs.starts", position)) !=
         std::string::npos;) {
      renamed.replace(position, 12, "attrs.begins");
      position += 12;
    }
    proved(plan("rule r {(s: crop (add A B)) => (add (crop A) (crop B))}",
                renamed));
    Context arithmetic{identity, 1000000};
    auto x = make(Symbol::Kind::Field, "x", 0, 0),
         y = make(Symbol::Kind::Field, "y", 0, 1);
    arithmetic.learn(call("==", {call("len", {x}), call("len", {y})}));
    arithmetic.learn(call("extent_le", {x, y}));
    expect(!extentBoundArithmetic(
               arithmetic,
               call("scale_extents", {x, make(Symbol::Kind::Integer, "-1")}),
               call("scale_extents", {y, make(Symbol::Kind::Integer, "-1")}),
               QueryMode::Entailment),
           "negative scaling cannot preserve inequality direction");
    auto multiplier = make(Symbol::Kind::Field, "m", 0, 2);
    arithmetic.learn(call("==", {call("len", {x}), call("len", {multiplier})}));
    expect(!extentBoundArithmetic(
               arithmetic, call("extent_mul", {multiplier, x}),
               call("extent_mul", {multiplier, y}), QueryMode::Entailment),
           "unknown-sign scaling cannot preserve inequality");
    arithmetic.learn(call("is_valid_extents", {multiplier}));
    expect(extentBoundArithmetic(
               arithmetic, call("extent_mul", {multiplier, x}),
               call("extent_mul", {multiplier, y}), QueryMode::Entailment),
           "nonnegative scaling preserves inequality");
    std::cout << "slice tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
