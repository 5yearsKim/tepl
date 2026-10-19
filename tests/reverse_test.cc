#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/pipeline.h"
#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/symbolic/normalize.h"
#include "src/core/syntax/parse.h"
#include "tests/index_graph_eval.h"

namespace {
using namespace tepl::core::semantics;
using namespace tepl::test;
std::string dialect;
void expect(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
tepl::core::InspectionResult analyze(const std::string& rule,
                                     bool algebraic = true,
                                     const std::string& definitions = {}) {
  auto parsed =
      tepl::parse((definitions.empty() ? dialect : definitions) + rule);
  for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  expect(parsed.ok(), "reverse fixture parses");
  return tepl::core::inspect(*parsed.program,
                             {algebraic ? tepl::core::SemanticPolicy::kAlgebraic
                                        : tepl::core::SemanticPolicy::kStrict,
                              true});
}
Plan plan(const std::string& rule, bool algebraic = true,
          const std::string& definitions = {}) {
  auto result = analyze(rule, algebraic, definitions);
  for (const auto& d : result.diagnostics) std::cerr << d.message << '\n';
  expect(result.ok(), "reverse fixture checks");
  auto p = *result.program->rules.at(0).semantic_plan;
  expect(p.proof.status == ProofStatus::Proved, p.proof.reason + " in " + rule);
  expect(p.proof.guard.empty(), "no inferred applicability restrictions");
  auto verified = verify(p, p.proof);
  expect(verified.status == ProofStatus::Proved, verified.reason);
  return p;
}
using Attributes = std::map<std::string, std::vector<List>>;
Metadata metadata(const Plan& p, const std::vector<Dense>& inputs,
                  const Attributes& attrs) {
  Metadata m;
  for (const auto& input : inputs) m.shapes.push_back(input.shape);
  for (const auto& n : p.nodes)
    if (n.descriptor && n.definition && !n.automatic &&
        !p.attributes.contains(n.descriptor->value))
      m.descriptors[n.descriptor->value] =
          attrs.at(p.descriptor_names.at(n.descriptor->value));
  return m;
}
void compare(const Plan& p, const std::vector<Dense>& inputs,
             const Attributes& attrs, const Dense& expected) {
  auto concrete = metadata(p, inputs, attrs);
  Context c{p, 1000000};
  c.sourceScope(p.lhs);
  for (const auto& [descriptor, values] : p.proof.constructors)
    for (std::size_t f = 0; f < values.size(); ++f)
      c.assignAttribute(descriptor, f, values[f]);
  Lowerer initial{c};
  initial.lower(p.lhs);
  Lowerer source{c};
  auto lhs = source.lower(p.lhs);
  RuleProof coordinates;
  for (const auto& [id, axes] : c.source().groups)
    coordinates.axis_definitions.push_back(axes);
  ConstructorEvaluator evaluator(concrete, coordinates);
  for (const auto& obligation : source.obligations)
    expect(std::get<bool>(evaluator.eval(obligation)), "source is legal");
  GraphEvaluator graph{evaluator, inputs};
  auto actual = graph.tensor(lhs);
  expect(actual.shape == expected.shape && actual.values == expected.values,
         "source agrees with independent physical reference");
  c.enableSetReasoning();
  auto correspondence = correspond(c, lhs, normalize(c, lhs.value),
                                   normalize(c, arithmeticSkeleton(p, p.rhs)));
  Lowerer target{c, &correspondence.anchors};
  target.interfaces = &correspondence.interfaces;
  auto rhs = target.lower(p.rhs);
  actual = graph.tensor(rhs);
  expect(actual.shape == expected.shape && actual.values == expected.values,
         "replacement agrees with independent physical reference");
}
bool legal(const Plan& p, List shape, List dimensions) {
  try {
    Context c{p, 1000000};
    c.sourceScope(p.lhs);
    Lowerer source{c};
    source.lower(p.lhs);
    RuleProof coordinates;
    for (const auto& [id, axes] : c.source().groups)
      coordinates.axis_definitions.push_back(axes);
    Metadata concrete{{shape}, {{0, {dimensions}}}};
    ConstructorEvaluator evaluator(concrete, coordinates);
    for (const auto& obligation : source.obligations)
      if (!std::get<bool>(evaluator.eval(obligation))) return false;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}
// Physical reference implementation uses tensor positions, not compiler
// coordinates, normalization, or inferred metadata expressions.
Dense reverse(const Dense& input, const List& dimensions) {
  Dense out{input.shape, {}};
  enumerate(out.shape, [&](List at) {
    for (auto d : dimensions) at[d] = input.shape[d] - at[d] - 1;
    out.values.push_back(input.at(at));
  });
  return out;
}
Dense multiply(const Dense& a, const Dense& b) {
  expect(a.shape == b.shape, "multiply shape");
  auto out = a;
  for (std::size_t i = 0; i < out.values.size(); ++i)
    out.values[i] *= b.values[i];
  return out;
}
Dense slice(const Dense& input, const List& starts, const List& sizes,
            const List& strides) {
  Dense out{sizes, {}};
  enumerate(sizes, [&](List at) {
    for (std::size_t i = 0; i < at.size(); ++i)
      at[i] = starts[i] + strides[i] * at[i];
    out.values.push_back(input.at(at));
  });
  return out;
}
S vector(const List& values) {
  std::vector<S> out;
  for (auto n : values) out.push_back(integer(n));
  return make(Symbol::Kind::List, {}, 0, 0, out);
}
void invalid(const std::string& access, const std::string& diagnostic) {
  auto result = analyze(
      "", true,
      "dialect D { op f(input:tensor)->tensor { attrs { dimensions:index[]; } "
      "dtype(t){yield t;} axes {bind S {input:all_axes(input)};yield S;} "
      "semantics {yield Read(input," +
          access + ");} } }");
  expect(!result.ok(), "invalid reverse definition rejected");
  for (const auto& d : result.diagnostics)
    if (d.message.find(diagnostic) != std::string::npos) return;
  throw std::runtime_error("missing reverse diagnostic: " + diagnostic);
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
    auto identity = plan("rule r {(reverse[@r] A)=>(reverse[@r] A)}", false);
    auto twice = plan(
        "rule r {(reverse[@outer] (reverse[@inner] A))=>A "
        "where {@outer.dimensions == @inner.dimensions}}",
        false);
    auto empty = plan(
        "rule r {(reverse[@r] A)=>A "
        "where {@r.dimensions == []}}",
        false);
    auto addition = plan(
        "rule r {(reverse[@r] (add A B)) "
        "=> (add (reverse[@r] A) (reverse[@r] B))}");
    auto product = plan(
        "rule r {(reverse[@r] (mul A B)) "
        "=> (mul (reverse[@r] A) (reverse[@r] B))}");
    auto broadcast = plan(
        "rule r {(reverse[@r] (broadcast[@b] C)) "
        "=> (broadcast[@b] C) "
        "where {len(@b.dimensions)==0}}",
        false);

    for (std::size_t rank = 0; rank <= 4; ++rank) {
      List shape;
      for (std::size_t i = 0; i < rank; ++i) shape.push_back(2 + i);
      auto a = data(shape, 1), b = data(shape, 4);
      Dense constant{shape, List(a.values.size(), 3)}, scalar{{}, {3}};
      for (std::size_t mask = 0; mask < (std::size_t{1} << rank); ++mask) {
        List dimensions;
        for (std::size_t i = rank; i > 0; --i)
          if (mask & (std::size_t{1} << (i - 1))) dimensions.push_back(i - 1);
        Attributes attrs{{"r", {dimensions}}};
        compare(identity, {a}, attrs, reverse(a, dimensions));
        compare(twice, {a}, {{"outer", {dimensions}}, {"inner", {dimensions}}},
                a);
        compare(addition, {a, b}, attrs, reverse(add(a, b), dimensions));
        compare(product, {a, b}, attrs, reverse(multiply(a, b), dimensions));
        attrs["b"] = {{}, shape};
        compare(broadcast, {scalar}, attrs, constant);
        compare(addition, {a, constant}, {{"r", {dimensions}}},
                reverse(add(a, constant), dimensions));
        compare(product, {a, constant}, {{"r", {dimensions}}},
                reverse(multiply(a, constant), dimensions));
      }
      compare(empty, {a}, {{"r", {{}}}}, a);
    }
    for (auto shape : {List{0}, List{0, 3}, List{2, 0, 3}, List{1, 1, 1}}) {
      auto a = data(shape, 1), b = data(shape, 2);
      List dimensions(shape.size());
      std::iota(dimensions.begin(), dimensions.end(), 0);
      compare(identity, {a}, {{"r", {dimensions}}}, reverse(a, dimensions));
      compare(twice, {a}, {{"outer", {dimensions}}, {"inner", {dimensions}}},
              a);
      compare(addition, {a, b}, {{"r", {dimensions}}},
              reverse(add(a, b), dimensions));
    }
    expect(legal(identity, {2, 3}, {1, 0}), "unsorted reverse axes legal");
    expect(legal(identity, {}, {}), "scalar reversal legal");
    expect(legal(identity, {0, 3}, {0}), "zero extent remains legal");
    expect(!legal(identity, {2, 3}, {1, 1}), "duplicate reverse axes rejected");
    expect(!legal(identity, {2, 3}, {-1}), "negative reverse axis rejected");
    expect(!legal(identity, {2, 3}, {2}), "out-of-range reverse axis rejected");

    auto singleton_dialect = dialect;
    const std::string assertion =
        "assert is_valid_axis_list(attrs.dimensions, rank(input));";
    auto insertion = singleton_dialect.find(assertion);
    expect(insertion != std::string::npos, "reverse assertion exists");
    singleton_dialect.insert(
        insertion + assertion.size(),
        "let selected_sizes = gather(shape(input), attrs.dimensions);"
        "assert selected_sizes == offset_extents("
        "extent_sub(selected_sizes, selected_sizes), 1);");
    auto singleton =
        plan("rule r {(reverse[@r] A)=>A}", false, singleton_dialect);
    auto singleton_input = data({1, 4, 1}, 2);
    compare(singleton, {singleton_input}, {{"r", {{2, 0}}}}, singleton_input);

    auto reverse_slice = plan(
        "rule r {(reverse[@r] (slice[@s] A)) "
        "=> (reverse[@r] (slice[@s] A))}",
        false);
    auto slice_reverse = plan(
        "rule r {(slice[@s] (reverse[@r] A)) "
        "=> (slice[@s] (reverse[@r] A))}",
        false);
    auto a = data({9, 11}, 2);
    Attributes attrs{{"r", {{1, 0}}}, {"s", {{1, 2}, {3, 4}, {2, 2}}}};
    compare(reverse_slice, {a}, attrs,
            reverse(slice(a, {1, 2}, {3, 4}, {2, 2}), {1, 0}));
    compare(slice_reverse, {a}, attrs,
            slice(reverse(a, {1, 0}), {1, 2}, {3, 4}, {2, 2}));

    Context c{identity, 1000000};
    auto domain = vector({3, 4, 5});
    auto axes = make(Symbol::Kind::AxisGroup, "test", 42, 0, {domain});
    auto coords = coordinates(axes);
    auto reflected = indexReverse(c, coords, domain, vector({2, 0}));
    auto selected = indexSelect(c, reflected, vector({2, 1, 0}));
    auto expected = indexReverse(c, indexSelect(c, coords, vector({2, 1, 0})),
                                 vector({5, 4, 3}), vector({0, 2}));
    expect(equalIndex(c, selected, expected), "selection remaps reverse axes");
    expect(equalIndex(c, indexReverse(c, reflected, domain, vector({2, 0})),
                      coords),
           "same domain and dimensions cancel");
    expect(
        !equalIndex(c, reflected, indexReverse(c, coords, domain, vector({1}))),
        "different reverse dimensions remain different reads");
    auto changed_domain =
        indexReverse(c, reflected, vector({4, 5, 6}), vector({2, 0}));
    expect(changed_domain->kind == IndexExpr::Kind::Reverse,
           "different reflection domains cannot cancel");
    auto remapped_axes = make(Symbol::Kind::AxisGroup, "replacement", 44, 0,
                              {vector({2, 3, 4})});
    auto replacement =
        indexOffset(c, vector({1, 1, 1}), coordinates(remapped_axes));
    auto substituted = indexSubstitute(c, reflected, axes, replacement);
    expect(substituted->kind == IndexExpr::Kind::Reverse &&
               equal(substituted->axes, domain),
           "coordinate substitution retains the original reflection extents");
    auto ones = vector({1, 4, 1});
    auto unit_coords =
        coordinates(make(Symbol::Kind::AxisGroup, "unit", 43, 0, {ones}));
    expect(equalIndex(c, indexReverse(c, unit_coords, ones, vector({2, 0})),
                      unit_coords),
           "known singleton axes are identity");
    Metadata no_metadata;
    RuleProof no_proof;
    ConstructorEvaluator bounds(no_metadata, no_proof);
    expect(std::get<List>(bounds.eval(indexLowerBound(c, reflected))) ==
               List({0, 0, 0}),
           "reverse lower bound");
    expect(std::get<List>(bounds.eval(indexUpperBound(c, reflected))) ==
               List({2, 3, 4}),
           "reverse upper bound");
    bool rejected = false;
    try {
      auto outside = indexReverse(c, indexOffset(c, vector({1, 1, 1}), coords),
                                  domain, vector({0}));
      indexLowerBound(c, outside);
    } catch (const ProofFailure&) {
      rejected = true;
    }
    expect(rejected,
           "bounds cannot assume arbitrary incoming coordinates are valid");

    auto unguarded =
        analyze("rule r {(reverse[@outer] (reverse[@inner] A))=>A}");
    expect(unguarded.ok() &&
               unguarded.program->rules[0].semantic_plan->proof.status ==
                   ProofStatus::Unknown,
           "reversal cancellation requires matching axis selections");
    auto wrong = analyze(
        "rule r {(reverse[@r] (add A B)) "
        "=> (add (reverse[@r] A) B)}");
    expect(wrong.ok() && wrong.program->rules[0].semantic_plan->proof.status ==
                             ProofStatus::Unknown,
           "dropping reversal from an operand is rejected");
    invalid("IndexReverse(Coords(S))", "IndexReverse requires");
    invalid(
        "IndexReverse(IndexScale(attrs.dimensions, Coords(S)), "
        "attrs.dimensions)",
        "requires Coords(group)");
    invalid("IndexReverse(Coords(S), rank(input))",
            "incorrect axis expression type");
    invalid("IndexReverse(Coords(S), attrs.unknown)", "unknown axis attribute");
    expect(format(*identity.nodes[identity.lhs].definition)
                   .find("IndexReverse(Coords(S), attrs.dimensions)") !=
               std::string::npos,
           "definition formatter preserves reversal");
    std::cout << "reverse tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
