#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

#include "src/core/pipeline.h"
#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/symbolic/normalize.h"
#include "src/core/syntax/parse.h"
#include "tests/index_graph_eval.h"

namespace {
using namespace tepl::core::semantics;
using namespace tepl::test;

const std::string dialect = R"(
dialect D {
  op slice(input: tensor) -> tensor {
    attrs { starts: index[]; sizes: index[]; strides: index[]; }
    dtype(t) { yield t; }
    axes { bind O = group(attrs.sizes); yield O; }
    semantics {
      yield Read(input, IndexOffset(attrs.starts,
        IndexScale(attrs.strides, Coords(O))));
    }
  }
  op computed(input: tensor) -> tensor {
    attrs {
      starts: index[]; extra: index[]; sizes: index[];
      strides: index[]; factors: index[];
    }
    dtype(t) { yield t; }
    axes {
      assert is_valid_extents(attrs.starts);
      assert is_valid_extents(attrs.extra);
      assert is_positive_extents(attrs.strides);
      assert is_positive_extents(attrs.factors);
      bind O = group(attrs.sizes); yield O;
    }
    semantics {
      yield Read(input, IndexOffset(extent_add(attrs.starts, attrs.extra),
        IndexScale(extent_mul(attrs.strides, attrs.factors), Coords(O))));
    }
  }
  op expanded(input: tensor) -> tensor {
    attrs {
      starts: index[]; extra: index[]; sizes: index[];
      strides: index[]; factors: index[];
    }
    dtype(t) { yield t; }
    axes { bind O = group(attrs.sizes); yield O; }
    semantics {
      yield Read(input, IndexOffset(attrs.starts, IndexOffset(attrs.extra,
        IndexScale(attrs.strides, IndexScale(attrs.factors, Coords(O))))));
    }
  }
  op selected(input: tensor) -> tensor {
    attrs {
      starts: index[]; sizes: index[]; strides: index[]; positions: index[];
    }
    dtype(t) { yield t; }
    axes { bind O = group(attrs.sizes); yield O; }
    semantics {
      yield Read(input, IndexSelect(
        IndexOffset(attrs.starts, IndexScale(attrs.strides, Coords(O))),
        attrs.positions));
    }
  }
  op distributed(input: tensor) -> tensor {
    attrs {
      starts: index[]; sizes: index[]; strides: index[]; positions: index[];
    }
    dtype(t) { yield t; }
    axes { bind O = group(attrs.sizes); yield O; }
    semantics {
      yield Read(input, IndexOffset(gather(attrs.starts, attrs.positions),
        IndexScale(gather(attrs.strides, attrs.positions),
          IndexSelect(Coords(O), attrs.positions))));
    }
  }
  op nested(input: tensor) -> tensor {
    attrs {
      starts: index[]; sizes: index[]; strides: index[];
      first: index[]; second: index[];
    }
    dtype(t) { yield t; }
    axes { bind O = group(attrs.sizes); yield O; }
    semantics {
      yield Read(input, IndexSelect(IndexSelect(
        IndexOffset(attrs.starts, IndexScale(attrs.strides, Coords(O))),
        attrs.first), attrs.second));
    }
  }
  op flattened(input: tensor) -> tensor {
    attrs {
      starts: index[]; sizes: index[]; strides: index[];
      first: index[]; second: index[];
    }
    dtype(t) { yield t; }
    axes { bind O = group(attrs.sizes); yield O; }
    semantics {
      yield Read(input, IndexSelect(
        IndexOffset(attrs.starts, IndexScale(attrs.strides, Coords(O))),
        gather(attrs.first, attrs.second)));
    }
  }
}
)";

void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
tepl::core::InspectionResult analyze(const std::string& definitions,
                                     const std::string& rule = {}) {
  auto parsed = tepl::parse(definitions + rule, "calculated_reads.tepl");
  for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  expect(parsed.ok(), "calculated read fixture parses");
  return tepl::core::inspect(*parsed.program,
                             {tepl::core::SemanticPolicy::kStrict, true});
}
Plan plan(const std::string& rule, const std::string& definitions = dialect) {
  auto checked = analyze(definitions, rule);
  for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  expect(checked.ok(), "calculated read fixture checks");
  auto p = *checked.program->rules.at(0).semantic_plan;
  expect(p.proof.status == ProofStatus::Proved, p.proof.reason);
  expect(p.proof.guard.empty(), "no inferred restrictions");
  auto verified = verify(p, p.proof);
  expect(verified.status == ProofStatus::Proved, verified.reason);
  return p;
}
using Attributes = std::map<std::string, std::vector<List>>;
Metadata metadata(const Plan& p, const Dense& input, const Attributes& attrs) {
  Metadata m{{input.shape}, {}};
  for (const auto& n : p.nodes)
    if (n.descriptor && n.definition && !n.automatic &&
        !p.attributes.contains(n.descriptor->value))
      m.descriptors[n.descriptor->value] =
          attrs.at(p.descriptor_names.at(n.descriptor->value));
  return m;
}
void compare(const Plan& p, const Dense& input, const Attributes& attrs,
             const Dense& expected) {
  auto concrete = metadata(p, input, attrs);
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
  std::vector<Dense> inputs{input};
  GraphEvaluator graph{evaluator, inputs};
  auto actual = graph.tensor(lhs);
  expect(actual.shape == expected.shape && actual.values == expected.values,
         "source agrees with independent physical reference");
  c.enableSetReasoning();
  auto correspondence = correspond(c, lhs, normalize(c, lhs.value),
                                   normalize(c, arithmeticSkeleton(c, p.rhs)));
  Lowerer target{c, &correspondence.anchors};
  target.interfaces = &correspondence.interfaces;
  auto rhs = target.lower(p.rhs);
  actual = graph.tensor(rhs);
  expect(actual.shape == expected.shape && actual.values == expected.values,
         "replacement agrees with independent physical reference");
}
// Evaluate source legality independently, including obligations that would be
// erased by normalization (for example selection of an empty coordinate tuple).
bool legal(const Plan& p, List shape, std::vector<List> fields) {
  try {
    Context c{p, 1000000};
    c.sourceScope(p.lhs);
    Lowerer source{c};
    source.lower(p.lhs);
    RuleProof coordinates;
    for (const auto& [id, axes] : c.source().groups)
      coordinates.axis_definitions.push_back(axes);
    Metadata concrete{{shape}, {{0, fields}}};
    ConstructorEvaluator evaluator(concrete, coordinates);
    for (const auto& obligation : source.obligations)
      if (!std::get<bool>(evaluator.eval(obligation))) return false;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}
Dense reference(const Dense& input, const List& sizes, const List& starts,
                const List& strides, const List& positions) {
  Dense out{sizes, {}};
  enumerate(sizes, [&](const List& at) {
    List read;
    for (auto p : positions) read.push_back(starts[p] + strides[p] * at[p]);
    out.values.push_back(input.at(read));
  });
  return out;
}
std::string withRead(const std::string& access) {
  return R"(dialect D {
    op f(input: tensor) -> tensor {
      attrs { sizes: index[]; positions: index[]; }
      dtype(t) { yield t; }
      axes { let local = attrs.sizes; bind O = group(attrs.sizes); yield O; }
      semantics { yield Read(input, )" +
         access + "); } } }";
}
void invalid(const std::string& access, const std::string& message) {
  auto checked = analyze(withRead(access));
  expect(!checked.ok(), "invalid coordinate definition rejected");
  for (const auto& d : checked.diagnostics)
    if (d.message.find(message) != std::string::npos) return;
  throw std::runtime_error("missing diagnostic: " + message);
}
}  // namespace

int main() {
  try {
    auto computed = plan(
        "rule r { (c: computed A) => (e: expanded A) derive { e.attrs = { "
        "starts: c.attrs.starts, extra: c.attrs.extra, sizes: c.attrs.sizes, "
        "strides: c.attrs.strides, factors: c.attrs.factors }; } }");
    auto automatic = plan("rule r { (c: computed A) => (slice A) }");
    auto selected = plan(
        "rule r { (s: selected A) => (d: distributed A) derive { d.attrs = { "
        "starts: s.attrs.starts, sizes: s.attrs.sizes, strides: "
        "s.attrs.strides, positions: s.attrs.positions }; } }");
    auto selection_without_metadata =
        analyze(dialect, "rule r { (s: selected A) => (distributed A) }");
    expect(selection_without_metadata.ok() &&
               selection_without_metadata.program->rules[0]
                       .semantic_plan->proof.status == ProofStatus::Unknown,
           "unrestricted selector inference remains unsupported");
    auto nested = plan(
        "rule r { (s: nested A) => (f: flattened A) derive { f.attrs = { "
        "starts: s.attrs.starts, sizes: s.attrs.sizes, strides: "
        "s.attrs.strides, first: s.attrs.first, second: s.attrs.second }; } }");
    auto identity = plan("rule r { (s: selected A) => (selected[s.attrs] A) }");
    auto computed_identity =
        plan("rule r { (s: computed A) => (computed[s.attrs] A) }");

    for (std::size_t rank = 0; rank <= 4; ++rank) {
      List sizes, starts, extra, strides, factors, effective_starts,
          effective_strides, positions;
      for (std::size_t i = 0; i < rank; ++i) {
        sizes.push_back(2 + i % 3);
        starts.push_back(i % 2);
        extra.push_back(1);
        strides.push_back(1 + i % 2);
        factors.push_back(2);
        effective_starts.push_back(starts.back() + extra.back());
        effective_strides.push_back(strides.back() * factors.back());
        positions.push_back(i);
      }
      List shape;
      for (std::size_t i = 0; i < rank; ++i)
        shape.push_back(effective_starts[i] +
                        effective_strides[i] * (sizes[i] - 1) + 1);
      auto input = data(shape, 3);
      auto expected = reference(input, sizes, effective_starts,
                                effective_strides, positions);
      Attributes attrs{{"c", {starts, extra, sizes, strides, factors}}};
      compare(computed, input, attrs, expected);
      compare(automatic, input, attrs, expected);

      // Distinct extents and a non-involutive permutation catch direction and
      // coefficient-order mistakes that square tensors or swaps would hide.
      if (rank > 1)
        std::rotate(positions.begin(), positions.begin() + 1, positions.end());
      shape.clear();
      for (auto p : positions)
        shape.push_back(starts[p] + strides[p] * (sizes[p] - 1) + 1);
      input = data(shape, 4);
      expected = reference(input, sizes, starts, strides, positions);
      compare(selected, input, {{"s", {starts, sizes, strides, positions}}},
              expected);
      List second(rank), composed;
      std::iota(second.rbegin(), second.rend(), 0);
      for (auto p : second) composed.push_back(positions[p]);
      shape.clear();
      for (auto p : composed)
        shape.push_back(starts[p] + strides[p] * (sizes[p] - 1) + 1);
      input = data(shape, 5);
      compare(nested, input,
              {{"s", {starts, sizes, strides, positions, second}}},
              reference(input, sizes, starts, strides, composed));
    }
    // A proper subset selects a coordinate from a larger output interface.
    compare(selected, data({5}, 1), {{"s", {{0, 1}, {3, 2}, {1, 2}, {1}}}},
            reference(data({5}, 1), {3, 2}, {0, 1}, {1, 2}, {1}));
    expect(legal(identity, {5, 4}, {{1, 0}, {2, 3}, {2, 2}, {1, 0}}),
           "valid coordinate permutation");
    expect(legal(identity, {}, {{}, {}, {}, {}}), "empty coordinate tuple");
    expect(!legal(identity, {4}, {{0}, {2}, {1}, {1}}),
           "out-of-range selector rejected");
    expect(!legal(identity, {4}, {{0}, {2}, {1}, {-1}}),
           "negative selector rejected");
    expect(!legal(identity, {4, 4}, {{0, 0}, {2, 2}, {1, 1}, {0, 0}}),
           "duplicate selectors rejected");
    expect(!legal(identity, {4, 4}, {{0}, {2}, {1}, {0}}),
           "read rank mismatch rejected");
    expect(!legal(identity, {4}, {{1}, {3}, {2}, {0}}),
           "selected access bounds checked");
    expect(!legal(identity, {4}, {{0}, {2}, {0}, {0}}),
           "nonpositive scales rejected");
    expect(!legal(computed_identity, {8}, {{0}, {0, 1}, {2}, {1}, {1}}),
           "computed offset lengths checked");
    expect(!legal(computed_identity, {8}, {{0}, {-1}, {2}, {1}, {1}}),
           "computed negative offset rejected");
    expect(!legal(computed_identity, {8}, {{0}, {0}, {2}, {1}, {0}}),
           "computed zero scale rejected");

    auto queries = plan(
        "rule r { (s: f A) => (f[s.attrs] A) }",
        withRead(
            "IndexOffset(offset_extents(all_axes(input), 1), "
            "IndexScale(offset_extents(range(rank(input)), 1), Coords(O)))"));
    auto input = data({4, 7}, 2);
    compare(queries, input, {{"s", {{2, 3}, {}}}},
            reference(input, {2, 3}, {1, 2}, {1, 2}, {0, 1}));
    auto gathered = plan(
        "rule r { (s: f A) => (f[s.attrs] A) }",
        withRead(
            "IndexScale(gather(attrs.sizes, attrs.positions), Coords(O))"));
    expect(legal(gathered, {4, 5}, {{2, 3}, {1, 0}}),
           "valid computed coefficient selection");
    expect(!legal(gathered, {4, 5}, {{2, 3}, {0, 0}}),
           "metadata gather rejects duplicate selectors");
    expect(!legal(gathered, {4, 5}, {{2, 3}, {2, 0}}),
           "metadata gather rejects out-of-range selectors");

    invalid("IndexScale(rank(input), Coords(O))",
            "incorrect axis expression type");
    invalid("IndexOffset(Coords(O), Coords(O))", "unsupported axis builtin");
    invalid("IndexScale(attrs.unknown, Coords(O))", "unknown axis attribute");
    invalid("IndexSelect(Coords(O), rank(input))",
            "incorrect axis expression type");
    invalid("IndexSelect(Coords(O))", "IndexSelect requires");
    invalid("IndexScale(attrs.sizes)", "IndexScale requires");
    invalid("IndexScale(gather(attrs.sizes), Coords(O))", "wrong arity");
    invalid("IndexScale(local, Coords(O))", "axes let locals are private");
    invalid("IndexScale(unknown(attrs.sizes), Coords(O))",
            "unsupported axis builtin");
    invalid(
        "IndexScale(offset_extents(attrs.sizes, 9223372036854775808), "
        "Coords(O))",
        "integer out of range");

    const auto& definition = *selected.nodes[selected.rhs].definition;
    expect(definition.extent_fields == std::set<std::size_t>({0, 1, 2}),
           "gather selector is not classified as an extent-valued field");
    auto printed = format(definition);
    expect(printed.find("IndexSelect(Coords(O), attrs.positions)") !=
               std::string::npos,
           "definition formatter preserves coordinate selection");
    expect(printed.find("gather(attrs.strides, attrs.positions)") !=
               std::string::npos,
           "definition formatter preserves computed metadata");
    auto tampered = automatic.proof;
    auto& fields = tampered.constructors.begin()->second;
    fields[0] = call("offset_extents", {fields[0], integer(1)});
    expect(verify(automatic, tampered).status == ProofStatus::Unknown,
           "independent verifier rejects changed calculated reads");
    auto wrong_selector = selected;
    wrong_selector.attributes.begin()->second[3] = call("range", {integer(0)});
    expect(
        verify(wrong_selector, selected.proof).status == ProofStatus::Unknown,
        "independent verifier rejects changed selection positions");
    std::cout << "calculated read tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
