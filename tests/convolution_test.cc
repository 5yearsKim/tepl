#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/attr_synth/axis_selectors.h"
#include "src/core/pipeline.h"
#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/symbolic/normalize.h"
#include "src/core/syntax/parse.h"
#include "tests/index_graph_eval.h"

namespace {
using namespace tepl::core::semantics;
using tepl::test::add;
using tepl::test::ConstructorEvaluator;
using tepl::test::Coordinate;
using tepl::test::Coordinates;
using tepl::test::data;
using tepl::test::Dense;
using tepl::test::enumerate;
using tepl::test::GraphEvaluator;
using tepl::test::transpose;
void expect(bool x, const std::string& message) {
  if (!x) throw std::runtime_error(message);
}
std::string dialect;
Plan plan(const std::string& rules, std::string definitions = {}) {
  auto p = tepl::parse((definitions.empty() ? dialect : definitions) + rules);
  expect(p.ok(), "convolution fixture parses");
  auto checked = tepl::core::inspect(
      *p.program, {tepl::core::SemanticPolicy::kAlgebraic, true});
  for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  expect(checked.ok(), "convolution fixture checks");
  auto& out = *checked.program->rules[0].semantic_plan;
  return out;
}
void proved(const Plan& p) {
  expect(p.proof.status == ProofStatus::Proved, p.proof.reason);
  expect(p.proof.guard.empty(), "no extra inferred guard");
  // Synthesis already runs verify(). Positive replay is checked once below;
  // corrupted candidates retain separate independent-verifier regressions.
}
// Direct physical reference: no axis maps, semantic IR, or normalization.
using ConvConfig = std::pair<List, List>;  // strides, dilations
Dense convolution(const Dense& x, const Dense& kernel, ConvConfig config = {}) {
  auto spatial = x.shape.size() - 2;
  auto [strides, dilations] = config;
  if (strides.empty()) strides = List(spatial, 1);
  if (dilations.empty()) dilations = List(spatial, 1);
  Dense out{{x.shape[0]}, {}};
  List window(kernel.shape.begin(), kernel.shape.begin() + spatial);
  for (std::size_t i = 0; i < spatial; ++i)
    out.shape.push_back(
        (x.shape[i + 1] - (dilations[i] * (window[i] - 1) + 1)) / strides[i] +
        1);
  out.shape.push_back(kernel.shape.back());
  window.push_back(x.shape.back());
  enumerate(out.shape, [&](const List& output) {
    std::int64_t sum = 0;
    enumerate(window, [&](const List& k) {
      List input{output[0]}, weight;
      for (std::size_t i = 0; i < spatial; ++i) {
        input.push_back(strides[i] * output[i + 1] + dilations[i] * k[i]);
        weight.push_back(k[i]);
      }
      input.push_back(k.back());
      weight.push_back(k.back());
      weight.push_back(output.back());
      sum += x.at(input) * kernel.at(weight);
    });
    out.values.push_back(sum);
  });
  return out;
}
void compare(const Plan& p, const std::vector<Dense>& inputs,
             const Dense& expected, List permutation = {},
             std::vector<ConvConfig> configs = {}) {
  Metadata concrete;
  for (const auto& input : inputs) concrete.shapes.push_back(input.shape);
  for (const auto& node : p.nodes)
    if (node.descriptor && node.definition && p.isCaptured(node)) {
      if (node.definition->fields == std::vector<std::string>{"permutation"})
        concrete.descriptors[node.descriptor->value] = {permutation};
      else if (!node.definition->fields.empty()) {
        auto d = node.descriptor->value;
        auto config = configs.empty()
                          ? ConvConfig{List(inputs[0].shape.size() - 2, 1),
                                       List(inputs[0].shape.size() - 2, 1)}
                          : configs.at(d);
        concrete.descriptors[d] = {config.first, config.second};
      }
    }
  Context c{p, 1000000};
  c.sourceScope(p.lhs);
  for (const auto& [descriptor, fields] : p.proof.constructors)
    for (std::size_t f = 0; f < fields.size(); ++f)
      c.assignAttribute(descriptor, f, fields[f]);
  Lowerer first{c};
  first.lower(p.lhs);
  Lowerer lower{c};
  auto lhs = lower.lower(p.lhs);
  RuleProof coordinates;
  for (const auto& [id, g] : c.source().groups)
    coordinates.axis_definitions.push_back(g);
  ConstructorEvaluator metadata(concrete, coordinates);
  GraphEvaluator eval{metadata, inputs};
  auto actual = eval.tensor(lhs);
  expect(actual.shape == expected.shape && actual.values == expected.values,
         "lowered convolution agrees with physical reference");
  // Check RHS in its independently reconstructed source coordinate frame.
  c.enableSetReasoning();
  auto match = correspond(c, lhs, normalize(c, lhs.value),
                          normalize(c, arithmeticSkeleton(c, p.rhs)));
  Lowerer target{c, &match.anchors};
  target.interfaces = &match.interfaces;
  auto rhs = target.lower(p.rhs);
  // RHS output-only groups are projected source coordinates in these fixtures.
  auto replaced = eval.tensor(rhs);
  expect(replaced.shape == expected.shape && replaced.values == expected.values,
         "RHS convolution agrees with physical reference");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    expect(argc == 2, "dialect path argument");
    std::string error;
    auto runfiles = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    std::ifstream f(runfiles->Rlocation(argv[1]));
    dialect = {std::istreambuf_iterator<char>(f), {}};
    expect(!dialect.empty(), "read dialect");
    auto input = plan(
        "rule r {(c: conv (add A B) W) => (add (conv[c.attrs] A W) "
        "(conv[c.attrs] B W))}");
    auto kernel = plan(
        "rule r {(c: conv A (add U V)) => (add (conv[c.attrs] A U) "
        "(conv[c.attrs] A V))}");
    auto composition = plan(
        "rule r {(c: conv (p: transpose (add A B)) W) => (add (conv[c.attrs] "
        "(transpose[p.attrs] A) W) (conv[c.attrs] (transpose[p.attrs] B) W))}");
    auto nested = plan(
        "rule r {(outer: conv (inner: conv (add A B) W) V) => (add "
        "(conv[outer.attrs] (conv[inner.attrs] A W) V) (conv[outer.attrs] "
        "(conv[inner.attrs] B W) V))}");
    proved(input);
    proved(kernel);
    proved(composition);
    proved(nested);
    for (const auto* p : {&input, &kernel, &composition})
      for (const auto& [id, fields] : p->proof.constructors) {
        expect(fields.size() == 2, "two inferred convolution attributes");
        expect(fields[0]->kind == Symbol::Kind::Field &&
                   fields[0]->name == "strides" &&
                   fields[1]->kind == Symbol::Kind::Field &&
                   fields[1]->name == "dilations",
               "inferred attributes are reusable source expressions");
      }
    // One normal case per spatial rank and each empty-axis case at rank one.
    // Avoid multiplying every empty case by every rank. Nontrivial batches and
    // channel reductions remain in the normal rank-one case.
    for (const auto [spatial, empty] : {std::pair<std::size_t, int>{1, 0},
                                        {2, 0},
                                        {3, 0},
                                        {1, 1},
                                        {1, 2},
                                        {1, 3}}) {
      List xs{empty == 1 ? 0 : spatial == 1 ? 2 : 1}, ws;
      for (std::size_t i = 0; i < spatial; ++i) {
        xs.push_back(3 + i % 2);
        ws.push_back(2);
      }
      xs.push_back(empty == 2 ? 0 : spatial == 1 ? 2 : 1);
      ws.push_back(xs.back());
      ws.push_back(empty == 3 ? 0 : spatial == 1 ? 2 : 1);
      auto a = data(xs, 1), b = data(xs, 2), w = data(ws, 3), v = data(ws, 4);
      compare(input, {a, b, w}, convolution(add(a, b), w));
      // Empty-axis behavior is checked at both nesting depths. The kernel
      // and transpose variants exercise distinct accesses on nonempty data.
      if (empty == 0) {
        compare(kernel, {a, w, v}, convolution(a, add(w, v)));
        List permutation(xs.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::reverse(permutation.begin() + 1, permutation.end() - 1);
        compare(composition, {a, b, w},
                convolution(transpose(add(a, b), permutation), w), permutation);
      }
      List outer_shape(spatial, 2);
      outer_shape.push_back(ws.back());
      outer_shape.push_back(spatial == 1 ? 2 : 1);
      auto outer = data(outer_shape, 5);
      compare(nested, {a, b, w, outer},
              convolution(convolution(add(a, b), w), outer));
    }
    // Per-axis coefficients, uneven division, and composition are checked
    // against physical execution rather than just a symbolic proof.
    {
      constexpr std::size_t spatial = 2;
      List xs{1}, ws, strides, dilations;
      for (std::size_t i = 0; i < spatial; ++i) {
        xs.push_back(5 + i);
        ws.push_back(2);
        strides.push_back(2 + i % 2);
        dilations.push_back(1 + i % 2);
      }
      xs.push_back(1);
      ws.push_back(1);
      ws.push_back(1);
      ConvConfig config{strides, dilations};
      auto a = data(xs, 1), b = data(xs, 2), w = data(ws, 3), v = data(ws, 4);
      compare(input, {a, b, w}, convolution(add(a, b), w, config), {},
              {config});
      compare(kernel, {a, w, v}, convolution(a, add(w, v), config), {},
              {config});
      List permutation(xs.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::reverse(permutation.begin() + 1, permutation.end() - 1);
      compare(composition, {a, b, w},
              convolution(transpose(add(a, b), permutation), w, config),
              permutation, {config, {}});
    }
    {
      constexpr std::size_t spatial = 2;
      List xs{1}, ws(spatial, 2), vs(spatial, 2);
      for (std::size_t i = 0; i < spatial; ++i) xs.push_back(8 + i);
      xs.push_back(1);
      ws.push_back(1);
      ws.push_back(1);
      vs.push_back(1);
      vs.push_back(1);
      auto a = data(xs, 1), b = data(xs, 2), w = data(ws, 3), v = data(vs, 4);
      ConvConfig inner{List(spatial, 2), List(spatial, 2)},
          outer{List(spatial, 2), List(spatial, 1)};
      compare(nested, {a, b, w, v},
              convolution(convolution(add(a, b), w, inner), v, outer), {},
              {outer, inner});
    }
    // Large stride remains one symbolic multiplication, not repeated addition.
    auto large_a = data({1, 8, 1}, 1), large_b = data({1, 8, 1}, 2),
         large_w = data({2, 1, 1}, 3);
    ConvConfig large_scale{{2048}, {2}};
    compare(input, {large_a, large_b, large_w},
            convolution(add(large_a, large_b), large_w, large_scale), {},
            {large_scale});
    auto explicit_identity =
        plan("rule r {(read_identity A) => A}",
             dialect +
                 "dialect Reads {op read_identity(x:tensor)->tensor "
                 "{dtype(t){yield t } "
                 "axes {bind S=group(shape(x)) yield S } semantics {yield "
                 "Read(x,Coords(S)) }}}");
    proved(explicit_identity);
    auto sampling_definitions = dialect + R"(
      dialect Sampling {
        op sample(input:tensor)->tensor {
          attrs {steps: u64[] }
          dtype(t){yield t }
          axes {
            assert len(attrs.steps) == rank(input)
            assert is_positive_extents(attrs.steps)
            assert is_positive_extents(shape(input))
            bind Points = group(offset_extents(
                extent_floor_div(offset_extents(shape(input), -1), attrs.steps), 1))
            yield Points
          }
          semantics {yield Read(input,IndexScale(attrs.steps,Coords(Points))) }
        }
      })";
    auto sampling = plan(
        "rule r {(s: sample (add A B)) => (add (sample[s.attrs] A) "
        "(sample[s.attrs] B))}",
        sampling_definitions);
    proved(sampling);
    for (const auto& [id, fields] : sampling.proof.constructors)
      expect(fields.size() == 1 && fields[0]->kind == Symbol::Kind::Field &&
                 fields[0]->name == "steps",
             "generic coefficient field inferred without operation names");
    auto sa = data({5, 7}, 1), sb = data({5, 7}, 2), summed = add(sa, sb);
    Dense sampled{{3, 3}, {}};
    enumerate(sampled.shape, [&](const List& at) {
      sampled.values.push_back(summed.at({2 * at[0], 3 * at[1]}));
    });
    compare(sampling, {sa, sb}, sampled, {}, {{{2, 3}, {}}});
    auto identity = plan(
        "rule r {(c: conv A W) => (out: conv A W) derive "
        "{out.attrs=c.attrs }}");
    proved(identity);
    expect(verify(identity, identity.proof).status == ProofStatus::Proved,
           "independent verifier accepts a correct inferred convolution");
    auto legal = [&](const List& input_shape, const List& kernel_shape,
                     ConvConfig config = {}) {
      Context theory{identity, 1000000};
      theory.sourceScope(identity.lhs);
      Lowerer source{theory};
      source.lower(identity.lhs);
      RuleProof frame;
      for (const auto& [id, group] : theory.source().groups)
        frame.axis_definitions.push_back(group);
      if (config.first.empty() && config.second.empty())
        config = {List(input_shape.size() - 2, 1),
                  List(input_shape.size() - 2, 1)};
      Metadata concrete{{input_shape, kernel_shape},
                        {{0, {config.first, config.second}}}};
      ConstructorEvaluator evaluator(concrete, frame);
      try {
        for (const auto& obligation : source.obligations)
          if (!std::get<bool>(evaluator.eval(obligation))) return false;
      } catch (const std::exception&) {
        return false;
      }
      return true;
    };
    expect(legal({2, 3, 2}, {2, 2, 3}), "legal convolution metadata");
    expect(legal({0, 3, 0}, {2, 0, 0}),
           "empty batch/channel convolution metadata");
    expect(!legal({2, 3, 2}, {0, 2, 3}), "zero-size kernel rejected");
    expect(!legal({2, 3, 2}, {4, 2, 3}), "oversized kernel rejected");
    expect(!legal({2, 3, 2}, {2, 3, 3}), "channel mismatch rejected");
    expect(!legal({2, 3, 2}, {2, 2, 2, 3}), "spatial rank mismatch rejected");
    expect(legal({1, 8, 1}, {2, 1, 1}, {{3}, {2}}),
           "strided dilated legal metadata");
    expect(!legal({1, 8, 1}, {2, 1, 1}, {{0}, {1}}), "zero stride rejected");
    expect(!legal({1, 8, 1}, {2, 1, 1}, {{1}, {0}}), "zero dilation rejected");
    expect(!legal({1, 8, 1}, {2, 1, 1}, {{-1}, {1}}),
           "negative stride rejected");
    expect(!legal({1, 8, 1}, {2, 1, 1}, {{1, 2}, {1}}),
           "stride length mismatch rejected");
    expect(!legal({1, 8, 1}, {2, 1, 1}, {{1}, {1, 2}}),
           "dilation length mismatch rejected");
    expect(!legal({1, 3, 1}, {2, 1, 1}, {{1}, {3}}),
           "effective kernel too large rejected");
    auto wrong = plan("rule r {(c: conv (add A B) W) => (conv[c.attrs] A W)}");
    expect(wrong.proof.status == ProofStatus::Unknown, "missing term rejected");
    auto renamed = dialect;
    auto at = renamed.find("op convolution");
    renamed.replace(at, 14, "op correlation");
    at = renamed.find("alias: conv");
    renamed.replace(at, 12, "alias: corr");
    proved(
        plan("rule r {(c: corr (add A B) W) => (add (corr[c.attrs] A W) "
             "(corr[c.attrs] B W))}",
             renamed));
    // A deliberately different access, with the same Product/Reduce skeleton.
    auto bad = dialect;
    const std::string spatial_access =
        "IndexAdd(IndexScale(attrs.strides, Coords(O)), "
        "IndexScale(attrs.dilations, Coords(Window)))";
    at = bad.find(spatial_access);
    bad.replace(at, spatial_access.size(),
                "IndexAdd(IndexScale(attrs.strides, Coords(O)), "
                "IndexScale(attrs.dilations, Coords(O)))");
    // Compare definitions in separate dialects without conflicting operation
    // names.
    auto begin = bad.find("    op convolution");
    auto end = bad.find('{', begin) + 1;
    std::size_t depth = 1;
    for (; end < bad.size() && depth; ++end) {
      if (bad[end] == '{')
        ++depth;
      else if (bad[end] == '}')
        --depth;
    }
    expect(depth == 0, "find operation body boundary");
    auto operation = bad.substr(begin, end - begin);
    at = operation.find("op convolution");
    operation.replace(at, 14, "op shifted");
    at = operation.find("alias: conv");
    operation.erase(at, 12);
    auto definitions = dialect + "dialect Wrong {" + operation + "}";
    auto different = plan(
        "rule r {(c: conv A W) => (out: shifted A W) derive "
        "{out.attrs={strides:c.attrs.strides,dilations:c.attrs.dilations} }}",
        definitions);
    expect(different.proof.status == ProofStatus::Unknown,
           "different calculated access rejected");
    expect(verify(different, identity.proof).status == ProofStatus::Unknown,
           "independent verifier rejects different calculated accesses");
    // Changing coefficients can preserve the arithmetic skeleton while
    // changing tensor accesses. Both synthesis and verification must reject it.
    auto tampered = identity;
    auto& guessed =
        tampered.descriptors.at(identity.nodes[identity.rhs].descriptor->value)
            .fields;
    if (guessed.empty())
      for (std::size_t f = 0; f < 2; ++f)
        guessed.push_back(
            attributeSymbol(identity, identity.nodes[identity.lhs], f));
    std::swap(guessed[0], guessed[1]);
    expect(verify(tampered, identity.proof).status == ProofStatus::Unknown,
           "independent verifier rejects tampered inferred coefficients");
    auto inferred_distribution = plan(
        "rule r {(c: conv (add A B) W) => (add (conv[c.attrs] A W) "
        "(conv[c.attrs] B W))}");
    proved(inferred_distribution);
    auto changed = plan(
        "rule r {(c: conv A W) => (out: conv A W) derive "
        "{out.attrs={strides:c.attrs.dilations,dilations:c.attrs.strides} }}");
    expect(changed.proof.status == ProofStatus::Unknown,
           "swapped scales rejected");
    expect(verify(changed, identity.proof).status == ProofStatus::Unknown,
           "independent verifier rejects changed scales");
    auto malformed = tepl::parse(
        "dialect D {op f(x:tensor)->tensor {axes {bind S {x:all_axes(x)} yield "
        "S } "
        "semantics {yield Read(x,Coords(Missing)) }}}");
    expect(malformed.ok() && !tepl::core::inspect(*malformed.program).ok(),
           "unknown coordinate rejected");
    Context c{input, 100000};
    auto h =
        coordinates(make(Symbol::Kind::AxisGroup, "same", 1, 1,
                         {make(Symbol::Kind::List, {}, 0, 0, {integer(3)})}));
    auto r =
        coordinates(make(Symbol::Kind::AxisGroup, "same", 2, 1,
                         {make(Symbol::Kind::List, {}, 0, 0, {integer(3)})}));
    auto t =
        coordinates(make(Symbol::Kind::AxisGroup, "t", 3, 1,
                         {make(Symbol::Kind::List, {}, 0, 0, {integer(3)})}));
    expect(equalIndex(c, indexAdd({indexAdd({h, r}), t}),
                      indexAdd({h, indexAdd({t, r})})),
           "associative coordinate addition");
    expect(!equalIndex(c, h, r),
           "same names and extents do not identify coordinates");
    expect(equalIndex(c, indexAdd({h, h}), indexAdd({h, h})),
           "repeated coordinate coefficient collection");
    auto vec = [](std::initializer_list<std::size_t> values) {
      std::vector<S> out;
      for (auto x : values) out.push_back(integer(x));
      return make(Symbol::Kind::List, {}, 0, 0, out);
    };
    expect(equalIndex(c, indexScale(c, vec({2}), indexScale(c, vec({3}), h)),
                      indexScale(c, vec({6}), h)),
           "nested scale multiplication");
    expect(equalIndex(c, indexScale(c, vec({2}), indexAdd({h, r})),
                      indexAdd({indexScale(c, vec({2}), h),
                                indexScale(c, vec({2}), r)})),
           "scale distributes over coordinate sums");
    auto q = make(Symbol::Kind::Field, "q", 0, 0);
    auto divisor = make(Symbol::Kind::Field, "s", 0, 1);
    c.learn(call("==", {call("len", {q}), call("len", {divisor})}));
    auto rounded =
        call("extent_mul", {divisor, call("extent_floor_div", {q, divisor})});
    expect(!extentBoundArithmetic(c, rounded, q, QueryMode::Entailment),
           "floor bound needs positive divisor and nonnegative numerator");
    c.learn(call("is_valid_extents", {q}));
    c.learn(call("is_positive_extents", {divisor}));
    expect(extentBoundArithmetic(c, rounded, q, QueryMode::Entailment),
           "floor bound theorem");
    auto negative =
        make(Symbol::Kind::List, {}, 0, 0, {make(Symbol::Kind::Integer, "-1")});
    auto floor_negative = call("extent_floor_div", {negative, vec({2})});
    expect(floor_negative->kind == Symbol::Kind::List &&
               floor_negative->args[0]->name == "-1",
           "literal metadata division uses mathematical floor");
    expect(
        !extentBoundArithmetic(c, call("offset_extents", {rounded, integer(1)}),
                               q, QueryMode::Entailment),
        "floor upper bound cannot justify an extra access");
    bool overflow = false;
    try {
      auto sum = h;
      for (int i = 0; i < 64; ++i) sum = indexAdd({sum, sum});
    } catch (const ProofFailure&) {
      overflow = true;
    }
    expect(overflow, "checked index coefficient overflow");
    std::cout << "convolution tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
