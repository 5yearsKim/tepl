#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/analyze.h"
#include "src/core/index_semantics/index_inference.h"
#include "src/core/index_semantics/normalize.h"
#include "src/core/index_semantics/synthesize.h"
#include "src/core/index_semantics/verify.h"
#include "src/parse.h"
#include "tests/index_constructor_eval.h"

namespace {
using namespace tepl::core::index_semantics;
using tepl::test::ConstructorEvaluator;
using tepl::test::Coordinate;
using tepl::test::Coordinates;
void expect(bool x, const std::string& message) {
  if (!x) throw std::runtime_error(message);
}
std::string dialect;
Plan plan(const std::string& rules, std::string definitions = {}) {
  auto p = tepl::parse((definitions.empty() ? dialect : definitions) + rules);
  expect(p.ok(), "convolution fixture parses");
  auto checked = tepl::core::analyze(
      *p.program, {tepl::core::SemanticPolicy::kAlgebraic, true});
  for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  expect(checked.ok(), "convolution fixture checks");
  auto& out = *checked.program->rules[0].semantic_plan;
  return out;
}
void proved(const Plan& p) {
  expect(p.proof.status == ProofStatus::Proved, p.proof.reason);
  expect(p.proof.guard.empty(), "no extra inferred guard");
  expect(verify(p, p.proof).status == ProofStatus::Proved,
         "independent verifier accepts convolution proof");
}
void enumerate(const List& sizes, const std::function<void(const List&)>& fn) {
  List at(sizes.size());
  auto loop = [&](auto&& self, std::size_t d) -> void {
    if (d == sizes.size()) {
      fn(at);
      return;
    }
    for (at[d] = 0; at[d] < sizes[d]; ++at[d]) self(self, d + 1);
  };
  loop(loop, 0);
}
struct Dense {
  List shape, values;
  std::int64_t at(const List& p) const {
    expect(p.size() == shape.size(), "read rank");
    std::size_t offset = 0;
    for (std::size_t d = 0; d < p.size(); ++d) {
      expect(p[d] >= 0 && p[d] < shape[d],
             "calculated read stays in bounds: axis " + std::to_string(d) +
                 " index " + std::to_string(p[d]) + " extent " +
                 std::to_string(shape[d]));
      offset = offset * shape[d] + p[d];
    }
    return values.at(offset);
  }
};
Dense data(List shape, int seed) {
  Dense x{shape, {}};
  enumerate(shape, [&](const auto&) {
    x.values.push_back((std::int64_t(x.values.size()) * 3 + seed) % 11 - 5);
  });
  return x;
}
Dense add(const Dense& a, const Dense& b) {
  expect(a.shape == b.shape, "addition shape");
  auto out = a;
  for (std::size_t i = 0; i < out.values.size(); ++i)
    out.values[i] += b.values[i];
  return out;
}
Dense transpose(const Dense& a, const List& p) {
  Dense out;
  for (auto i : p) out.shape.push_back(a.shape.at(i));
  enumerate(out.shape, [&](const List& at) {
    List original(at.size());
    for (std::size_t i = 0; i < at.size(); ++i) original.at(p[i]) = at[i];
    out.values.push_back(a.at(original));
  });
  return out;
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
// Interpret a lowered graph with concrete coordinates. This checks lowering
// against the direct convolution above, rather than trusting the proof result.
struct GraphEvaluator {
  ConstructorEvaluator& metadata;
  const std::vector<Dense>& inputs;
  std::map<Coordinate, std::int64_t> assignment;
  Coordinates axes(const S& x) {
    return std::get<Coordinates>(metadata.eval(x));
  }
  List index(const I& e) {
    if (e->kind == IndexExpr::Kind::Coordinates) {
      List out;
      for (const auto& a : axes(e->axes)) out.push_back(assignment.at(a));
      return out;
    }
    auto out = index(e->args[0]);
    if (e->kind == IndexExpr::Kind::Scale) {
      for (auto& n : out) n *= e->coefficient;
    } else if (e->kind == IndexExpr::Kind::Weighted) {
      auto factors = std::get<List>(metadata.eval(e->axes));
      expect(out.size() == factors.size(), "scale tuple length");
      for (std::size_t i = 0; i < out.size(); ++i) out[i] *= factors[i];
    } else if (e->kind == IndexExpr::Kind::Select) {
      auto p = std::get<List>(metadata.eval(e->axes));
      List selected;
      for (auto n : p) selected.push_back(out.at(n));
      return selected;
    } else {
      for (std::size_t i = 1; i < e->args.size(); ++i) {
        auto next = index(e->args[i]);
        if (e->kind == IndexExpr::Kind::Concat)
          out.insert(out.end(), next.begin(), next.end());
        else {
          expect(out.size() == next.size(), "paired spatial tuple lengths");
          for (std::size_t j = 0; j < out.size(); ++j) out[j] += next[j];
        }
      }
    }
    return out;
  }
  std::int64_t value(const E& e) {
    if (e->kind == Expr::Kind::Const) return e->constant;
    if (e->kind == Expr::Kind::Read) {
      List p;
      if (e->access)
        p = index(e->access);
      else
        for (const auto& a : axes(e->axes)) p.push_back(assignment.at(a));
      return inputs.at(e->capture).at(p);
    }
    if (e->kind == Expr::Kind::Add)
      return value(e->args[0]) + value(e->args[1]);
    if (e->kind == Expr::Kind::Product)
      return value(e->args[0]) * value(e->args[1]);
    auto coordinates = axes(e->axes);
    auto sizes = std::get<List>(metadata.eval(call("axis_extents", {e->axes})));
    std::int64_t sum = 0;
    enumerate(sizes, [&](const List& at) {
      for (std::size_t i = 0; i < at.size(); ++i) {
        expect(!assignment.contains(coordinates[i]), "fresh reduction binder");
        assignment[coordinates[i]] = at[i];
      }
      sum += value(e->args[0]);
      for (const auto& a : coordinates) assignment.erase(a);
    });
    return sum;
  }
  Dense tensor(const TensorMeaning& t) {
    Dense out{std::get<List>(metadata.eval(t.shape)), {}};
    auto coordinates = axes(t.axes);
    enumerate(out.shape, [&](const List& at) {
      for (std::size_t i = 0; i < at.size(); ++i)
        assignment[coordinates[i]] = at[i];
      out.values.push_back(value(t.value));
      assignment.clear();
    });
    return out;
  }
};
void compare(const Plan& p, const std::vector<Dense>& inputs,
             const Dense& expected, List permutation = {},
             std::vector<ConvConfig> configs = {}) {
  Metadata concrete;
  for (const auto& input : inputs) concrete.shapes.push_back(input.shape);
  for (const auto& node : p.nodes)
    if (node.descriptor && node.definition && !node.automatic) {
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
                          normalize(c, arithmeticSkeleton(p, p.rhs)));
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
        "rule r {(conv[@c] (add A B) W) => (add (conv[@c] A W) (conv[@c] B "
        "W))}");
    auto kernel = plan(
        "rule r {(conv[@c] A (add U V)) => (add (conv[@c] A U) (conv[@c] A "
        "V))}");
    auto composition = plan(
        "rule r {(conv[@c] (transpose[@p] (add A B)) W) => "
        "(add (conv[auto] (transpose[@p] A) W) (conv[auto] (transpose[@p] B) "
        "W))}");
    auto nested = plan(
        "rule r {(conv[@outer] (conv[@inner] (add A B) W) V) => "
        "(add (conv[@outer] (conv[@inner] A W) V) (conv[@outer] (conv[@inner] "
        "B W) V))}");
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
    for (std::size_t spatial = 1; spatial <= 3; ++spatial) {
      for (int empty = 0; empty < 4; ++empty) {
        List xs{empty == 1 ? 0 : 2}, ws;
        for (std::size_t i = 0; i < spatial; ++i) {
          xs.push_back(3 + i % 2);
          ws.push_back(2);
        }
        xs.push_back(empty == 2 ? 0 : 2);
        ws.push_back(xs.back());
        ws.push_back(empty == 3 ? 0 : 2);
        auto a = data(xs, 1), b = data(xs, 2), w = data(ws, 3), v = data(ws, 4);
        compare(input, {a, b, w}, convolution(add(a, b), w));
        compare(kernel, {a, w, v}, convolution(a, add(w, v)));
        List permutation(xs.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::reverse(permutation.begin() + 1, permutation.end() - 1);
        compare(composition, {a, b, w},
                convolution(transpose(add(a, b), permutation), w), permutation);
        List outer_shape(spatial, 2);
        outer_shape.push_back(ws.back());
        outer_shape.push_back(2);
        auto outer = data(outer_shape, 5);
        compare(nested, {a, b, w, outer},
                convolution(convolution(add(a, b), w), outer));
      }
    }
    // Per-axis coefficients, uneven division, and composition are checked
    // against physical execution rather than just a symbolic proof.
    for (std::size_t spatial = 1; spatial <= 3; ++spatial) {
      List xs{1}, ws, strides, dilations;
      for (std::size_t i = 0; i < spatial; ++i) {
        xs.push_back(7 + i);
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
    for (std::size_t spatial = 1; spatial <= 2; ++spatial) {
      List xs{1}, ws(spatial, 2), vs(spatial, 2);
      for (std::size_t i = 0; i < spatial; ++i) xs.push_back(12 + i);
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
                 "{dtype(t){yield t;} "
                 "axes {bind S=group(shape(x));yield S;} semantics {yield "
                 "Read(x,Coords(S));}}}");
    proved(explicit_identity);
    auto sampling_definitions = dialect + R"(
      dialect Sampling {
        op sample(input:tensor)->tensor {
          attrs {steps:index[];}
          dtype(t){yield t;}
          axes {
            assert len(attrs.steps) == rank(input);
            assert is_positive_extents(attrs.steps);
            assert is_positive_extents(shape(input));
            bind Points = group(offset_extents(
                extent_floor_div(offset_extents(shape(input), -1), attrs.steps), 1));
            yield Points;
          }
          semantics {yield Read(input,IndexScale(attrs.steps,Coords(Points)));}
        }
      })";
    auto sampling = plan(
        "rule r {(sample[@s] (add A B)) => "
        "(add (sample[auto] A) (sample[auto] B))}",
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
    auto identity = plan("rule r {(conv[@c] A W) => (conv[auto] A W)}");
    proved(identity);
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
    auto wrong = plan("rule r {(conv[@c] (add A B) W) => (conv[@c] A W)}");
    expect(wrong.proof.status == ProofStatus::Unknown, "missing term rejected");
    auto renamed = dialect;
    auto at = renamed.find("op convolution");
    renamed.replace(at, 14, "op correlation");
    at = renamed.find("alias: conv;");
    renamed.replace(at, 12, "alias: corr;");
    proved(plan(
        "rule r {(corr[@c] (add A B) W) => (add (corr[auto] A W) (corr[auto] "
        "B W))}",
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
    auto end = bad.find("    // Insert axes only.", begin);
    auto operation = bad.substr(begin, end - begin);
    at = operation.find("op convolution");
    operation.replace(at, 14, "op shifted");
    at = operation.find("alias: conv;");
    operation.erase(at, 12);
    auto definitions = dialect + "dialect Wrong {" + operation + "}";
    auto different = plan(
        "rule r {(conv[@c] A W) => (shifted[auto, strides=@c.strides, "
        "dilations=@c.dilations] A W)}",
        definitions);
    expect(different.proof.status == ProofStatus::Unknown,
           "different calculated access rejected");
    expect(verify(different, identity.proof).status == ProofStatus::Unknown,
           "independent verifier rejects different calculated accesses");
    // Changing coefficients can preserve the arithmetic skeleton while
    // changing tensor accesses. Both synthesis and verification must reject it.
    auto tampered = identity.proof;
    auto& guessed = tampered.constructors.begin()->second;
    std::swap(guessed[0], guessed[1]);
    expect(verify(identity, tampered).status == ProofStatus::Unknown,
           "independent verifier rejects tampered inferred coefficients");
    auto partially_pinned = plan(
        "rule r {(conv[@c] (add A B) W) => "
        "(add (conv[auto, strides=@c.strides] A W) "
        "(conv[auto, dilations=@c.dilations] B W))}");
    proved(partially_pinned);
    auto changed = plan(
        "rule r {(conv[@c] A W) => "
        "(conv[auto, strides=@c.dilations, dilations=@c.strides] A W)}");
    expect(changed.proof.status == ProofStatus::Unknown,
           "swapped scales rejected");
    expect(verify(changed, identity.proof).status == ProofStatus::Unknown,
           "independent verifier rejects changed scales");
    auto malformed = tepl::parse(
        "dialect D {op f(x:tensor)->tensor {axes {bind S {x:all_axes(x)};yield "
        "S;} "
        "semantics {yield Read(x,Coords(Missing));}}}");
    expect(malformed.ok() && !tepl::core::analyze(*malformed.program).ok(),
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
    Context coefficient_context{input, 1000000};
    coefficient_context.sourceScope(input.lhs);
    Lowerer source_coordinates{coefficient_context};
    source_coordinates.lower(input.lhs);
    const auto& convolution_node = *std::find_if(
        input.nodes.begin(), input.nodes.end(), [](const Instance& n) {
          return n.descriptor && n.definition && !n.automatic;
        });
    auto stride = coefficient_context.field(convolution_node, 0);
    auto dilation = coefficient_context.field(convolution_node, 1);
    auto hole = make(Symbol::Kind::Hole, "unknown", 101, 0);
    auto result = matchIndexCoefficients(
        coefficient_context, indexScale(coefficient_context, hole, h),
        indexScale(coefficient_context, stride, h));
    expect(
        result && result->size() == 1 && equal(result->begin()->second, stride),
        "direct coefficient matcher constructs a source expression");
    auto scoped = matchIndexCoefficients(
        coefficient_context, indexScale(coefficient_context, hole, h),
        indexScale(coefficient_context, stride, r));
    expect(!scoped,
           "matching refuses equally sized coordinates from different scopes");
    bool conflicting = false, coupled = false;
    try {
      matchIndexCoefficients(
          coefficient_context,
          indexAdd({indexScale(coefficient_context, hole, h),
                    indexScale(coefficient_context, hole, r)}),
          indexAdd({indexScale(coefficient_context, stride, h),
                    indexScale(coefficient_context, dilation, r)}));
    } catch (const ProofFailure& error) {
      conflicting =
          std::string(error.what()).find("conflicting") != std::string::npos;
    }
    expect(conflicting,
           "one field cannot satisfy conflicting coefficient equations");
    try {
      auto other = make(Symbol::Kind::Hole, "unknown", 101, 1);
      matchIndexCoefficients(
          coefficient_context,
          indexScale(coefficient_context, call("extent_mul", {hole, other}), h),
          indexScale(coefficient_context, stride, h));
    } catch (const ProofFailure& error) {
      coupled = std::string(error.what()).find("coupled") != std::string::npos;
    }
    expect(coupled, "coupled coefficient equations are explicitly unsupported");
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
