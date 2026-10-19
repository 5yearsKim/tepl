#include <iostream>
#include <limits>
#include <stdexcept>

#include "src/core/pipeline.h"
#include "src/core/semantics/context.h"
#include "src/core/semantics/symbolic/index_expr.h"
#include "src/core/semantics/symbolic/metadata_sequences.h"
#include "src/core/syntax/parse.h"
#include "src/executable/codec.h"

using namespace tepl::core::semantics;
namespace {
void expect(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}
S list(std::initializer_list<std::size_t> xs) {
  std::vector<S> out;
  for (auto n : xs) out.push_back(integer(n));
  return make(Symbol::Kind::List, {}, 0, 0, out);
}
void arithmetic() {
  Plan p;
  Context c{p, 1000000};
  expect(c.entails(call(">=", {integer(0), integer(0)})),
         "constant comparison");
  expect(!c.entails(call(">", {integer(0), integer(0)})),
         "strict inequality remains false");
  auto n = make(Symbol::Kind::Field, "dimension", 0, 0);
  c.learn(call(">=", {n, integer(1)}));
  c.learn(call("<=", {n, integer(100)}));
  expect(c.entails(call(">", {call("+", {n, integer(2)}), integer(0)})),
         "positive addition");
  expect(c.entails(call("<=", {call("*", {n, integer(2)}), integer(200)})),
         "bounded multiplication");
  expect(!c.entails(call("<", {n, integer(100)})),
         "inclusive bound does not prove strict bound");
  auto maximum =
      make(Symbol::Kind::Integer, "170141183460469231731687303715884105727");
  auto overflow = call("+", {maximum, integer(1)});
  expect(callIs(overflow, "+"), "overflow must not constant fold");
  expect(!c.integerInterval(overflow), "overflow has no established interval");
  expect(!c.integerInterval(call("index", {list({2, 3}), integer(2)})),
         "invalid index has no value");
}
void bounds() {
  Plan p;
  Context c{p, 1000000};
  auto a = make(Symbol::Kind::Field, "a", 0, 0),
       b = make(Symbol::Kind::Field, "b", 0, 1),
       d = make(Symbol::Kind::Shape, {}, 0);
  for (const auto& x : {a, b}) {
    c.learn(call("==", {call("len", {x}), call("len", {d})}));
    c.learn(call("is_valid_extents", {x}));
  }
  c.learn(call("extent_le", {call("extent_add", {a, b}), d}));
  expect(extentBoundArithmetic(c, a, d, QueryMode::Entailment),
         "positive source inequality implies bound on a summand");
  expect(!extentBoundArithmetic(c, d, a, QueryMode::Entailment),
         "source inequality cannot be reversed");
  auto wide = make(Symbol::Kind::Integer, "18446744073709551615");
  auto i = make(Symbol::Kind::Field, "position", 1, 0);
  c.learn(call("==", {call("len", {d}), integer(1)}));
  auto dimension = call("index", {d, integer(0)});
  expect(!c.entails(call("<=", {dimension, make(Symbol::Kind::Integer,
                                                "9223372036854775807")})),
         "u64 dimensions are not narrowed to i64");
  expect(c.entails(call("<=", {dimension, wide})), "full u64 shape bound");
  auto safe = make(Symbol::Kind::Call, "+", 0, 0, {wide, integer(1)});
  c.checkMetadata(safe);
  bool failed = false;
  try {
    c.checkMetadataType(safe, tepl::core::Type{tepl::core::TypeKind::kIndex});
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "safe intermediate must still fit its final attribute type");
  for (int lo = -3; lo <= 3; ++lo)
    for (int hi = lo; hi <= 3; ++hi) {
      Plan plan;
      Context context{plan, 100000};
      auto n = make(Symbol::Kind::Field, "n", 0, 0);
      context.learn(
          call(">=", {n, make(Symbol::Kind::Integer, std::to_string(lo))}));
      context.learn(
          call("<=", {n, make(Symbol::Kind::Integer, std::to_string(hi))}));
      auto interval = context.integerInterval(call("*", {n, n}));
      expect(bool(interval), "small product interval is defined");
      for (int x = lo; x <= hi; ++x)
        expect(interval->first <= x * x && interval->second >= x * x,
               "interval contains every concrete product");
    }
}
void shapes() {
  for (std::size_t rank = 0; rank < 12; ++rank) {
    Plan p;
    Context c{p, 1000000};
    auto shape = make(Symbol::Kind::Shape, {}, 0);
    c.learn(call("==", {call("len", {shape}), integer(rank)}));
    expect(
        c.entails(call(
            "==", {call("slice", {shape, integer(0), integer(rank)}), shape})),
        "whole sequence slice");
    for (std::size_t i = 0; i < rank; ++i) {
      auto d = call("index", {shape, integer(i)});
      expect(c.entails(call(">=", {d, integer(0)})),
             "shape element is nonnegative");
      expect(c.entails(call("is_valid_extents",
                            {make(Symbol::Kind::List, {}, 0, 0, {d})})),
             "shape list remains valid");
      expect(!c.entails(call(">", {d, integer(0)})),
             "zero extent remains possible");
    }
    expect(!c.integerInterval(call("index", {shape, integer(rank)})),
           "out-of-range shape index rejected");
  }
}
void sequences() {
  Plan p;
  Instance input;
  input.capture = tepl::core::CaptureId{0};
  p.nodes.push_back(input);
  Context c{p, 1000000};
  c.sourceScope(0);
  auto a = make(Symbol::Kind::Shape, {}, 0),
       b = make(Symbol::Kind::Shape, {}, 1);
  c.learn(call("==", {call("len", {a}), call("len", {b})}));
  auto v = call("sequence_variable", {integer(0)});
  auto identity = call("sequence_map", {a, v, v});
  expect(equal(c.canonical(identity), a),
         "symbolic map identity for arbitrary rank");
  auto count = call("len", {a});
  auto positive = make(Symbol::Kind::Call, "+", 0, 0, {count});
  c.checkMetadata(call("range", {positive}));
  auto indices = call("range", {call("len", {a})});
  auto body = call("+", {call("index", {a, v}), call("index", {b, v})});
  auto mapped = c.canonical(call("sequence_map", {indices, v, body}));
  expect(equal(mapped, c.canonical(call("extent_add", {a, b}))),
         "pointwise map becomes vector arithmetic");
  auto repeat = c.canonical(call("sequence_map", {a, v, integer(1)}));
  expect(
      equal(c.canonical(call("len", {repeat})), c.canonical(call("len", {a}))),
      "constant map preserves symbolic length");
  auto all = c.canonical(
      call("all", {call("sequence_map", {a, v, call(">=", {v, integer(0)})})}));
  expect(c.entails(all), "general all over nonnegative shape elements");
  expect(
      equalExtentArithmetic(
          c, call("extent_mul", {a, call("sequence_map", {a, v, integer(1)})}),
          a),
      "uniform vector one is a coefficient");
  auto unknown = make(Symbol::Kind::Field, "arbitrary_sequence", 0, 0);
  c.learn(
      call("==", {unknown, call("sequence_map", {unknown, v, integer(1)})}));
  expect(c.entails(call("is_positive_extents", {unknown})),
         "uniform sequence facts apply without a fixed rank");
  auto duplicates = list({0, 0, 1});
  auto selected = c.canonical(
      call("sequence_map", {duplicates, v, call("index", {list({7, 9}), v})}));
  expect(equal(selected, list({7, 7, 9})),
         "metadata indexing permits repetition");
}
void definedness() {
  Plan p;
  Instance input;
  input.capture = tepl::core::CaptureId{0};
  p.nodes.push_back(input);
  Context c{p, 1000000};
  c.sourceScope(0);
  auto max =
      make(Symbol::Kind::Integer, "170141183460469231731687303715884105727");
  auto overflow = make(Symbol::Kind::Call, "+", 0, 0, {max, integer(1)});
  auto cancelled = make(Symbol::Kind::Call, "-", 0, 0, {overflow, overflow});
  bool failed = false;
  try {
    c.checkMetadata(cancelled);
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "cancellation cannot erase overflow in a derived attribute");
  auto invalid =
      make(Symbol::Kind::Call, "index", 0, 0, {list({}), integer(0)});
  c.checkMetadata(make(Symbol::Kind::Call, "if", 0, 0,
                       {boolean(true), integer(1), invalid}));
  failed = false;
  try {
    c.checkMetadata(invalid);
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "out of range derived indexing is rejected");
  auto variable = call("sequence_variable", {integer(0)});
  c.checkMetadata(call("sequence_map", {list({}), variable, invalid}));
  failed = false;
  try {
    c.checkMetadata(call("sequence_map", {list({0}), variable, cancelled}));
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "substitution in a comprehension cannot erase overflow");
  failed = false;
  try {
    c.checkMetadata(make(Symbol::Kind::Call, "replace", 0, 0,
                         {list({}), integer(0), integer(1)}));
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "unused invalid list replacement remains rejected");
  auto shape = make(Symbol::Kind::Shape, {}, 0);
  failed = false;
  try {
    c.checkMetadata(call("sequence_map", {shape, variable, cancelled}));
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed,
         "symbolic pointwise lifting cannot erase an invalid intermediate");
  auto minimum =
      make(Symbol::Kind::Integer, "-170141183460469231731687303715884105728");
  failed = false;
  try {
    c.checkMetadata(make(Symbol::Kind::Call, "-", 0, 0, {minimum}));
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "negation overflow remains rejected");
  c.checkMetadata(
      call("sequence_map", {list({0, 1}), variable,
                            make(Symbol::Kind::Call, "-", 0, 0, {variable})}));
}
void reflections() {
  for (std::size_t rank = 0; rank <= 4; ++rank) {
    std::vector<S> sizes;
    for (std::size_t i = 0; i < rank; ++i) sizes.push_back(integer(i + 2));
    auto domain = make(Symbol::Kind::List, {}, 0, 0, sizes);
    auto axes = make(Symbol::Kind::AxisGroup, "scope", 0, 0, {domain});
    for (std::size_t a = 0; a < (1u << rank); ++a)
      for (std::size_t b = 0; b < (1u << rank); ++b) {
        Plan p;
        Context c{p, 1000000};
        std::vector<S> pa, pb, parity;
        for (std::size_t i = 0; i < rank; ++i) {
          if (a & (1u << i)) pa.push_back(integer(i));
          if (b & (1u << i)) pb.push_back(integer(i));
          if ((a ^ b) & (1u << i)) parity.push_back(integer(i));
        }
        auto first = make(Symbol::Kind::List, {}, 0, 0, pa),
             second = make(Symbol::Kind::List, {}, 0, 0, pb);
        auto combined =
            indexReverse(c, indexReverse(c, coordinates(axes), domain, first),
                         domain, second);
        auto expected =
            indexReverse(c, coordinates(axes), domain,
                         make(Symbol::Kind::List, {}, 0, 0, parity));
        expect(equalIndex(c, combined, expected),
               "reflections combine by parity for every selector overlap");
      }
  }
}
void pipeline() {
  const std::string dialect =
      "dialect Q {op reorder(x:tensor)->tensor {attrs {p:index[];} "
      "dtype(t){yield t;} axes {bind S{x:all_axes(x)};assert "
      "is_valid_axis_list(attrs.p,rank(x));assert len(attrs.p)==rank(x);yield "
      "permute(S,attrs.p);} semantics {yield x;}}}";
  auto check = [&](const std::string& rule, bool expected) {
    auto parsed = tepl::parse(dialect + rule);
    expect(parsed.ok(), "metadata regression fixture parses");
    auto result = tepl::core::inspect(
        *parsed.program, {tepl::core::SemanticPolicy::kAlgebraic, true});
    bool proved = result.ok() && result.program->rules[0].semantic_plan &&
                  result.program->rules[0].semantic_plan->proof.status ==
                      ProofStatus::Proved;
    if (proved != expected) {
      for (const auto& d : result.diagnostics) std::cerr << d.message << '\n';
      if (result.ok())
        std::cerr << result.program->rules[0].semantic_plan->proof.reason
                  << '\n';
      throw std::runtime_error(rule);
    }
    if (proved) {
      auto compiled = tepl::core::compile(
          *parsed.program, {tepl::core::SemanticPolicy::kAlgebraic});
      expect(compiled.ok(), "proved metadata rule compiles");
      (void)tepl::executable::load(
          tepl::executable::serialize(*compiled.program));
    }
  };
  check(
      "rule unrelated_name {(Q.reorder[@a] X)=>(Q.reorder[@b] X) derive "
      "{@b={p:[x for x in @a.p]};}}",
      true);
  check(
      "rule indexing {X:[N] (Q.reorder[@a] X)=>(Q.reorder[@b] X) derive "
      "{@b={p:[@a.p[0]]};}}",
      true);
  check(
      "rule bad {X:[N] (Q.reorder[@a] X)=>(Q.reorder[@b] X) derive "
      "{@b={p:[@a.p[1]]};}}",
      false);
  check(
      "rule wide_intermediate {X:[N] (Q.reorder[@a] X)=>(Q.reorder[@b] X) "
      "derive {let t=18446744073709551615+1; @b={p:[@a.p[0]+(t-t)]};}}",
      true);
  check(
      "rule bad_overflow {X:[N] (Q.reorder[@a] X)=>(Q.reorder[@b] X) derive "
      "{let t=18446744073709551615*18446744073709551615; @b={p:[t-t]};}}",
      false);
}

}  // namespace
int main() {
  try {
    arithmetic();
    bounds();
    shapes();
    sequences();
    definedness();
    reflections();
    pipeline();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
