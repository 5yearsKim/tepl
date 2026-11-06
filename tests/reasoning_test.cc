#include <iostream>
#include <limits>
#include <stdexcept>

#include "src/core/pipeline.h"
#include "src/core/semantics/context.h"
#include "src/core/semantics/symbolic/index_expr.h"
#include "src/core/semantics/symbolic/metadata_sequences.h"
#include "src/core/semantics/symbolic/normalize.h"
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
  auto too_large = make(Symbol::Kind::Integer, "9223372036854775808");
  auto conversion = make(Symbol::Kind::Call, "i64", 0, 0, {too_large});
  auto cancelled =
      make(Symbol::Kind::Call, "-", 0, 0, {conversion, conversion});
  bool rejected = false;
  try {
    c.checkMetadata(cancelled);
  } catch (const ProofFailure&) {
    rejected = true;
  }
  expect(rejected,
         "cancellation cannot hide an overflowing integer conversion");
  expect(!c.integerInterval(call("index", {list({2, 3}), integer(2)})),
         "invalid index has no value");
}
void incrementalFacts() {
  Plan p;
  Context c{p, 1000000};
  auto a = make(Symbol::Kind::Field, "a", 0, 0);
  auto b = make(Symbol::Kind::Field, "b", 0, 1);
  auto d = make(Symbol::Kind::Field, "d", 0, 2);
  c.learn(call("is_positive_extents", {a}));
  c.learn(call("==", {a, b}));
  c.canonical(a);
  c.canonical(call("len", {a}));
  expect(!c.entails(call("is_positive_extents", {d})),
         "an unrelated field has no positivity evidence");
  c.learn(call("==", {b, d}));
  expect(equal(c.canonical(a), c.canonical(d)) &&
             c.entails(call("is_positive_extents", {d})),
         "an alias update refreshes dependent facts and previous query misses");
  c.learn(call("==", {d, list({2, 3})}));
  expect(equal(c.canonical(a), c.canonical(list({2, 3}))) &&
             equal(c.canonical(call("len", {a})), integer(2)),
         "cached aliases follow a later concrete assignment");
  expect(!c.entails(call("==", {a, list({3, 2})})),
         "incremental equality preserves sequence ordering");

  auto shape = make(Symbol::Kind::Shape, {}, 0);
  auto selected = call("gather", {shape, list({0, 1})});
  c.learn(call("is_positive_extents", {selected}));
  c.canonical(selected);
  c.learn(call("==", {call("len", {shape}), integer(2)}));
  expect(c.entails(call("is_positive_extents", {shape})),
         "new length evidence refreshes facts about a cached selection");
  auto other = make(Symbol::Kind::Shape, {}, 1);
  expect(!c.entails(call("is_positive_extents", {other})),
         "dependency updates do not leak facts to another shape");

  auto hole = make(Symbol::Kind::Hole, {}, 0, 0);
  auto dependent = make(Symbol::Kind::Hole, {}, 1, 0);
  c.assignAttribute(1, 0, call("concat", {hole, list({1})}));
  expect(containsHole(c.resolve(dependent)),
         "an unassigned constructor dependency remains unresolved");
  c.assignAttribute(0, 0, list({0}));
  expect(equal(c.resolve(dependent), list({0, 1})),
         "attribute assignment invalidates transitive resolution caches");
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
    c.checkMetadataType(safe, tepl::core::Type{tepl::core::TypeKind::kU64});
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
void selectedShapeEquality() {
  Plan p;
  Context c{p, 1000000};
  auto a = make(Symbol::Kind::Shape, {}, 0);
  auto b = make(Symbol::Kind::Shape, {}, 1);
  for (const auto& shape : {a, b})
    c.learn(call("==", {call("len", {shape}), integer(3)}));
  auto element = [](const S& shape, std::size_t i) {
    return call("index", {shape, integer(i)});
  };
  auto lhs = call("gather", {a, list({0, 2})});
  auto rhs = call("gather", {b, list({0, 1})});
  auto equality = call("==", {lhs, rhs});
  c.learn(call("==", {element(a, 0), element(b, 0)}));
  expect(!c.entails(equality), "one matching dimension cannot prove a pair");
  c.learn(call("==", {element(a, 2), element(b, 1)}));
  expect(c.entails(equality),
         "matching dimensions establish equality of selected shape lists");
  expect(c.entails(call(
             "==", {a, make(Symbol::Kind::List, {}, 0, 0,
                            {element(a, 0), element(a, 1), element(a, 2)})})),
         "shape reconstruction preserves element equalities");
  expect(!c.entails(call("==", {lhs, call("gather", {b, list({1, 0})})})),
         "equal shape lists still require matching element order");
  expect(!c.entails(call("==", {lhs, call("gather", {b, list({0})})})),
         "equal shape lists still require matching lengths");
  expect(!c.entails(call("==", {a, b})),
         "matching selected dimensions do not identify whole shapes");
  auto intermediate = call("concat", {lhs, call("gather", {b, list({2})})});
  expect(c.entails(
             call("==", {call("gather", {intermediate, list({0, 1})}), rhs})),
         "selected intermediate shapes project through concatenation");
  expect(equal(c.canonical(call("index", {intermediate, integer(2)})),
               c.canonical(element(b, 2))),
         "concatenation indexing crosses an extent block boundary");
  expect(!c.integerInterval(call("index", {intermediate, integer(3)})),
         "out-of-range concatenation indices remain invalid");
  auto axes_a = make(Symbol::Kind::AxisGroup, "first", 0, 0, {lhs});
  auto axes_b = make(Symbol::Kind::AxisGroup, "second", 1, 0, {rhs});
  expect(!c.entails(call("==", {axes_a, axes_b})),
         "equal extents never identify unrelated coordinate frames");
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
  auto bound = call(
      "all", {call("sequence_map", {a, v, call("<=", {v, integer(100)})})});
  c.learn(bound);
  auto hundred =
      call("offset_extents", {call("extent_sub", {a, a}), integer(100)});
  expect(c.entails(call("extent_le", {a, hundred})),
         "explicit pointwise upper bounds apply to whole sequences");
  expect(!c.entails(call("extent_le", {hundred, a})),
         "an upper bound does not imply its converse");
  expect(truth(c.canonical(call(
             "all", {call("sequence_map",
                          {list({}), v, call("<=", {v, integer(100)})})}))),
         "a pointwise bound on an empty sequence is true");
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
void uniformVectors() {
  Plan p;
  Context c{p, 1000000};
  auto shape = make(Symbol::Kind::Shape, {}, 0);
  auto zero = call("extent_sub", {shape, shape});
  auto one = call("offset_extents", {zero, integer(1)});
  auto two = call("extent_add", {one, one});
  auto negative =
      call("offset_extents", {zero, make(Symbol::Kind::Integer, "-1")});
  expect(equal(c.uniformElement(zero), integer(0)), "symbolic zero vector");
  expect(equal(c.uniformElement(one), integer(1)), "symbolic unit vector");
  expect(equal(c.uniformElement(two), integer(2)), "uniform vector addition");
  expect(equal(c.uniformElement(call("extent_mul", {two, two})), integer(4)),
         "uniform vector multiplication");
  expect(c.entails(call("is_positive_extents", {one})),
         "unit vector is positive at arbitrary length");
  expect(!c.entails(call("is_positive_extents", {zero})),
         "unknown length zero vector is not necessarily positive");
  expect(!c.entails(call("is_valid_extents", {negative})),
         "negative uniform vector is not necessarily nonnegative");
  expect(!c.uniformElement(shape), "arbitrary extents are not uniform");
  auto axes = make(Symbol::Kind::AxisGroup, "frame", 0, 0, {shape});
  auto index = coordinates(axes);
  expect(equalIndex(c, indexScale(c, one, index), index),
         "symbolic unit scaling preserves coordinates");
  expect(equalIndex(c, indexOffset(c, zero, index), index),
         "symbolic zero offset preserves coordinates");
  auto other = make(Symbol::Kind::Shape, {}, 1);
  auto other_one =
      call("offset_extents", {call("extent_sub", {other, other}), integer(1)});
  expect(!equalIndex(c, indexScale(c, other_one, index), index),
         "unit scaling cannot erase an unproved length match");
  auto selector = make(Symbol::Kind::Field, "selector", 0, 0);
  c.learn(call("is_valid_axis_list", {selector, call("len", {shape})}));
  expect(equal(c.uniformElement(call("gather", {one, selector})), integer(1)),
         "legal selection preserves uniform elements");
  expect(equal(c.uniformElement(call("concat", {one, other_one})), integer(1)),
         "concatenation preserves a common uniform value");
  expect(!c.uniformElement(call("concat", {one, two})),
         "different uniform values cannot be merged");
  auto partial = make(Symbol::Kind::Field, "partial", 9, 0);
  c.learn(call("==", {call("len", {partial}), integer(3)}));
  for (int i : {0, 1})
    c.learn(call("==", {call("index", {partial, integer(i)}), integer(1)}));
  expect(equal(c.uniformElement(call("gather", {partial, list({1, 0})})),
               integer(1)),
         "individual element facts establish a uniform selected tuple");
  expect(!c.uniformElement(call("gather", {partial, list({1, 2})})),
         "an unconstrained selected element prevents uniformity");
  expect(!c.uniformElement(call("gather", {partial, list({3})})),
         "out-of-range selected elements do not establish uniformity");
}
void emptyAxes() {
  Plan p;
  Context c{p, 1000000};
  auto shape = make(Symbol::Kind::Shape, {}, 0);
  auto group = make(Symbol::Kind::AxisGroup, "group", 0, 0, {shape});
  auto parent = make(Symbol::Kind::AxisGroup, "parent", 1, 0,
                     {make(Symbol::Kind::Shape, {}, 1)});
  auto selector = make(Symbol::Kind::Field, "selector", 0, 0);
  c.bind(parent, group, selector);
  c.learn(call("==", {call("len", {group}), call("len", {selector})}));
  expect(!callIs(c.canonical(group), "empty_axes"),
         "unknown cardinality is preserved");
  c.learn(call("==", {selector, list({})}));
  expect(callIs(c.canonical(group), "empty_axes"),
         "registered view normalizes after its cardinality becomes zero");
  expect(equal(c.canonical(c.positions(parent, group)), list({})),
         "empty group has no positions");
  expect(callIs(c.canonical(call("gather", {group, list({})})), "empty_axes"),
         "selection of an empty group remains empty");
  auto zero_extent =
      make(Symbol::Kind::AxisGroup, "zero_extent", 2, 0, {list({0})});
  expect(!callIs(c.canonical(zero_extent), "empty_axes"),
         "one zero-sized dimension is still one coordinate axis");
}
void registeredAxisEquality() {
  Plan p;
  Context c{p, 1000000};
  auto batch = make(Symbol::Kind::AxisGroup, "batch", 0, 0, {list({7})});
  auto row = make(Symbol::Kind::AxisGroup, "row", 1, 0, {list({7})});
  auto col = make(Symbol::Kind::AxisGroup, "col", 2, 0, {list({7})});
  auto nested = [&](S x) {
    for (int i = 0; i < 3; ++i) {
      auto view = call("gather", {x, list({0})});
      c.bind(x, view, list({0}));
      x = view;
    }
    return x;
  };
  auto b = nested(batch), r = nested(row);
  auto frame = call("concat", {b, r, col});
  auto physical_b = call("filter_axes", {frame, b});
  auto physical_r = call("filter_axes", {frame, r});
  c.bind(frame, physical_b, list({0}));
  c.bind(frame, physical_r, list({1}));
  auto actual = call("concat", {physical_b, physical_r});
  auto expected = call("concat", {batch, row});
  expect(c.entails(call("==", {actual, expected})),
         "registered filters and nested selections preserve coordinate order");
  expect(equal(c.canonical(physical_b), physical_b),
         "equality checking preserves registered coordinate witnesses");
  expect(!c.entails(call("==", {actual, call("concat", {row, batch})})),
         "equal-sized axes cannot be swapped");
  expect(!c.entails(call("==", {actual, call("concat", {batch, col})})),
         "equal-sized axes cannot be substituted");
  c.learn(call("distinct_axes", {frame}));
  expect(equal(c.canonical(c.positions(physical_b, b)), list({0})),
         "an ordered retained group has identity positions in its filter");
  auto reversed = call("concat", {r, b});
  auto physical_pair = call("filter_axes", {frame, reversed});
  c.bind(frame, physical_pair, list({0, 1}));
  expect(
      !equal(c.canonical(c.positions(physical_pair, reversed)), list({0, 1})),
      "a filter cannot claim identity positions for a reversed group");
  expect(!c.entails(call("==", {physical_pair, reversed})),
         "physical filtering does not inherit the requested group's order");
  auto permutation = list({1, 0, 2});
  auto permuted = call("gather", {frame, permutation});
  c.bind(frame, permuted, permutation);
  auto selected = call("filter_axes", {permuted, b});
  c.bind(permuted, selected, list({1}));
  expect(c.entails(call("==", {selected, batch})),
         "registered filters project through a permuted concatenation");
  auto selected_pair = call("filter_axes", {permuted, call("concat", {b, r})});
  c.bind(permuted, selected_pair, list({0, 1}));
  expect(c.entails(call("==", {selected_pair, call("concat", {row, batch})})),
         "a permuted filter retains its parent's physical axis order");
  expect(!c.entails(call("==", {selected_pair, expected})),
         "filtering a permutation cannot silently restore the original order");
}
void registeredAxisMembership() {
  Plan p;
  Context c{p, 1000000};
  auto a_shape = make(Symbol::Kind::Shape, {}, 0);
  auto b_shape = make(Symbol::Kind::Shape, {}, 1);
  auto d_shape = make(Symbol::Kind::Shape, {}, 2);
  auto a = make(Symbol::Kind::AxisGroup, "a", 0, 0, {a_shape});
  auto b = make(Symbol::Kind::AxisGroup, "b", 1, 0, {b_shape});
  auto d = make(Symbol::Kind::AxisGroup, "d", 2, 0, {d_shape});
  auto frame = call("concat", {a, b, d});
  auto selected = call("gather", {frame, list({3, 1})});
  auto first = call("gather", {b, list({0})});
  auto expected = call("concat", {d, first});
  c.enableSetReasoning();
  expect(!c.entails(call("same_set", {selected, expected})),
         "unknown block lengths do not justify a concrete projection");
  c.learn(call("==", {call("len", {a_shape}), integer(1)}));
  c.learn(call("==", {call("len", {b_shape}), integer(2)}));
  c.learn(call("==", {call("len", {d_shape}), integer(1)}));
  auto second = call("gather", {b, list({1})});
  c.partition(b, {first, second}, {list({0}), list({1})});
  c.enableSetReasoning();
  c.bind(frame, selected, list({3, 1}));
  expect(c.entails(call("same_set", {selected, expected})),
         "membership projects registered selections into context-sized blocks");
  expect(equal(c.canonical(selected), selected),
         "membership reasoning preserves the original coordinate witness");
  auto removed = call("gather", {frame, list({0, 2})});
  expect(c.entails(call("same_set",
                        {selected, call("difference_axes", {frame, removed})})),
         "projected selections compose with complement membership");
  expect(!c.entails(call("same_set", {selected, call("concat", {d, second})})),
         "membership distinguishes positions within a block");
  expect(!c.entails(call("==", {selected, call("concat", {first, d})})),
         "set equality never proves a reversed coordinate order");
  auto invalid = call("gather", {frame, list({4, 1})});
  expect(!c.entails(call("same_set", {invalid, expected})),
         "out-of-range selection cannot establish membership equality");
  auto alias = make(Symbol::Kind::AxisGroup, "alias", 3, 0, {a_shape});
  c.learn(call("==", {alias, a}));
  auto filtered = call("filter_axes", {alias, a});
  c.bind(alias, filtered, list({0}));
  auto alias_selection =
      call("gather", {call("concat", {filtered, d}), list({0})});
  c.bind(call("concat", {filtered, d}), alias_selection, list({0}));
  expect(c.entails(call("same_set", {alias_selection, a})),
         "view expansion respects established coordinate aliases");
  expect(
      !c.entails(call("same_set", {alias_selection, call("empty_axes", {})})),
      "coordinate aliases cannot turn a retained group into an empty set");
}
void symbolicBlockSelections() {
  Plan p;
  Context c{p, 1000000};
  auto a = make(Symbol::Kind::Shape, {}, 0);
  auto b = make(Symbol::Kind::Shape, {}, 1);
  auto d = make(Symbol::Kind::Shape, {}, 2);
  auto length = [](const S& x) { return call("len", {x}); };
  auto block = [](const S& start, const S& size) {
    return call("offset_positions", {call("range", {size}), start});
  };
  auto ab = call("+", {length(a), length(b)});
  auto bc = call("+", {length(b), length(d)});
  auto adjacent = call(
      "concat", {block(integer(0), length(a)), block(length(a), length(b))});
  expect(c.entails(call("==", {adjacent, call("range", {ab})})),
         "adjacent symbolic ranges merge without enumerating rank");
  auto all = call("concat", {a, b, d});
  expect(c.entails(call("==", {call("gather", {all, block(length(a), bc)}),
                               call("concat", {b, d})})),
         "a symbolic suffix can span several blocks");
  auto gap =
      call("concat", {block(integer(0), length(a)), block(ab, length(d))});
  expect(!c.entails(call("==", {gap, call("range", {bc})})),
         "gaps cannot become a contiguous range");
  c.learn(call("==", {length(a), integer(1)}));
  c.learn(call("==", {length(b), integer(2)}));
  c.learn(call("==", {length(d), integer(1)}));
  auto expected = call("concat", {d, call("gather", {b, list({1, 0})})});
  expect(
      c.entails(call("==", {call("gather", {all, list({3, 2, 1})}), expected})),
      "numeric sequences use context-known block lengths and retain order");
  expect(!c.entails(call("==", {call("gather", {all, list({4})}), d})),
         "out-of-range numeric projections stay unresolved");
  auto permutation = make(Symbol::Kind::Field, "permutation", 4, 0);
  c.learn(call("axis_partition", {permutation, integer(3)}));
  c.learn(call("==", {length(permutation), integer(3)}));
  auto permuted_prefix =
      call("concat", {permutation, block(integer(3), integer(2))});
  expect(!c.entails(call("==", {permuted_prefix, call("range", {integer(5)})})),
         "a permutation of a block does not establish its order");
  expect(!c.entails(call("==", {call("order_positions", {permutation}),
                                call("range", {integer(3)})})),
         "sorting an unknown permutation does not have an identity index map");
  auto values = make(Symbol::Kind::Field, "values", 5, 0);
  c.learn(call("is_valid_axis_list", {list({0, 1}), length(values)}));
  auto tail_shape = make(Symbol::Kind::Shape, {}, 6);
  auto tail_size = length(tail_shape);
  c.learn(call("==", {length(values), call("+", {integer(2), tail_size})}));
  c.learn(call("==", {call("index", {values, integer(0)}), integer(7)}));
  auto tail = call("gather", {values, block(integer(2), tail_size)});
  auto rebuilt = call("concat", {list({7, 8}), tail});
  expect(!c.entails(call("==", {values, rebuilt})),
         "a symbolic suffix does not prove an unestablished prefix");
  c.learn(call("==", {call("index", {values, integer(1)}), integer(8)}));
  expect(c.entails(call("==", {values, rebuilt})),
         "prefix facts and a symbolic suffix reconstruct an arbitrary-length "
         "sequence");
}
void literalComplementSelections() {
  Plan p;
  Context c{p, 1000000};
  auto middle = make(Symbol::Kind::Shape, {}, 0);
  auto tail = make(Symbol::Kind::Shape, {}, 1);
  c.learn(call("==", {call("len", {middle}), integer(1)}));
  auto values = call("concat", {list({7}), middle, tail});
  auto complement = [&](const S& base, const S& removed) {
    return call("ordered_complement",
                {call("range", {call("len", {base})}), removed});
  };
  expect(c.entails(call(
             "==", {call("gather", {values, complement(values, list({1}))}),
                    call("concat", {list({7}), tail})})),
         "a literal exclusion preserves an arbitrary-length suffix");
  auto prefix = call("concat", {list({7, 8, 9}), tail});
  expect(
      c.entails(call(
          "==", {call("gather", {prefix, complement(prefix, list({2, 0, 2}))}),
                 call("concat", {list({8}), tail})})),
      "complement projections deduplicate exclusions and preserve order");
  expect(!projectSelection(c, values, complement(values, list({5}))),
         "an exclusion inside an unknown-size block stays unresolved");
  expect(!projectSelection(
             c, values,
             call("ordered_complement",
                  {call("range", {call("len", {tail})}), list({1})})),
         "complement projection requires the original universe length");
  expect(!projectSelection(
             c, call("concat", {list({7}), middle}),
             complement(call("concat", {list({7}), middle}), list({2}))),
         "out-of-range exclusions do not produce a projection witness");
  auto bound = call("+", {integer(2), call("len", {tail})});
  expect(c.entails(call("is_valid_axis_list", {list({1, 0}), bound})),
         "distinct literal positions use a symbolic rank lower bound");
  expect(!c.entails(call("is_valid_axis_list", {list({1, 1}), bound})),
         "a symbolic rank cannot make duplicate positions valid");
  expect(!c.entails(call("is_valid_axis_list", {list({2}), integer(2)})),
         "the selector upper bound remains strict");

  auto shape = make(Symbol::Kind::Shape, {}, 2);
  auto frame = make(Symbol::Kind::AxisGroup, "frame", 2, 0, {shape});
  c.learn(call("==", {call("len", {shape}), integer(2)}));
  c.learn(call("distinct_axes", {frame}));
  auto dropped = call("gather", {frame, list({0})});
  auto kept = call("gather", {frame, list({1})});
  expect(c.entails(call("same_set",
                        {kept, call("difference_axes", {frame, dropped})})),
         "a positional complement equals removing selected distinct axes");
  expect(!c.entails(call("same_set", {dropped, kept})),
         "equal axis extents never identify different physical positions");
  auto duplicate = call("concat", {dropped, dropped});
  auto occurrence = call("gather", {duplicate, list({0})});
  expect(
      !c.entails(call("==", {call("difference_axes", {duplicate, occurrence}),
                             call("gather", {duplicate, list({1})})})),
      "position removal cannot replace set difference on repeated axes");
}
void singletonReductions() {
  for (std::size_t extent : {0, 1, 2}) {
    Plan p;
    p.algebraic = true;
    Context c{p, 1000000};
    auto domain = list({extent});
    auto frame = make(Symbol::Kind::AxisGroup, "bound", 0, 0, {domain});
    auto read = std::make_shared<Expr>(Expr{Expr::Kind::Read, 0, 0, frame});
    auto sum = expression(Expr::Kind::Reduce, {read}, frame);
    auto terms = normalize(c, sum);
    expect(terms.size() == 1, "one summand has one normalized term");
    expect(callIs(terms[0].reductions, "empty_axes") == (extent == 1),
           "only a unit domain may lose its reduction binder");
    expect(equal(terms[0].reads[0]->axes, frame),
           "singleton reduction preserves the physical input read frame");
  }
  Plan p;
  p.algebraic = true;
  Context c{p, 1000000};
  auto domain = make(Symbol::Kind::Shape, {}, 0);
  auto frame = make(Symbol::Kind::AxisGroup, "unknown", 0, 0, {domain});
  auto read = std::make_shared<Expr>(Expr{Expr::Kind::Read, 0, 0, frame});
  auto sum = expression(Expr::Kind::Reduce, {read}, frame);
  expect(!callIs(normalize(c, sum)[0].reductions, "empty_axes"),
         "an unestablished singleton domain retains its binder");
  auto shapes = make(Symbol::Kind::Shape, {}, 1);
  c.learn(call("==", {call("len", {shapes}), integer(2)}));
  auto selection = call("gather", {shapes, list({0})});
  expect(!c.entails(call("==", {selection, list({1})})),
         "an unknown selected extent is not assumed to be one");
  c.learn(call("==", {call("index", {shapes, integer(0)}), integer(1)}));
  expect(c.entails(call("==", {selection, list({1})})),
         "established scalar extent facts determine a metadata selection");
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
  auto bad_gather =
      make(Symbol::Kind::Call, "gather", 0, 0, {list({2}), list({1})});
  auto cancelled_vector =
      make(Symbol::Kind::Call, "extent_sub", 0, 0, {bad_gather, bad_gather});
  failed = false;
  try {
    c.checkMetadata(cancelled_vector);
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "uniform zero reasoning cannot erase an invalid gather");
  auto mismatched =
      make(Symbol::Kind::Call, "extent_add", 0, 0, {list({1}), list({1, 1})});
  failed = false;
  try {
    c.checkMetadata(mismatched);
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "uniform arithmetic cannot erase a vector length mismatch");
  auto huge = call("offset_extents", {list({0}), max});
  auto overflowing_vector =
      make(Symbol::Kind::Call, "extent_add", 0, 0, {huge, huge});
  expect(!c.uniformElement(overflowing_vector),
         "overflow is not a uniform literal coefficient");
  failed = false;
  try {
    c.checkMetadata(overflowing_vector);
  } catch (const ProofFailure&) {
    failed = true;
  }
  expect(failed, "uniform vector overflow remains rejected");
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
      "dialect Q {op reorder(x:tensor)->tensor {attrs {p: u64[] } "
      "dtype(t){yield t } axes {bind S{x:all_axes(x)} assert "
      "is_valid_axis_list(attrs.p,rank(x)) assert len(attrs.p)==rank(x) yield "
      "permute(S,attrs.p) } semantics {yield x }}}";
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
      "rule unrelated_name {(a: Q.reorder X)=>(b: Q.reorder X) derive "
      "{b.attrs={p:[x for x in a.attrs.p]} }}",
      true);
  check(
      "rule indexing {X:[N] (a: Q.reorder X)=>(b: Q.reorder X) derive "
      "{b.attrs={p:[a.attrs.p[0]]} }}",
      true);
  check(
      "rule bad {X:[N] (a: Q.reorder X)=>(b: Q.reorder X) derive "
      "{b.attrs={p:[a.attrs.p[1]]} }}",
      false);
  check(
      "rule wide_intermediate {X:[N] (a: Q.reorder X)=>(b: Q.reorder X) derive "
      "{let t=18446744073709551615+1 b.attrs={p:[a.attrs.p[0]+(t-t)]} }}",
      true);
  check(
      "rule bad_overflow {X:[N] (a: Q.reorder X)=>(b: Q.reorder X) derive {let "
      "t=18446744073709551615*18446744073709551615 b.attrs={p:[t-t]} }}",
      false);
}

}  // namespace
int main() {
  try {
    arithmetic();
    incrementalFacts();
    bounds();
    shapes();
    selectedShapeEquality();
    registeredAxisEquality();
    registeredAxisMembership();
    symbolicBlockSelections();
    literalComplementSelections();
    singletonReductions();
    sequences();
    uniformVectors();
    emptyAxes();
    definedness();
    reflections();
    pipeline();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
