#include <iostream>
#include <numeric>
#include <stdexcept>

#include "src/core/semantics/index_maps/axis_bindings.h"
#include "src/core/semantics/index_maps/axis_sets.h"

using namespace tepl::core::semantics;
namespace {
void expect(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
S list(std::initializer_list<std::size_t> values) {
  std::vector<S> result;
  for (auto v : values) result.push_back(integer(v));
  return make(Symbol::Kind::List, {}, 0, 0, std::move(result));
}
void maps() {
  AxisBindings bindings;
  auto parent =
      make(Symbol::Kind::AxisGroup, "parent", 0, 0, {list({2, 3, 4, 5, 6})});
  auto destination = make(Symbol::Kind::AxisGroup, "destination", 1, 0,
                          {list({6, 5, 4, 3, 2})});
  auto split = bindings.split(parent, list({3, 1}));
  expect(equal(split.selected_positions, list({3, 1})), "selected order");
  expect(equal(split.remaining_positions, list({0, 2, 4})), "remaining order");
  bindings.split(parent, list({3, 1}));
  expect(bindings.splits().size() == 1, "interned split");
  expect(!bindings.lookup(destination, split.selected), "unresolved map");
  bindings.bind(destination, parent, list({4, 3, 2, 1, 0}));
  expect(equal(bindings.lookup(destination, split.selected), list({1, 3})),
         "composed map invalidates an earlier miss");
  expect(equal(bindings.tensorInterface(destination).lookup(split.selected),
               list({1, 3})),
         "intermediate tensor publishes subgroup map");
  auto other =
      make(Symbol::Kind::AxisGroup, "other", 2, 0, {list({2, 3, 4, 5, 6})});
  expect(!bindings.lookup(destination, other),
         "coordinate scope distinct from extent");
  expect(
      equal(bindings.split(parent, list({})).selected, call("empty_axes", {})),
      "empty selection");
  // A subgroup registered after its parent was mapped must still compose.
  auto later = bindings.split(parent, list({4, 0}));
  expect(equal(bindings.lookup(destination, later.selected), list({0, 4})),
         "late subgroup composition");
  auto in_selection = AxisBindings::restrict(list({0, 4}), list({4, 2, 0}));
  expect(equal(in_selection, list({2, 0})),
         "restriction preserves group order");
  expect(
      equal(AxisBindings::compose(list({4, 2, 0}), in_selection), list({0, 4})),
      "restricted map reconstructs original positions");
  auto ordered = call("concat", {split.selected, split.remaining});
  expect(bindings.orderedParts(ordered).size() == 2,
         "explicit ordered decomposition is reusable");
  auto selected_from_concat = call("gather", {ordered, list({4, 0, 2})});
  expect(equal(bindings.lookup(destination, selected_from_concat),
               list({0, 1, 4})),
         "selection from a composed ordered group reuses parent maps");
  auto p = list({3, 0, 4, 1}), q = list({1, 3, 2});
  auto a = call("gather", {p, call("membership_positions", {p, q})});
  auto b = call("gather", {q, call("matched_positions", {p, q})});
  expect(equal(a, b) && equal(a, list({3, 1})),
         "paired slots retain selector order");
}
void mappingStorage() {
  AxisBindings bindings;
  auto shape = make(Symbol::Kind::Shape, {}, 0);
  auto frame = make(Symbol::Kind::AxisGroup, "frame", 0, 0, {shape});
  auto k = make(Symbol::Kind::AxisGroup, "K", 1, 0, {shape});
  auto i = make(Symbol::Kind::AxisGroup, "I", 2, 0, {shape});
  auto positions = make(Symbol::Kind::Field, "positions", 0, 0);
  auto remaining = call("ordered_complement",
                        {call("range", {call("len", {frame})}), positions});
  // Keep an equivalent expression syntactically distinct to exercise aliases
  // and partition-specific witnesses without expression-factory folding.
  auto alias = make(Symbol::Kind::Call, "gather", 0, 0,
                    {positions, call("range", {call("len", {positions})})});
  expect(!equal(positions, alias), "distinct selector expressions");

  expect(bindings.partition(frame, {k, i}, {positions, remaining}),
         "partition reserves mapping IDs");
  auto first = bindings.partitions().front();
  auto k_id = first.members[0].mapping;
  auto i_id = first.members[1].mapping;
  auto snapshot = bindings.partitionsFor(frame);
  expect(bindings.mappingIds().empty() && !bindings.positions(frame, k) &&
             !bindings.group(frame, positions) && !bindings.lookup(frame, k),
         "reserved members do not publish mappings early");

  expect(bindings.bind(frame, i, remaining) && bindings.bind(frame, k, alias),
         "publish mappings in binding order");
  expect(bindings.mappingIds()[0] == i_id && bindings.mappingIds()[1] == k_id,
         "registration order is independent of reservation order");
  expect(equal(bindings.positions(frame, k), alias),
         "first published selector is preferred");
  expect(equal(bindings.selector(bindings.partitions()[0].members[0]),
               positions) &&
             equal(bindings.selector(snapshot[0].members[0]), positions),
         "partition and snapshot retain the original witness");
  expect(bindings.bind(frame, k, positions), "add reverse selector alias");
  expect(bindings.mappingIds().size() == 2 &&
             equal(bindings.group(frame, positions), k) &&
             equal(bindings.group(frame, alias), k) &&
             equal(bindings.lookup(frame, k), alias),
         "aliases share one preferred mapping and invalidate earlier misses");
  expect(!bindings.bind(frame, k, positions), "repeated alias is a no-op");

  expect(bindings.partition(frame, {k, i}, {alias, remaining}),
         "a partition can use another witness for the same members");
  expect(
      bindings.partitions()[1].members[0].mapping == k_id &&
          !bindings.partitions()[1].members[0].selector &&
          equal(bindings.selector(bindings.partitions()[1].members[0]), alias),
      "partition reuses the preferred mapping without a selector copy");
  auto interface = bindings.tensorInterface(frame);
  expect(equal(interface.lookup(k), alias),
         "tensor interface publishes preferred mapping");
  for (std::size_t n = 0; n < 128; ++n) {
    auto group = make(Symbol::Kind::AxisGroup, "extra", n + 3, 0, {shape});
    bindings.bind(frame, group, make(Symbol::Kind::Field, "extra", n + 3, 0));
  }
  expect(equal(bindings.mapping(k_id).group, k) &&
             equal(bindings.mapping(k_id).positions, alias) &&
             equal(bindings.selector(bindings.partitions()[0].members[0]),
                   positions) &&
             equal(interface.lookup(k), alias),
         "mapping IDs and tensor snapshots survive storage growth");
}
void membership() {
  std::size_t work = 0;
  AxisSets sets([&] { ++work; });
  auto i = sets.atom(list({0})), c = sets.atom(list({1}));
  auto selected = sets.conjunction(i, c);
  auto remaining = sets.conjunction(i, sets.negate(c));
  expect(sets.proves(sets.equivalent(i, sets.disjunction(selected, remaining))),
         "selected and remaining cover parent");
  expect(sets.proves(sets.negate(sets.conjunction(selected, remaining))),
         "selected and remaining disjoint");
  expect(!sets.proves(sets.equivalent(i, c)),
         "unknown membership stays unknown");
  expect(work < 1000, "bounded membership identity calculation");
  sets.assume(0);
  expect(!sets.proves(1),
         "inconsistent membership records cannot certify a rule");
}
}  // namespace
int main() {
  try {
    maps();
    mappingStorage();
    membership();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
