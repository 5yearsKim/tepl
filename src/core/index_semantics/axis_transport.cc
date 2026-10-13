#include "src/core/index_semantics/context.h"

namespace tepl::core::index_semantics {
void Context::transportFrame(const S& from, const S& to) {
  if (!callIs(from, "concat") || !bindings_.beginTransport(from, to)) return;
  std::vector<S> groups, selectors;
  for (const auto& part : from->args) {
    auto p = positions(from, part);
    groups.push_back(select(to, p));
    selectors.push_back(p);
  }
  partition(to, groups, selectors);
}

S Context::transportInterface(const S& from, const S& to, const S& read_axes) {
  auto axes = replace(read_axes, from, to);
  // Snapshot before transport adds more records. A producer read may see its
  // interface through a permutation: compose that map with consumer selectors.
  auto consumers = bindings_.partitionsFor(to);
  auto producers = bindings_.partitionsFor(read_axes);
  auto read_map = positions(from, read_axes);
  if (evidence(call("axis_partition", {read_map, call("len", {from})})))
    for (const auto& p : consumers) {
      std::vector<S> selectors;
      for (const auto& s : p.selectors)
        selectors.push_back(select(call("inverse", {read_map}), s));
      partition(axes, p.groups, selectors);
    }
  for (const auto& p : producers) {
    std::vector<S> groups;
    for (const auto& g : p.groups) groups.push_back(replace(g, from, to));
    partition(axes, groups, p.selectors);
  }
  for (const auto& w : bindings_.wiringFor(read_axes))
    bind(axes, replace(w.group, from, to), w.positions);
  if (auto old_cover = bindings_.cover(read_axes)) {
    auto transported = replace(old_cover, from, to);
    if (!equal(transported, axes) || !bindings_.cover(axes))
      cover(axes, transported);
  }
  return axes;
}
}  // namespace tepl::core::index_semantics
