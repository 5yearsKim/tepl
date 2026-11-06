#include "src/core/semantics/context.h"

namespace tepl::core::semantics {
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
      std::vector<S> groups;
      for (const auto& member : p.members) {
        groups.push_back(bindings_.mapping(member.mapping).group);
        selectors.push_back(
            select(call("inverse", {read_map}), bindings_.selector(member)));
      }
      partition(axes, groups, selectors);
    }
  for (const auto& p : producers) {
    std::vector<S> groups, selectors;
    for (const auto& member : p.members) {
      auto group = bindings_.mapping(member.mapping).group;
      selectors.push_back(bindings_.selector(member));
      groups.push_back(replace(group, from, to));
    }
    partition(axes, groups, selectors);
  }
  for (const auto& mapping : bindings_.mappingsFor(read_axes))
    bind(axes, replace(mapping.group, from, to), mapping.positions);
  if (auto old_cover = bindings_.cover(read_axes)) {
    auto transported = replace(old_cover, from, to);
    if (!equal(transported, axes) || !bindings_.cover(axes))
      cover(axes, transported);
  }
  return axes;
}
}  // namespace tepl::core::semantics
