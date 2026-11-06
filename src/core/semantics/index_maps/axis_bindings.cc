#include "src/core/semantics/index_maps/axis_bindings.h"

#include <functional>
#include <stdexcept>

namespace tepl::core::semantics {
S TensorAxisInterface::lookup(const S& group) const {
  for (const auto& mapping : mappings) {
    if (equal(mapping.group, group)) return mapping.positions;
    if (callIs(group, "gather") && equal(mapping.group, group->args[0]))
      return AxisBindings::compose(mapping.positions, group->args[1]);
  }
  return {};
}
bool AxisBindings::orderedView(const S& view, const std::vector<S>& parts) {
  auto decomposition = call("concat", parts);
  if (equal(view, decomposition)) return false;
  return ordered_views_.emplace(key(view), decomposition).second;
}
S AxisBindings::orderedView(const S& view) const {
  auto it = ordered_views_.find(key(view));
  return it == ordered_views_.end() ? S{} : it->second;
}
std::vector<S> AxisBindings::orderedParts(const S& view) const {
  std::vector<S> result;
  std::set<Key> active;
  std::function<void(const S&)> visit = [&](const S& x) {
    if (!active.insert(key(x)).second) {
      result.push_back(x);
      return;
    }
    if (auto decomposition = orderedView(x))
      visit(decomposition);
    else if (callIs(x, "concat"))
      for (const auto& part : x->args) visit(part);
    else if (!callIs(x, "empty_axes"))
      result.push_back(x);
    active.erase(key(x));
  };
  visit(view);
  return result;
}
S AxisBindings::compose(const S& parent, const S& child) {
  return call("gather", {parent, child});
}
S AxisBindings::restrict(const S& group_positions,
                         const S& selected_positions) {
  return matchPositions(group_positions, selected_positions).right_positions;
}
S AxisBindings::select(const S& parent, const S& selector) {
  if (auto existing = group(parent, selector)) return existing;
  auto view = call("gather", {parent, selector});
  bind(parent, view, selector);
  return view;
}
AxisPartition AxisBindings::split(const S& parent, const S& selector) {
  auto identity = std::pair(key(parent), key(selector));
  if (auto it = split_index_.find(identity); it != split_index_.end())
    return splits_[it->second];
  auto rest = call("ordered_complement",
                   {call("range", {call("len", {parent})}), selector});
  auto selected = select(parent, selector);
  auto remaining = select(parent, rest);
  AxisPartition result{parent, selected, remaining, selector, rest};
  split_index_[identity] = splits_.size();
  splits_.push_back(result);
  return result;
}
S AxisBindings::lookup(const S& interface, const S& group) const {
  auto identity = std::pair(key(interface), key(group));
  if (auto it = lookup_cache_.find(identity); it != lookup_cache_.end())
    return it->second;
  std::set<std::pair<Key, Key>> active;
  return lookup_cache_[identity] = lookup(interface, group, active);
}
S AxisBindings::lookup(const S& interface, const S& group,
                       std::set<std::pair<Key, Key>>& active) const {
  auto cached = lookup_cache_.find({key(interface), key(group)});
  if (cached != lookup_cache_.end() && cached->second) return cached->second;
  if (auto p = positions(interface, group)) return p;
  if (equal(interface, group)) return call("range", {call("len", {interface})});
  if (callIs(group, "empty_axes")) return make(Symbol::Kind::List);
  auto id = std::pair(key(interface), key(group));
  if (!active.insert(id).second) return {};
  struct Pop {
    decltype(active)& ids;
    decltype(id) value;
    ~Pop() { ids.erase(value); }
  } pop{active, id};
  if (callIs(group, "gather") && callIs(group->args[0], "concat")) {
    if (auto parent = lookup(interface, group->args[0], active))
      return compose(parent, group->args[1]);
  }
  if (callIs(group, "concat")) {
    std::vector<S> parts;
    for (const auto& g : group->args) {
      auto p = lookup(interface, g, active);
      if (!p) return {};
      parts.push_back(p);
    }
    return call("concat", std::move(parts));
  }
  // Search registered origins breadth first. Each origin is visited once;
  // equivalent paths need no repeated exploration of a cyclic mapping graph.
  struct Origin {
    S axes, positions;
  };
  std::vector<Origin> origins{{group, call("range", {call("len", {group})})}};
  std::set<Key> visited;
  for (std::size_t next = 0; next < origins.size(); ++next) {
    auto origin = origins[next];
    if (!visited.insert(key(origin.axes)).second) continue;
    if (equal(interface, origin.axes)) return origin.positions;
    if (auto parent = positions(interface, origin.axes))
      return compose(parent, origin.positions);
    if (callIs(origin.axes, "gather"))
      origins.push_back({origin.axes->args[0],
                         compose(origin.axes->args[1], origin.positions)});
    if (auto incoming = group_mappings_.find(key(origin.axes));
        incoming != group_mappings_.end())
      for (auto index : incoming->second) {
        const auto& edge = mapping(index);
        if (!equal(edge.interface, origin.axes))
          origins.push_back(
              {edge.interface, compose(edge.positions, origin.positions)});
      }
  }
  if (callIs(interface, "concat")) {
    S offset = integer(0);
    for (const auto& part : interface->args) {
      if (auto p = lookup(part, group, active))
        return call("offset_positions", {p, offset});
      offset = call("+", {offset, call("len", {part})});
    }
  }
  return {};
}
TensorAxisInterface AxisBindings::tensorInterface(const S& axes) const {
  TensorAxisInterface result{axes, {}};
  auto found = interface_mappings_.find(key(axes));
  if (found != interface_mappings_.end()) {
    result.mappings.reserve(found->second.size());
    for (auto id : found->second) {
      const auto& value = mapping(id);
      result.mappings.push_back({value.group, value.positions});
    }
  }
  return result;
}

bool AxisBindings::overlap(const S& left, const S& right, const S& view,
                           const S& slots, const S& paired_slots) {
  if (overlap_views_.contains(key(view))) return false;
  overlap_views_[key(view)] = overlaps_.size();
  overlaps_.push_back({left, right, view, slots, paired_slots});
  lookup_cache_.clear();
  return true;
}
const AxisBindings::Overlap* AxisBindings::overlapFor(const S& view) const {
  auto found = overlap_views_.find(key(view));
  return found == overlap_views_.end() ? nullptr : &overlaps_[found->second];
}
AxisBindings::MappingId AxisBindings::reserveMapping(const S& interface,
                                                     const S& group,
                                                     const S& positions) {
  auto identity = std::pair(key(interface), key(group));
  auto found = by_group_.find(identity);
  if (found != by_group_.end()) return found->second;
  MappingId id{mappings_.size()};
  by_group_.emplace(std::move(identity), id);
  mappings_.push_back({interface, group, positions});
  return id;
}
bool AxisBindings::bind(const S& interface, const S& group, const S& p) {
  auto id = reserveMapping(interface, group, p);
  auto selector_key = std::pair(key(interface), key(p));
  // Preserve the first published witness. Additional selector expressions
  // remain aliases for reverse lookup, without introducing another mapping.
  if (mapping(id).registered) {
    if (by_selector_.contains(selector_key)) return false;
    by_selector_.emplace(std::move(selector_key), id);
    lookup_cache_.clear();
    return true;
  }
  if (!equal(mapping(id).positions, p)) {
    // A partition can reserve this ID before recursive bind calls publish it.
    // Preserve the partition's exact selector if the first published witness
    // differs from the reservation.
    auto prior = interface_partitions_.find(key(interface));
    if (prior != interface_partitions_.end())
      for (auto partition : prior->second)
        for (auto& member : partitions_[partition].members)
          if (member.mapping == id && !member.selector)
            member.selector = mapping(id).positions;
    mappings_[id.value].positions = p;
  }
  mappings_[id.value].registered = true;
  lookup_cache_.clear();
  by_selector_.insert_or_assign(std::move(selector_key), id);
  interface_mappings_[key(interface)].push_back(id);
  group_mappings_[key(group)].push_back(id);
  mapping_order_.push_back(id);
  // Publish surviving components with their parent interface. This is
  // compositional construction, not a search through unrelated map records.
  if (!equal(interface, group)) {
    auto children = mappingsFor(group);
    for (const auto& child : children)
      if (!equal(child.group, group) && !equal(child.group, interface))
        bind(interface, child.group, compose(p, child.positions));
    if (callIs(group, "concat")) {
      S offset = integer(0);
      for (const auto& part : group->args) {
        bind(interface, part,
             compose(p, call("offset_positions",
                             {call("range", {call("len", {part})}), offset})));
        offset = call("+", {offset, call("len", {part})});
      }
    }
  }
  return true;
}
bool AxisBindings::partition(const S& interface, const std::vector<S>& groups,
                             const std::vector<S>& selectors) {
  if (groups.size() != selectors.size())
    throw std::invalid_argument("partition groups and selectors must match");
  auto identity = key(interface) + ":" + key(call("concat", groups)) + ":" +
                  key(call("concat", selectors));
  if (!partition_keys_.insert(identity).second) return false;
  Partition result{interface, {}};
  result.members.reserve(groups.size());
  for (std::size_t i = 0; i < groups.size(); ++i) {
    auto id = reserveMapping(interface, groups[i], selectors[i]);
    result.members.push_back(
        {id, equal(mapping(id).positions, selectors[i]) ? S{} : selectors[i]});
  }
  interface_partitions_[key(interface)].push_back(partitions_.size());
  partitions_.push_back(std::move(result));
  return true;
}
bool AxisBindings::cover(const S& interface, const S& groups) {
  auto found = covers_.find(key(interface));
  if (found != covers_.end() && equal(found->second, groups)) return false;
  covers_[key(interface)] = groups;
  return true;
}
bool AxisBindings::beginTransport(const S& from, const S& to) {
  return transported_frames_.emplace(key(from), key(to)).second;
}
S AxisBindings::positions(const S& interface, const S& group) const {
  auto found = by_group_.find({key(interface), key(group)});
  if (found == by_group_.end()) return {};
  const auto& value = mapping(found->second);
  return value.registered ? value.positions : S{};
}
S AxisBindings::group(const S& interface, const S& positions) const {
  auto found = by_selector_.find({key(interface), key(positions)});
  return found == by_selector_.end() ? S{} : mapping(found->second).group;
}
S AxisBindings::cover(const S& interface) const {
  auto found = covers_.find(key(interface));
  return found == covers_.end() ? S{} : found->second;
}
std::vector<AxisBindings::Mapping> AxisBindings::mappingsFor(
    const S& interface) const {
  std::vector<Mapping> result;
  auto found = interface_mappings_.find(key(interface));
  if (found != interface_mappings_.end()) {
    result.reserve(found->second.size());
    for (auto id : found->second) result.push_back(mapping(id));
  }
  return result;
}
std::vector<AxisBindings::Partition> AxisBindings::partitionsFor(
    const S& interface) const {
  std::vector<Partition> result;
  auto found = interface_partitions_.find(key(interface));
  if (found != interface_partitions_.end()) {
    result.reserve(found->second.size());
    for (auto id : found->second) {
      auto snapshot = partitions_[id];
      for (auto& member : snapshot.members) member.selector = selector(member);
      result.push_back(std::move(snapshot));
    }
  }
  return result;
}
}  // namespace tepl::core::semantics
