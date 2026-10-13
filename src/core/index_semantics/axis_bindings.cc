#include "src/core/index_semantics/axis_bindings.h"

#include <functional>

namespace tepl::core::index_semantics {
S TensorAxisInterface::lookup(const S& group) const {
  for (const auto& mapping : mappings) {
    if (equal(mapping.source, group)) return mapping.positions;
    if (callIs(group, "gather") && equal(mapping.source, group->args[0]))
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
  std::set<std::string> active;
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
  bind(parent, result.selected, selector);
  bind(parent, result.remaining, rest);
  return result;
}
S AxisBindings::lookup(const S& interface, const S& group) const {
  auto identity = std::pair(key(interface), key(group));
  if (auto it = lookup_cache_.find(identity); it != lookup_cache_.end())
    return it->second;
  std::set<std::pair<std::string, std::string>> active;
  return lookup_cache_[identity] = lookup(interface, group, active);
}
S AxisBindings::lookup(
    const S& interface, const S& group,
    std::set<std::pair<std::string, std::string>>& active) const {
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
  std::set<std::string> visited;
  for (std::size_t next = 0; next < origins.size(); ++next) {
    auto origin = origins[next];
    if (!visited.insert(key(origin.axes)).second) continue;
    if (equal(interface, origin.axes)) return origin.positions;
    if (auto parent = positions(interface, origin.axes))
      return compose(parent, origin.positions);
    if (callIs(origin.axes, "gather"))
      origins.push_back({origin.axes->args[0],
                         compose(origin.axes->args[1], origin.positions)});
    if (auto incoming = group_wiring_.find(key(origin.axes));
        incoming != group_wiring_.end())
      for (auto index : incoming->second) {
        const auto& edge = wiring_[index];
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
  for (const auto& w : wiringFor(axes))
    result.mappings.push_back({w.group, axes, w.positions});
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
bool AxisBindings::bind(const S& interface, const S& group, const S& p) {
  auto identity = std::pair(key(interface), key(group));
  // Preserve the first established witness for a group. Later registrations
  // may express the same positions through a different composed expression.
  if (positions_.contains(identity)) {
    auto selector = std::pair(key(interface), key(p));
    if (groups_.contains(selector)) return false;
    groups_[selector] = group;
    lookup_cache_.clear();
    return true;
  }
  lookup_cache_.clear();
  positions_[identity] = p;
  groups_[{key(interface), key(p)}] = group;
  interface_wiring_[key(interface)].push_back(wiring_.size());
  group_wiring_[key(group)].push_back(wiring_.size());
  wiring_.push_back({interface, group, p});
  // Publish surviving components with their parent interface. This is
  // compositional construction, not a search through unrelated map records.
  if (!equal(interface, group)) {
    auto children = wiringFor(group);
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
  auto identity = key(interface) + ":" + key(call("concat", groups)) + ":" +
                  key(call("concat", selectors));
  if (!partition_keys_.insert(identity).second) return false;
  interface_partitions_[key(interface)].push_back(partitions_.size());
  partitions_.push_back({interface, groups, selectors});
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
  auto found = positions_.find({key(interface), key(group)});
  return found == positions_.end() ? S{} : found->second;
}
S AxisBindings::group(const S& interface, const S& positions) const {
  auto found = groups_.find({key(interface), key(positions)});
  return found == groups_.end() ? S{} : found->second;
}
S AxisBindings::cover(const S& interface) const {
  auto found = covers_.find(key(interface));
  return found == covers_.end() ? S{} : found->second;
}
std::vector<AxisBindings::Wiring> AxisBindings::wiringFor(
    const S& interface) const {
  std::vector<Wiring> result;
  auto found = interface_wiring_.find(key(interface));
  if (found != interface_wiring_.end())
    for (auto id : found->second) result.push_back(wiring_[id]);
  return result;
}
std::vector<AxisBindings::Partition> AxisBindings::partitionsFor(
    const S& interface) const {
  std::vector<Partition> result;
  auto found = interface_partitions_.find(key(interface));
  if (found != interface_partitions_.end())
    for (auto id : found->second) result.push_back(partitions_[id]);
  return result;
}
}  // namespace tepl::core::index_semantics
