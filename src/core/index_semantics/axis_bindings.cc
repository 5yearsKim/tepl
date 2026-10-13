#include "src/core/index_semantics/axis_bindings.h"

namespace tepl::core::index_semantics {
bool AxisBindings::bind(const S& interface, const S& group, const S& p) {
  auto identity = std::pair(key(interface), key(group));
  // Preserve the first established witness for a group. Later registrations
  // may express the same positions through a different composed expression.
  if (positions_.contains(identity)) return false;
  positions_[identity] = p;
  groups_[{key(interface), key(p)}] = group;
  interface_wiring_[key(interface)].push_back(wiring_.size());
  wiring_.push_back({interface, group, p});
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
