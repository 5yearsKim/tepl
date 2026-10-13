#pragma once

#include <set>

#include "src/core/index_semantics/expr.h"

namespace tepl::core::index_semantics {

// Position witnesses are ordered: destination[positions] == source.
struct AxisMap {
  S source, destination, positions;
};
struct AxisPartition {
  S parent, selected, remaining;
  S selected_positions, remaining_positions;
};
struct TensorAxisInterface {
  S axes;
  std::vector<AxisMap> mappings;
  S lookup(const S& group) const;
};

// One owner for established coordinate records and their lookup indexes.
// Registration changes all indexes together; Context owns the corresponding
// evidence and cache invalidation.
class AxisBindings {
 public:
  struct Wiring {
    S interface, group, positions;
  };
  struct Partition {
    S interface;
    std::vector<S> groups, selectors;
  };
  struct Overlap {
    S left, right, view, slots, paired_slots;
  };

  // Pure geometry operations. They do not call Context or a theorem prover.
  // Registration is permitted only after selector validity is established.
  S select(const S& parent, const S& selector);
  AxisPartition split(const S& parent, const S& selector);
  S lookup(const S& interface, const S& group) const;
  bool orderedView(const S& view, const std::vector<S>& parts);
  S orderedView(const S& view) const;
  bool isView(const S& view) const { return group_wiring_.contains(key(view)); }
  std::vector<S> orderedParts(const S& view) const;
  TensorAxisInterface tensorInterface(const S& axes) const;
  const std::vector<AxisPartition>& splits() const { return splits_; }
  static S compose(const S& parent_positions, const S& child_positions);
  // Invert a selection only for a group already proved to survive in it.
  static S restrict(const S& group_positions, const S& selected_positions);

  bool bind(const S& interface, const S& group, const S& positions);
  bool partition(const S& interface, const std::vector<S>& groups,
                 const std::vector<S>& selectors);
  bool cover(const S& interface, const S& groups);
  bool beginTransport(const S& from, const S& to);
  bool overlap(const S& left, const S& right, const S& view, const S& slots,
               const S& paired_slots);
  S positions(const S& interface, const S& group) const;
  S group(const S& interface, const S& positions) const;
  S cover(const S& interface) const;

  const std::vector<Wiring>& wiring() const { return wiring_; }
  const std::vector<Partition>& partitions() const { return partitions_; }
  const std::vector<Overlap>& overlaps() const { return overlaps_; }
  const Overlap* overlapFor(const S& view) const;
  // Snapshots allow registration during transport without invalidating
  // iteration. Only records for the requested interface are copied.
  std::vector<Wiring> wiringFor(const S& interface) const;
  std::vector<Partition> partitionsFor(const S& interface) const;

 private:
  S lookup(const S&, const S&,
           std::set<std::pair<std::string, std::string>>&) const;
  std::map<std::string, S> ordered_views_;
  std::vector<AxisPartition> splits_;
  std::map<std::pair<std::string, std::string>, std::size_t> split_index_;
  mutable std::map<std::pair<std::string, std::string>, S> lookup_cache_;
  std::vector<Wiring> wiring_;
  std::vector<Partition> partitions_;
  std::vector<Overlap> overlaps_;
  std::map<std::string, std::size_t> overlap_views_;
  std::map<std::pair<std::string, std::string>, S> positions_, groups_;
  std::map<std::string, std::vector<std::size_t>> interface_wiring_,
      interface_partitions_, group_wiring_;
  std::map<std::string, S> covers_;
  std::set<std::string> partition_keys_;
  std::set<std::pair<std::string, std::string>> transported_frames_;
};

}  // namespace tepl::core::index_semantics
