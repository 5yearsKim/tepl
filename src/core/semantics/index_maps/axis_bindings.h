#pragma once

#include <set>

#include "src/core/semantics/symbolic/expr.h"

namespace tepl::core::semantics {

// Positions retain group order: gather(interface, positions) == group.
struct AxisSelection {
  S group, positions;
};
struct AxisPartition {
  S parent, selected, remaining;
  S selected_positions, remaining_positions;
};
struct TensorAxisInterface {
  S axes;
  std::vector<AxisSelection> mappings;
  S lookup(const S& group) const;
};

// One owner for established coordinate records and their lookup indexes.
// Registration changes all indexes together; Context owns the corresponding
// evidence and cache invalidation.
class AxisBindings {
 public:
  struct MappingId {
    std::size_t value;
    bool operator==(const MappingId&) const = default;
  };
  struct Mapping {
    S interface, group, positions;
    // Partitions may reserve an ID before bind publishes its lookup records.
    bool registered = false;
  };
  struct PartitionMember {
    MappingId mapping;
    // Retain the partition's witness only when it differs from the preferred
    // mapping. Refinement depends on the original selector expression.
    S selector;
  };
  struct Partition {
    S interface;
    std::vector<PartitionMember> members;
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
  bool isView(const S& view) const {
    return group_mappings_.contains(key(view));
  }
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

  // IDs stay stable across insertion. Copy a record before calling code that
  // can register more mappings; references into the vector may be invalidated.
  const Mapping& mapping(MappingId id) const { return mappings_[id.value]; }
  S selector(const PartitionMember& member) const {
    return member.selector ? member.selector
                           : mapping(member.mapping).positions;
  }
  const std::vector<MappingId>& mappingIds() const { return mapping_order_; }
  const std::vector<Partition>& partitions() const { return partitions_; }
  const std::vector<Overlap>& overlaps() const { return overlaps_; }
  const Overlap* overlapFor(const S& view) const;
  // Snapshots allow registration during transport without invalidating
  // iteration. Only records for the requested interface are copied.
  std::vector<Mapping> mappingsFor(const S& interface) const;
  std::vector<Partition> partitionsFor(const S& interface) const;

 private:
  mutable symbolic::KeyArena keys_;
  Key key(const S& s) const { return keys_.intern(semantics::key(s)); }
  S lookup(const S&, const S&, std::set<std::pair<Key, Key>>&) const;
  MappingId reserveMapping(const S& interface, const S& group,
                           const S& positions);
  std::map<Key, S> ordered_views_;
  std::vector<AxisPartition> splits_;
  std::map<std::pair<Key, Key>, std::size_t> split_index_;
  mutable std::map<std::pair<Key, Key>, S> lookup_cache_;
  std::vector<Mapping> mappings_;
  std::vector<MappingId> mapping_order_;
  std::vector<Partition> partitions_;
  std::vector<Overlap> overlaps_;
  std::map<Key, std::size_t> overlap_views_;
  std::map<std::pair<Key, Key>, MappingId> by_group_, by_selector_;
  std::map<Key, std::vector<MappingId>> interface_mappings_, group_mappings_;
  std::map<Key, std::vector<std::size_t>> interface_partitions_;
  std::map<Key, S> covers_;
  std::set<Key> partition_keys_;
  std::set<std::pair<Key, Key>> transported_frames_;
};

}  // namespace tepl::core::semantics
