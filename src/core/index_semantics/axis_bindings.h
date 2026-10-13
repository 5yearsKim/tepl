#pragma once

#include <set>

#include "src/core/index_semantics/expr.h"

namespace tepl::core::index_semantics {

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

  bool bind(const S& interface, const S& group, const S& positions);
  bool partition(const S& interface, const std::vector<S>& groups,
                 const std::vector<S>& selectors);
  bool cover(const S& interface, const S& groups);
  bool beginTransport(const S& from, const S& to);
  S positions(const S& interface, const S& group) const;
  S group(const S& interface, const S& positions) const;
  S cover(const S& interface) const;

  const std::vector<Wiring>& wiring() const { return wiring_; }
  const std::vector<Partition>& partitions() const { return partitions_; }
  // Snapshots allow registration during transport without invalidating
  // iteration. Only records for the requested interface are copied.
  std::vector<Wiring> wiringFor(const S& interface) const;
  std::vector<Partition> partitionsFor(const S& interface) const;

 private:
  std::vector<Wiring> wiring_;
  std::vector<Partition> partitions_;
  std::map<std::pair<std::string, std::string>, S> positions_, groups_;
  std::map<std::string, std::vector<std::size_t>> interface_wiring_,
      interface_partitions_;
  std::map<std::string, S> covers_;
  std::set<std::string> partition_keys_;
  std::set<std::pair<std::string, std::string>> transported_frames_;
};

}  // namespace tepl::core::index_semantics
