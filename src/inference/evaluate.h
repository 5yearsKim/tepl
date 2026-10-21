#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>

#include "src/shared/metadata/inference.h"

namespace tepl::inference {
using AxisId = std::size_t;
using AxisSequence = std::vector<AxisId>;
using AxisSet = std::set<AxisId>;
using List = std::vector<std::int64_t>;
using Attributes = std::map<std::size_t, std::vector<List>>;
struct Metadata {
  std::map<std::size_t, std::vector<std::uint64_t>> shapes;
  Attributes descriptors;
};
struct InputPort {
  std::size_t capture, position;
  std::uint64_t extent;
  bool operator==(const InputPort&) const = default;
};
struct SourceNetwork {
  std::vector<InputPort> ports;
  AxisSequence classes;
  std::map<std::size_t, AxisSequence> inputs;
  AxisSequence output;
  bool operator==(const SourceNetwork&) const = default;
};
struct Requirements {
  AxisSet live, batch, contract, left_free, right_free;
  AxisSequence required_order;
  bool operator==(const Requirements&) const = default;
};
struct Obstruction {
  std::size_t node = 0;
  AxisId first = 0, second = 0;
  bool operator==(const Obstruction&) const = default;
};
enum class Status {
  Feasible,
  Solved,
  Impossible,
  Unsupported,
  OutsideDomain,
  InvalidSource,
  InvalidReplacement,
  NotEquivalent,
  MissingMetadata,
  ResourceLimit
};
struct Result {
  Status status = Status::Unsupported;
  std::string reason;
  SourceNetwork source;
  std::map<std::size_t, Requirements> requirements;
  std::map<std::size_t, AxisSequence> layouts;
  Attributes attributes;
  std::optional<Obstruction> obstruction;
  std::size_t work = 0;
};
struct Limits {
  std::size_t work = std::numeric_limits<std::size_t>::max();
  std::size_t axes = std::numeric_limits<std::size_t>::max();
};

// Structural validation of completed instructions; throws invalid_argument.
void validate(const Program&);
// Analyze once and either construct attributes or replay the explicit RHS.
// Impossible is used only for exact synthesis infeasibility in the domain.
Result evaluate(const Program&, const Metadata&, const Limits& = {});
// Evaluate only the generated predicate. Synthesis returns Feasible with no
// attributes/layouts on success. Explicit validation necessarily replays RHS.
Result feasibility(const Program&, const Metadata&, const Limits& = {});
// Independently replay result attributes or rederive an ordering obstruction.
bool check(const Program&, const Metadata&, const Result&);
}  // namespace tepl::inference
