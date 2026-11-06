#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace tepl::inference {

// Completed metadata instructions. There are no definition ASTs, symbolic
// holes, or general proof queries in this representation.
struct DomainRequirements {
  static constexpr std::uint64_t minimum_extent = 2;
  // Rank-zero inputs and empty axis selections are included in the domain.
};
enum class Mode { Synthesize, Validate };
enum class Kind { Input, Permutation, Contraction, Summation, Addition, Reverse, Broadcast, Multiplication, Negation, UnitRemoval };
struct Instruction {
  Kind kind = Kind::Input;
  std::size_t node = 0;
  std::optional<std::size_t> capture, descriptor, operation;
  std::vector<std::size_t> inputs;    // Earlier instructions in this block.
  std::vector<std::size_t> captures;  // Sorted subtree capture set.
  // Canonical role order: permutation or reduction dimensions; or left/right
  // batch, left/right contract.
  std::vector<std::size_t> fields;
  bool operator==(const Instruction&) const = default;
};
struct Component {
  std::vector<Instruction> source, replacement;
  bool operator==(const Component&) const = default;
};
struct Program {
  Mode mode = Mode::Synthesize;
  // Postorder schedules; each root is the final instruction. Reversing the
  // replacement schedule gives a valid parent-before-child traversal.
  std::vector<Instruction> source, replacement;
  // Empty for a single component. Source expansion preserves captured
  // descriptors; replacement schedules contain original, independent holes.
  std::vector<Component> components;
  bool operator==(const Program&) const = default;
};

}  // namespace tepl::inference
