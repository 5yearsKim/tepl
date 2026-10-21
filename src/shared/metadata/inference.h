#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace tepl::inference {

// Completed metadata instructions. There are no definition ASTs, symbolic
// holes, or general proof queries in this representation.
enum class Mode { Synthesize, Validate };
enum class Kind { Input, Permutation, Contraction };
struct Instruction {
  Kind kind = Kind::Input;
  std::size_t node = 0;
  std::optional<std::size_t> capture, descriptor, operation;
  std::vector<std::size_t> inputs;    // Earlier instructions in this block.
  std::vector<std::size_t> captures;  // Sorted subtree capture set.
  // Canonical role order: permutation; or left/right batch, left/right
  // contract.
  std::vector<std::size_t> fields;
  bool operator==(const Instruction&) const = default;
};
struct Program {
  Mode mode = Mode::Synthesize;
  // Postorder schedules; each root is the final instruction. Reversing the
  // replacement schedule gives a valid parent-before-child traversal.
  std::vector<Instruction> source, replacement;
  bool operator==(const Program&) const = default;
};

}  // namespace tepl::inference
