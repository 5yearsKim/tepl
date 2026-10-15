#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "src/shared/source/source.h"

namespace tepl::core::axis {
// Operation definitions retain arbitrary-length groups and position
// expressions.
enum class Type { Integer, Boolean, Positions, Axes, Operand };
struct Meta;
using MetaPtr = std::shared_ptr<const Meta>;
struct Meta {
  enum class Kind {
    Integer,
    Boolean,
    Operand,
    Attribute,
    Group,
    List,
    Call
  } kind;
  Type type;
  std::string name;
  std::size_t index = 0;
  std::vector<MetaPtr> args;
  SourceOrigin origin;
};
struct Binding {
  std::size_t operand;
  MetaPtr positions;
};
struct Group {
  std::string name;
  std::vector<Binding> bindings;
  SourceOrigin origin;
  MetaPtr domain;  // Output-only groups have explicit extent sequences.
};
struct ShapeContract {
  std::vector<std::string> operands, fields;
  std::vector<Group> groups;
  std::vector<MetaPtr> assertions;
  MetaPtr output;
  // Numeric sequences may be constructed by extent arithmetic (domains or
  // per-axis coordinate coefficients), rather than only position selectors.
  std::set<std::size_t> extent_fields;
  std::set<std::size_t> implicit_operands;
};

}  // namespace tepl::core::axis
