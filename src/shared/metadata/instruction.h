#pragma once
#include <memory>
#include <string>
#include <vector>
namespace tepl::core::completed {
// Metadata instructions accepted by the runtime. Inference holes have no form
// here.
struct Symbol;
using SymbolPtr = std::shared_ptr<const Symbol>;
struct Symbol {
  enum class Sort { Integer, Boolean, Positions, Axes, Extents, DType };
  enum class Kind {
    Integer,
    Boolean,
    Shape,
    DType,
    Field,
    AxisGroup,
    List,
    Call
  };
  Kind kind;
  std::string name;
  std::size_t owner = 0, index = 0;
  std::vector<SymbolPtr> args;
  Sort sort = Sort::Positions;
};
}  // namespace tepl::core::completed
