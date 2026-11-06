#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "src/core/semantics/symbolic/key.h"
#include "src/shared/source/source.h"

namespace tepl::core::symbolic {
// Immutable, rule-scoped metadata expressions. No tensor values or fixed ranks.
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
    Hole,
    AxisGroup,
    List,
    Call
  } kind;
  std::string name;
  std::size_t owner = 0, index = 0;
  std::vector<SymbolPtr> args;
  Sort sort = Sort::Positions;
  Symbol(Kind k, std::string n = {}, std::size_t o = 0, std::size_t i = 0,
         std::vector<SymbolPtr> a = {}, Sort t = Sort::Positions)
      : kind(k),
        name(std::move(n)),
        owner(o),
        index(i),
        args(std::move(a)),
        sort(t) {}
  // Copies are external, editable IR nodes: recompute their structural
  // identity.
  Symbol(const Symbol& s)
      : Symbol(s.kind, s.name, s.owner, s.index, s.args, s.sort) {}
  Symbol(Symbol&&) = default;
  const Key& structuralKey() const { return structural_key; }

 private:
  friend struct SymbolFactory;
  Key structural_key;
};

struct SymbolFactory {
  static void key(Symbol& s, Key value) { s.structural_key = std::move(value); }
};
}  // namespace tepl::core::symbolic
