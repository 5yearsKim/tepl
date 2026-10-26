#pragma once
#include "src/core/semantics/ir.h"

namespace tepl::core::semantics {
using S = SymbolPtr;
using Key = symbolic::Key;
S make(Symbol::Kind, std::string = {}, std::size_t = 0, std::size_t = 0,
       std::vector<S> = {}, std::optional<Symbol::Sort> = {});
S attributeSymbol(const Plan&, const Instance&, std::size_t field);
std::optional<Symbol::Sort> scalarMetadataSort(core::Type);
S integer(std::size_t);
S boolean(bool);
S call(std::string, std::vector<S>);
struct SequenceSelection {
  S base, positions;
};
// A concatenation of selections from one base, in the original part order.
std::optional<SequenceSelection> concatSelection(const std::vector<S>&);
// Match entries from left in left's order. The two position lists are a pair:
// gather(left, left_positions) == gather(right, right_positions).
struct PositionMatch {
  S left_positions;
  S right_positions;
};
PositionMatch matchPositions(const S& left, const S& right);
Key key(const S&);
bool equal(const S&, const S&);
// Deterministic term order, independent of process-wide interning history.
bool symbolLess(const S&, const S&);
bool callIs(const S&, const std::string&);
bool containsHole(const S&);
bool truth(const S&);

}  // namespace tepl::core::semantics
