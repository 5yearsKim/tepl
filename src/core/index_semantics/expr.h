#pragma once
#include "src/core/index_semantics/ir.h"

namespace tepl::core::index_semantics {
using S = SymbolPtr;
S make(Symbol::Kind, std::string = {}, std::size_t = 0, std::size_t = 0,
       std::vector<S> = {});
S integer(std::size_t);
S boolean(bool);
S call(std::string, std::vector<S>);
std::string key(const S&);
bool equal(const S&, const S&);
bool callIs(const S&, const std::string&);
bool containsHole(const S&);
bool truth(const S&);

}  // namespace tepl::core::index_semantics
