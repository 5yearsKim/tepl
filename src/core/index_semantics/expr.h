#pragma once
#include <set>

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

// One source-scoped metadata/coordinate theory, shared by lowering and proof.
// Axis identities are never unified by comparing their extents.
struct Context {
  const Plan& plan;
  std::size_t budget;
  std::map<std::pair<std::size_t, std::size_t>, S> solutions;
  std::map<std::string, S> representatives;
  std::map<std::string, S> canonical_cache;
  std::map<std::string, bool> known_cache;
  std::set<std::string> active_queries;
  std::map<const Symbol*, S> validated_scope, validated_expressions;
  std::vector<S> facts, guards;
  std::vector<std::string> steps;
  std::map<std::size_t, std::size_t> source_fields;
  std::set<std::size_t> source_captures, source_nodes;
  std::map<std::pair<std::size_t, std::size_t>, S> coordinate_groups;
  std::map<std::size_t, S> tensor_axes;
  std::map<std::size_t, std::vector<S>> tensor_groups;
  std::map<std::size_t, std::vector<S>> input_shapes;
  // Established coordinate wiring: positions(interface, group) = selector.
  struct Wiring {
    S interface, group, positions;
  };
  std::vector<Wiring> wiring;
  std::map<std::string, S> covers;
  bool allow_guards = true;
  void tick();
  void sourceScope(std::size_t);
  void validateSource(const S&);
  void wellFormed(const S&);
  void note(const std::string&);
  S canonical(const S&);
  void learn(const S&);
  bool known(const S&);
  bool prove(const S&);
  void require(const S&);
  S resolve(const S&);
  S field(const Instance&, std::size_t);
  S positions(const S&, const S&);
  S select(const S&, const S&);
  S replace(const S&, const S&, const S&);
  S ordered(const S& candidates, const S& fallback);
  void bind(const S& interface, const S& group, const S& positions);
  S support(const S&);
  S intersect(const S&, const S&);
  S difference(const S&, const S&);
  S filter(const S&, const S&);
  S extents(const S&);
};
}  // namespace tepl::core::index_semantics
