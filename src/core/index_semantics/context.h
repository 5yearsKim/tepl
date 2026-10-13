#pragma once

#include <chrono>
#include <tuple>

#include "src/core/index_semantics/axis_bindings.h"
#include "src/core/index_semantics/proof_policy.h"

namespace tepl::core::index_semantics {

enum class QueryMode { Evidence, Entailment };

struct SourceCoordinates {
  std::map<std::size_t, std::size_t> fields;
  std::set<std::size_t> captures, nodes;
  std::map<std::pair<std::size_t, std::size_t>, S> groups;
  std::map<std::size_t, S> tensor_axes;
  std::map<std::size_t, std::vector<S>> tensor_groups, input_shapes;
};

// Rule-scoped coordinator. Lowering and synthesis use mutation APIs rather
// than modifying mapping indexes, premises, or cached attribute solutions.
// Axis identities are never unified by comparing their extents.
class Context {
 public:
  Context(const Plan& plan, std::size_t work) : plan(plan), budget_(work) {}
  ~Context();
  const Plan& plan;
  const SourceCoordinates& source() const { return source_; }
  const std::vector<S>& guards() const { return guards_; }
  const std::vector<std::string>& steps() const { return steps_; }
  void allowGuards(bool allowed) { allow_guards_ = allowed; }
  void registerCoordinates(std::size_t node, std::size_t group, const S&);
  void recordInputs(std::size_t node, std::vector<S> shapes);
  void recordTensor(std::size_t node, const S& axes, std::vector<S> groups);
  void assignAttribute(std::size_t descriptor, std::size_t field, const S&);

  void tick();
  void sourceScope(std::size_t);
  void validateSource(const S&);
  void wellFormed(const S&);
  void note(const std::string&);
  S canonical(const S&);
  void learn(const S&);
  // Evidence lookup and bounded proof search have explicit entry points.
  bool evidence(const S&);
  // False means unestablished, including a bounded attempt that ran out of
  // local work. It does not mean the proposition is mathematically false.
  bool entails(const S&);
  void require(const S&);
  S resolve(const S&);
  S field(const Instance&, std::size_t);
  S positions(const S&, const S&, QueryMode = QueryMode::Entailment);
  S select(const S&, const S&, QueryMode = QueryMode::Entailment);
  void selectionFacts(const S&, const S&, const S&);
  void partition(const S&, const std::vector<S>&, const std::vector<S>&);
  void bind(const S& interface, const S& group, const S& positions);
  void cover(const S& interface, const S& groups);
  void transportFrame(const S& from, const S& to);
  S transportInterface(const S& from, const S& to, const S& read_axes);
  S replace(const S&, const S&, const S&);
  S support(const S&, QueryMode = QueryMode::Entailment);
  S intersect(const S&, const S&, QueryMode = QueryMode::Entailment);
  S difference(const S&, const S&, QueryMode = QueryMode::Entailment);
  S filter(const S&, const S&, QueryMode = QueryMode::Entailment);
  S extents(const S&);

 private:
  struct ActiveKey {
    std::set<std::string>& keys;
    std::string id;
    ~ActiveKey() { keys.erase(id); }
  };
  static bool independent(const S&, const S&);
  struct FactState {
    std::vector<S> raw;
    std::set<std::string> keys, normalized, permutations;
    std::map<std::string, S> representatives, selectors;
    std::map<std::string, std::vector<S>> valid_selectors;
  } facts_;
  struct QueryState {
    std::map<std::string, S> canonical;
    std::map<std::string, bool> entailment, evidence;
    std::map<std::pair<QueryMode, std::string>, S> support;
    std::map<std::tuple<QueryMode, std::string, std::string>, S> positions;
    std::set<std::string> active_queries, active_canonical, active_support,
        active_positions, active_premises;
    std::size_t start_budget = 0;
  } queries_;
  struct AttributeState {
    std::map<std::pair<std::size_t, std::size_t>, S> solutions;
    std::map<std::string, S> resolved;
  } attributes_;
  struct Counters {
    std::size_t canonical = 0, cache_hits = 0, queries = 0,
                fact_comparisons = 0, invalidations = 0, positions = 0,
                resolutions = 0;
  } counters;
  AxisBindings bindings_;
  SourceCoordinates source_;
  std::map<const Symbol*, S> validated_scope_, validated_expressions_;
  std::vector<S> guards_;
  std::vector<std::string> steps_;
  std::size_t budget_;
  bool allow_guards_ = true;
  std::chrono::steady_clock::time_point started_ =
      std::chrono::steady_clock::now();

  void invalidate();
  bool established(const S&);
  bool deriveEvidence(const S&);
  bool structuralEvidence(const S&, QueryMode);
  bool query(const S&, QueryMode);
  bool prove(const S&);
  bool partitionPiece(const S&) const;
  bool elementary(const S&) const;
  bool partitionDisjoint(const S&, const S&);
  S lookupPositions(const S&, const S&, QueryMode);
  S selectKnown(const S&, const S&, QueryMode);
  S computeSupport(const S&, QueryMode);
};

}  // namespace tepl::core::index_semantics
