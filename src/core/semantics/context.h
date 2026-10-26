#pragma once

#include <chrono>
#include <tuple>

#include "src/core/semantics/index_maps/axis_bindings.h"
#include "src/core/semantics/index_maps/axis_sets.h"
#include "src/core/semantics/proof/proof_policy.h"

namespace tepl::core::semantics {

enum class QueryMode { Evidence, Entailment };
using MetadataInteger = __int128;
using IntegerInterval = std::pair<MetadataInteger, MetadataInteger>;
std::optional<MetadataInteger> metadataInteger(const S&);

struct SourceCoordinates {
  std::map<std::size_t, std::size_t> fields;
  std::map<std::size_t, std::size_t> field_nodes;
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
  Key key(const S& s) const { return keys_.intern(semantics::key(s)); }
  bool equal(const S& a, const S& b) const { return key(a) == key(b); }

  const SourceCoordinates& source() const { return source_; }
  const std::vector<S>& guards() const { return guards_; }
  const std::vector<RuleProof::GuardOrigin>& guardOrigins() const {
    return guard_origins_;
  }
  const std::vector<std::string>& steps() const { return steps_; }
  // Snapshot established vector inequalities; consumers cannot add premises.
  std::vector<std::pair<S, S>> extentBounds() const;
  std::vector<S> arithmeticFacts() const;
  std::optional<IntegerInterval> integerInterval(const S&);
  IntegerInterval fieldIntegerInterval(const S&) const;
  S uniformElement(const S&);
  void registerCoordinates(std::size_t node, std::size_t group, const S&);
  void recordInputs(std::size_t node, std::vector<S> shapes);
  void recordTensor(std::size_t node, const S& axes, std::vector<S> groups);
  void recordShape(const S& axes, const S& shape);
  void assignAttribute(std::size_t descriptor, std::size_t field, const S&);

  void enableSetReasoning() { sets_ready_ = true; }
  void tick();
  void sourceScope(std::size_t);
  void validateSource(const S&);
  void wellFormed(const S&);
  void checkMetadata(const S&);
  void checkMetadataType(const S&, core::Type);
  void note(const std::string&);
  S canonical(const S&);
  void learn(const S&);
  // Evidence lookup and bounded proof search have explicit entry points.
  bool evidence(const S&);
  // False means unestablished, including a bounded attempt that ran out of
  // local work. It does not mean the proposition is mathematically false.
  bool entails(const S&);
  void require(const S&, GuardKind = GuardKind::Legality);
  S resolve(const S&);
  S field(const Instance&, std::size_t);
  S positions(const S&, const S&, QueryMode = QueryMode::Entailment);
  S select(const S&, const S&, QueryMode = QueryMode::Entailment);
  void selectionFacts(const S&, const S&, const S&);
  void partition(const S&, const std::vector<S>&, const std::vector<S>&,
                 bool refine = true);
  TensorAxisInterface tensorInterface(const S& axes) const;
  S mappedPositions(const S& interface, const S& group) const;
  void bind(const S& interface, const S& group, const S& positions);
  void cover(const S& interface, const S& groups);
  void transportFrame(const S& from, const S& to);
  S transportInterface(const S& from, const S& to, const S& read_axes);
  S replace(const S&, const S&, const S&);
  S support(const S&, QueryMode = QueryMode::Entailment);
  S intersect(const S&, const S&, QueryMode = QueryMode::Entailment);
  S difference(const S&, const S&, QueryMode = QueryMode::Entailment);
  S filter(const S&, const S&, QueryMode = QueryMode::Entailment);
  // Ordered views of an existing interface; these do not introduce identities.
  S membershipPositions(const S&, const S&);
  S extents(const S&);

 private:
  struct ActiveKey {
    std::set<Key>& keys;
    Key id;
    ~ActiveKey() { keys.erase(id); }
  };
  static bool independent(const S&, const S&);
  struct FactState {
    std::vector<S> raw;
    std::set<Key> keys, normalized, permutations;
    std::map<Key, S> representatives, selectors;
    std::map<Key, std::vector<S>> valid_selectors;
  } facts_;
  struct QueryState {
    std::map<Key, S> canonical, axis_sequences;
    std::map<Key, bool> entailment, evidence;
    std::map<Key, IntegerInterval> intervals;
    std::map<Key, S> uniform_elements;
    std::map<std::pair<QueryMode, Key>, S> support;
    std::map<std::tuple<QueryMode, Key, Key>, S> positions;
    std::set<Key> active_queries, active_canonical, active_support,
        active_positions, active_premises, active_set_operations,
        active_intervals, active_uniform_elements;
    std::size_t start_budget = 0;
    std::map<Key, std::pair<std::size_t, std::size_t>> term_order;
  } queries_;
  struct AttributeState {
    std::map<std::pair<std::size_t, std::size_t>, S> solutions;
    std::map<Key, S> resolved;
  } attributes_;
  struct Counters {
    std::size_t canonical = 0, cache_hits = 0, queries = 0,
                fact_comparisons = 0, invalidations = 0, positions = 0,
                resolutions = 0;
  } counters;
  mutable symbolic::KeyArena keys_;
  AxisBindings bindings_;
  std::map<Key, std::vector<S>> shape_witnesses_;
  std::vector<std::pair<S, S>> selected_extents_;
  std::unique_ptr<AxisSets> sets_;
  std::map<Key, AxisSets::Id> set_expressions_;
  std::vector<S> set_atoms_;
  bool active_sets_ = false, sets_ready_ = false;
  SourceCoordinates source_;
  std::map<const Symbol*, S> validated_scope_, validated_expressions_;
  std::vector<S> guards_;
  std::vector<RuleProof::GuardOrigin> guard_origins_;
  std::vector<std::string> steps_;
  std::size_t budget_;
  std::chrono::steady_clock::time_point started_ =
      std::chrono::steady_clock::now();

  void invalidate();
  void represent(S, S);
  S canonicalChildren(const S&);
  bool established(const S&);
  bool deriveEvidence(const S&);
  bool structuralEvidence(const S&, QueryMode);
  bool integerComparison(const S&, QueryMode);
  bool query(const S&, QueryMode);
  bool prove(const S&);
  bool partitionPiece(const S&) const;
  bool elementary(const S&) const;
  bool partitionDisjoint(const S&, const S&);
  S lookupPositions(const S&, const S&, QueryMode);
  S selectKnown(const S&, const S&, QueryMode);
  S computeSupport(const S&, QueryMode);
  S axisSequence(const S&);
  S sequenceEquality(const S&, const S&);
  bool orderedPositions(const S&) const;
  bool nonnegative(const S&);
  std::optional<std::pair<S, S>> positionBlock(const S&);
  S complementBlocks(const S&, const S&);
  bool setEntails(const S&);
  AxisSets::Id setExpression(const S&);
  S simplifySet(const S&);
};

}  // namespace tepl::core::semantics
