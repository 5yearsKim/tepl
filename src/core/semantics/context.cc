#include "src/core/semantics/context.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "src/shared/builtins/catalog.h"
#include "src/shared/builtins/intrinsics/catalog.h"

namespace tepl::core::semantics {
std::vector<std::pair<S, S>> Context::extentBounds() const {
  std::vector<std::pair<S, S>> out;
  for (const auto& fact : facts_.raw)
    if (callIs(fact, "extent_le"))
      out.emplace_back(fact->args[0], fact->args[1]);
  return out;
}
void Context::registerCoordinates(std::size_t node, std::size_t group,
                                  const S& axes) {
  source_.groups[{node, group}] = axes;
}
void Context::recordInputs(std::size_t node, std::vector<S> shapes) {
  source_.input_shapes[node] = std::move(shapes);
}
void Context::recordTensor(std::size_t node, const S& axes,
                           std::vector<S> groups) {
  source_.tensor_axes[node] = axes;
  source_.tensor_groups[node] = std::move(groups);
}
void Context::recordShape(const S& axes, const S& shape) {
  // Called only for legal source nodes or after a target's extent check.
  auto normalized = canonical(shape);
  for (const auto& value : {shape, normalized}) {
    auto& witnesses = shape_witnesses_[key(value)];
    if (std::none_of(witnesses.begin(), witnesses.end(),
                     [&](const S& old) { return equal(old, axes); }))
      witnesses.push_back(axes);
  }
  queries_.evidence.clear();
  queries_.entailment.clear();
}
void Context::assignAttribute(std::size_t descriptor, std::size_t field,
                              const S& value) {
  auto [found, added] =
      attributes_.solutions.emplace(std::pair(descriptor, field), value);
  if (!added) {
    if (!equal(found->second, value))
      throw ProofFailure("conflicting attribute assignments",
                         FailureKind::InvalidCandidate);
    return;
  }
  attributes_.resolved.clear();
}
void Context::sourceScope(std::size_t id) {
  tick();
  const auto& n = plan.nodes.at(id);
  source_.nodes.insert(id);
  if (n.capture) source_.captures.insert(n.capture->value);
  if (n.descriptor && n.definition && !n.automatic) {
    source_.fields[n.descriptor->value] = n.definition->fields.size();
    source_.field_nodes[n.descriptor->value] = id;
  }
  for (auto child : n.inputs) sourceScope(child);
  if (id == plan.lhs)
    for (const auto& assumption : plan.assumptions) {
      validateSource(assumption);
      learn(assumption);
    }
}
void Context::validateSource(const S& s) {
  tick();
  if (s && validated_scope_.contains(s.get())) return;
  if (!s || s->kind == Symbol::Kind::Hole)
    throw ProofFailure(
        "constructor or guard contains an unresolved expression");
  if (s->kind == Symbol::Kind::Field &&
      (!source_.fields.contains(s->owner) ||
       s->index >= source_.fields.at(s->owner)))
    throw ProofFailure(
        "constructor or guard references a field outside the LHS");
  if ((s->kind == Symbol::Kind::Shape || s->kind == Symbol::Kind::DType) &&
      !source_.captures.contains(s->owner))
    throw ProofFailure(
        "constructor or guard references a tensor outside the LHS");
  if (s->kind == Symbol::Kind::AxisGroup && !source_.nodes.contains(s->owner))
    throw ProofFailure("coordinate expression is outside the LHS frame");
  for (const auto& a : s->args) validateSource(a);
  validated_scope_[s.get()] = s;
}
void Context::wellFormed(const S& s) {
  tick();
  if (s && validated_expressions_.contains(s.get())) return;
  validateSource(s);
  for (const auto& a : s->args) wellFormed(a);
  if (s->kind == Symbol::Kind::AxisGroup) {
    auto found = source_.groups.find({s->owner, s->index});
    if (found == source_.groups.end() || s->args.size() != 1)
      throw ProofFailure("unregistered source coordinate family");
    auto domain = found->second->kind == Symbol::Kind::AxisGroup
                      ? found->second->args[0]
                      : make(Symbol::Kind::List);
    require(call("==", {s->args[0], domain}));
  }
  if (s->kind != Symbol::Kind::Call) {
    validated_expressions_[s.get()] = s;
    return;
  }
  bool valid_arity = false;
  if (const auto* entry = intrinsics::resolve(s->name))
    valid_arity = entry->acceptsArity(s->args.size());
  else if (const auto* entry = builtins::resolveBuiltin(s->name))
    valid_arity = entry->acceptsArity(s->args.size());
  else if (s->name.starts_with("dtype_") && resolveDType(s->name.substr(6)))
    valid_arity = s->args.empty();
  else if (resolveBinaryOperator(s->name))
    valid_arity = s->args.size() == 2;
  else if (resolveUnaryOperator(s->name))
    valid_arity = s->args.size() == 1;
  else
    throw ProofFailure("undefined metadata operator: " + s->name);
  if (!valid_arity)
    throw ProofFailure("invalid metadata operator arity: " + s->name);
  if (s->name == "extent_add" || s->name == "extent_sub" ||
      s->name == "extent_le" || s->name == "extent_mul" ||
      s->name == "extent_floor_div")
    require(call("==", {call("len", {s->args[0]}), call("len", {s->args[1]})}));
  if (s->name == "extent_floor_div")
    require(call("is_positive_extents", {s->args[1]}));
  if (s->name == "empty_axes" && !s->args.empty())
    throw ProofFailure("invalid empty-axis constructor");
  if (s->name == "membership_positions" || s->name == "matched_positions")
    for (const auto& argument : s->args)
      if (argument->sort != Symbol::Sort::Positions)
        throw ProofFailure(
            "paired position selection requires position sequences");
  if (s->name == "place_axes") {
    require(call("is_valid_axis_list", {s->args[1], call("len", {s})}));
    require(call("==", {call("len", {s->args[1]}), call("len", {s->args[0]})}));
    require(call("distinct_axes", {call("concat", {s->args[0], s->args[2]})}));
  }
  if (s->name == "inverse")
    require(
        call("is_valid_axis_list", {s->args[0], call("len", {s->args[0]})}));
  if (s->name == "gather")
    require(
        call("is_valid_axis_list", {s->args[1], call("len", {s->args[0]})}));
  if (s->name == "positions") {
    require(call("distinct_axes", {s->args[0]}));
    require(call("distinct_axes", {s->args[1]}));
    require(call("subset", {s->args[1], s->args[0]}));
  }
  if (s->name == "replace_axes") {
    require(call("distinct_axes", {s->args[1]}));
    require(call("distinct_axes", {s->args[2]}));
    require(call("==", {call("len", {s->args[1]}), call("len", {s->args[2]})}));
  }
  validated_expressions_[s.get()] = s;
}

Context::~Context() {
  if (!std::getenv("TEPL_INDEX_PROFILE")) return;
  std::cerr << "index-context ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - started_)
                   .count()
            << " canonical=" << counters.canonical
            << " hits=" << counters.cache_hits
            << " queries=" << counters.queries
            << " fact-comparisons=" << counters.fact_comparisons
            << " invalidations=" << counters.invalidations
            << " positions=" << counters.positions
            << " facts=" << facts_.raw.size()
            << " resolutions=" << counters.resolutions
            << " bindings=" << bindings_.mappingIds().size() << '\n';
}
void Context::tick() {
  if (budget_ == 0)
    throw ProofFailure("symbolic proof work limit exceeded",
                       FailureKind::BudgetExceeded);
  --budget_;
}
void Context::note(const std::string& law) {
  if (std::find(steps_.begin(), steps_.end(), "law: " + law) == steps_.end())
    steps_.push_back("law: " + law);
}
void Context::require(const S& f, GuardKind kind) {
  if (containsHole(f))
    throw ProofFailure("legality obligation contains an unresolved auto field");
  if (entails(f)) return;
  auto c = canonical(f);
  guards_.push_back(c);
  guard_origins_.push_back({c, kind});
  // Failed obligations are diagnostic data. They never strengthen premises.
  throw ProofFailure("declared conditions and LHS legality do not establish " +
                     std::string(guardKindName(kind)) + ": " + format(*c));
}
S Context::resolve(const S& s) {
  ++counters.resolutions;
  tick();
  auto found = attributes_.resolved.find(key(s));
  if (found != attributes_.resolved.end()) return found->second;
  if (s->kind == Symbol::Kind::Hole) {
    auto it = attributes_.solutions.find({s->owner, s->index});
    return it == attributes_.solutions.end() ? s : resolve(it->second);
  }
  if (s->args.empty()) return s;
  std::vector<S> args;
  for (const auto& a : s->args) args.push_back(resolve(a));
  auto result = s->kind == Symbol::Kind::Call ? call(s->name, std::move(args))
                                              : make(s->kind, s->name, s->owner,
                                                     s->index, std::move(args));
  attributes_.resolved[key(s)] = result;
  return result;
}
S Context::field(const Instance& n, std::size_t f) {
  if (!n.descriptor) throw ProofFailure("attribute has no rule descriptor");
  if (auto found = plan.attributes.find(n.descriptor->value);
      found != plan.attributes.end()) {
    const auto& value = found->second.at(f);
    validateSource(value);
    return resolve(value);
  }
  return resolve(attributeSymbol(plan, n, f));
}

}  // namespace tepl::core::semantics
