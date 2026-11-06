#pragma once
#include <eggc/rewrite.hpp>

#include "analysis.h"
namespace tepl_runtime::matching {
struct Pattern {
  enum class Kind { Capture, Bind, Literal, Operation };
  Kind kind;
  std::size_t id = 0;
  std::string text;
  std::optional<DType> dtype;
  std::optional<std::size_t> descriptor;
  std::vector<Pattern> operands;
  static Pattern capture(std::size_t id) {
    return {Kind::Capture, id, {}, {}, {}, {}};
  }
  static Pattern bind(std::size_t id, Pattern inner) {
    return {Kind::Bind, id, {}, {}, {}, {std::move(inner)}};
  }
  static Pattern literal(std::string text, std::optional<DType> dtype) {
    return {Kind::Literal, 0, std::move(text), dtype, {}, {}};
  }
  static Pattern operation(std::size_t id,
                           std::optional<std::size_t> descriptor,
                           std::vector<Pattern> operands) {
    return {Kind::Operation, id, {}, {}, descriptor, std::move(operands)};
  }
};
struct Bindings {
  std::map<std::size_t, eggc::Id> captures;
  std::map<std::size_t, Value> descriptors, dimensions, dtypes, locals;
};
template <class T>
bool bind(std::map<std::size_t, T>& values, std::size_t key, T value) {
  auto it = values.find(key);
  if (it != values.end()) return it->second == value;
  values.emplace(key, std::move(value));
  return true;
}
template <Program P>
std::vector<Bindings> visit(const Pattern& pattern, eggc::Id id,
                            const BaseGraph<P>& graph, Bindings bindings,
                            Budget& budget, const eggc::StopCheck& stop = {}) {
  budget.tick();
  if (stop && stop()) throw EvalError(ErrorKind::Limit, "search cancelled");
  id = graph.find(id);
  if (pattern.kind == Pattern::Kind::Capture)
    return bind(bindings.captures, pattern.id, id)
               ? std::vector<Bindings>{bindings}
               : std::vector<Bindings>{};
  if (pattern.kind == Pattern::Kind::Bind)
    return bind(bindings.captures, pattern.id, id)
               ? visit<P>(pattern.operands[0], id, graph, std::move(bindings),
                          budget, stop)
               : std::vector<Bindings>{};
  std::vector<Bindings> out;
  for (const auto& node : graph.nodes(id)) {
    budget.tick();
    if (pattern.kind == Pattern::Kind::Literal) {
      if (node.kind() != NodeKind::Literal ||
          (pattern.dtype && pattern.dtype != node.dtype()) ||
          (pattern.text.find('.') != pattern.text.npos) !=
              (node.text().find('.') != node.text().npos))
        continue;
      bool decimal = pattern.text.find('.') != pattern.text.npos;
      if ((decimal ? floating(pattern.text) == floating(node.text())
                   : integer(pattern.text) == integer(node.text())))
        return {bindings};
      continue;
    }
    if (node.kind() != NodeKind::Operation ||
        P::operation_id(node.op()) != pattern.id ||
        pattern.operands.size() != node.children().size())
      continue;
    auto candidate = bindings;
    if (pattern.descriptor && !bind(candidate.descriptors, *pattern.descriptor,
                                    node.attribute_value()))
      continue;
    std::vector<Bindings> candidates{candidate};
    for (std::size_t i = 0; i < pattern.operands.size(); ++i) {
      std::vector<Bindings> next;
      for (auto& c : candidates) {
        auto xs = visit<P>(pattern.operands[i], node.children()[i], graph,
                           std::move(c), budget, stop);
        budget.size(next.size() + xs.size());
        next.insert(next.end(), std::make_move_iterator(xs.begin()),
                    std::make_move_iterator(xs.end()));
      }
      candidates = std::move(next);
      if (candidates.empty()) break;
    }
    budget.size(out.size() + candidates.size());
    out.insert(out.end(), std::make_move_iterator(candidates.begin()),
               std::make_move_iterator(candidates.end()));
  }
  return out;
}
template <Program P>
std::vector<Bindings> matches(const Pattern& pattern, eggc::Id root,
                              const BaseGraph<P>& graph, Budget& budget,
                              const eggc::StopCheck& stop = {}) {
  return visit<P>(pattern, root, graph, {}, budget, stop);
}
struct ShapeElement {
  bool sequence;
  std::optional<std::size_t> symbol;
};
struct DTypeConstraint {
  std::optional<DType> exact;
  std::optional<std::size_t> variable;
};
struct Constraint {
  std::size_t capture;
  std::vector<ShapeElement> shape;
  std::optional<DTypeConstraint> dtype;
};
template <Program P>
bool constraints(const std::vector<Constraint>& cs, Bindings& b,
                 const BaseGraph<P>& graph) {
  for (const auto& c : cs) {
    const auto& facts = graph.analysis_data(b.captures.at(c.capture));
    const auto& s = facts.shape.get("shape");
    std::size_t fixed = 0;
    bool sequence = false;
    for (const auto& e : c.shape) {
      fixed += !e.sequence;
      sequence = sequence || e.sequence;
    }
    if (sequence ? s.size() < fixed : s.size() != fixed) return false;
    std::size_t at = 0;
    for (const auto& e : c.shape) {
      Value v;
      if (e.sequence) {
        auto n = s.size() - fixed;
        v = tepl_runtime::shape(
            std::vector<std::uint64_t>(s.begin() + at, s.begin() + at + n));
        at += n;
      } else
        v = Value(Integer(s[at++]));
      if (e.symbol && !bind(b.dimensions, *e.symbol, v)) return false;
    }
    if (c.dtype) {
      auto d = facts.dtype.get("dtype");
      if (c.dtype->exact && *c.dtype->exact != d) return false;
      if (c.dtype->variable && !bind(b.dtypes, *c.dtype->variable, to_value(d)))
        return false;
    }
  }
  return true;
}
}  // namespace tepl_runtime::matching
