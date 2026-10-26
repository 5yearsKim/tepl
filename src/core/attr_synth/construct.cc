#include "src/core/attr_synth/construct.h"

#include <functional>

#include "src/core/attr_synth/index_inference.h"
#include "src/core/semantics/symbolic/tree.h"

namespace tepl::core::attr_synth {
using namespace semantics;
namespace {
struct Constructor {
  Context& c;
  Lowerer& lowerer;
  const Correspondence& match;
  const TensorMeaning& source;
  S reductions;
  const Terms& source_terms;
  bool unresolvedIndexAttributes(const Instance& n) {
    bool found = false;
    walkTree(n.definition->value, [&](const ValuePtr& value) {
      if (!value->access) return;
      walkTree(value->access, [&](const IndexValuePtr& index) {
        if (!index->parameter) return;
        walkTree(index->parameter, [&](const MetaPtr& metadata) {
          if (metadata->kind == Meta::Kind::Attribute)
            found |= containsHole(c.field(n, metadata->index));
        });
      });
    });
    return found;
  }
  void inferReadAttributes(const Instance& n,
                           const std::vector<TensorMeaning>& inputs,
                           const std::vector<S>& groups) {
    IndexAssignments inferred;
    auto merge = [&](IndexAssignments& into, const IndexAssignments& values) {
      for (const auto& [id, value] : values) {
        auto [old, added] = into.emplace(id, value);
        if (!added && !equalExtentArithmetic(c, old->second, value))
          throw ProofFailure(
              "ambiguous or conflicting calculated-read attribute "
              "constructors");
      }
    };
    auto read = [&](const E& pattern) {
      std::optional<IndexAssignments> solution;
      for (const auto& term : source_terms)
        for (const auto& actual : term.reads) {
          c.tick();
          if (actual->capture != pattern->capture) continue;
          auto candidate = matchIndexAttributes(
              c, pattern->access ? pattern->access : coordinates(pattern->axes),
              actual->access ? actual->access : coordinates(actual->axes));
          if (!candidate) continue;
          if (!solution)
            solution = *candidate;
          else
            merge(*solution, *candidate);
        }
      if (!solution)
        throw ProofFailure(
            "no scoped coordinate match for calculated-read attributes; "
            "composed or inverse inference may be required");
      merge(inferred, *solution);
    };
    walkTree(n.definition->value, [&](const ValuePtr& v) {
      if (v->kind != Value::Kind::Read) return;
      auto indices = lowerer.indexValue(*v->access, n, inputs, groups);
      // Compose through already constructed producers; proposals are checked
      // again when target lowering establishes the completed node's legality.
      auto expression = lowerer.evaluateAt(inputs[v->index].value,
                                           inputs[v->index].axes, indices);
      for (const auto& capture : tensorReads(expression)) read(capture);
    });
    if (inferred.empty())
      throw ProofFailure(
          "calculated reads provide no direct attribute constructor");
    for (const auto& [id, value] : inferred)
      c.assignAttribute(id.first, id.second, value);
    c.note("index-coefficient-matching");
  }
  void assign(const Meta& m, const Instance& n,
              const std::vector<TensorMeaning>& inputs,
              const std::vector<S>& groups, const S& value) {
    auto old = lowerer.metadata(m, n, inputs, groups);
    old = c.resolve(old);
    if (old->kind == Symbol::Kind::Hole) {
      c.assignAttribute(old->owner, old->index, value);
    } else if (containsHole(old))
      throw ProofFailure(
          "inverse metadata expression is outside constructive fragment");
    else
      c.require(call("==", {old, value}));
  }
  TensorMeaning build(std::size_t id) {
    c.tick();
    const auto& n = c.plan.nodes[id];
    if (n.capture || n.literal) return lowerer.lower(id);
    std::vector<TensorMeaning> inputs;
    for (auto child : n.inputs) inputs.push_back(build(child));
    if (!n.automatic) return lowerer.operation(id, std::move(inputs));
    std::set<std::size_t> reduced;
    walkTree(n.definition->value, [&](const ValuePtr& v) {
      if (v->kind == Value::Kind::Reduce) reduced.insert(v->index);
    });
    const auto trial_obligations = lowerer.obligations.size();
    const bool infer_indices = unresolvedIndexAttributes(n);
    std::vector<std::size_t> deferred_domains;
    std::set<std::pair<std::set<std::size_t>, bool>> signatures;
    std::vector<S> groups;
    for (std::size_t g = 0; g < n.definition->groups.size(); ++g) {
      const auto& group = n.definition->groups[g];
      std::set<std::size_t> incidence;
      for (const auto& b : group.bindings) incidence.insert(b.operand);
      if (!signatures.emplace(incidence, reduced.contains(g)).second)
        throw ProofFailure(
            "multiple binding groups share an unresolved incidence class");
      if (group.domain) {
        if (!incidence.empty())
          throw ProofFailure("explicit domain groups cannot bind operands");
        std::vector<S> operand_axes;
        for (const auto& input : inputs) operand_axes.push_back(input.axes);
        auto wanted = c.difference(match.boundaries.at(id),
                                   call("union_axes", std::move(operand_axes)));
        // A single non-reduced domain covers the output interface even when
        // an identity coordinate map reuses axes from its input. Subtracting
        // input axes would erase that frame before calculated-read matching.
        if (infer_indices && !reduced.contains(g) &&
            n.definition->groups.size() == 1)
          wanted = match.interfaces.at(id);
        auto order = c.filter(match.interfaces.at(id), wanted);
        auto domain = lowerer.metadata(*group.domain, n, inputs, groups);
        if (infer_indices && containsHole(c.resolve(domain)) &&
            c.resolve(domain)->kind != Symbol::Kind::Hole)
          deferred_domains.push_back(g);
        else
          assign(*group.domain, n, inputs, groups, c.extents(order));
        groups.push_back(order);
        continue;
      }
      S wanted;
      for (auto o : incidence)
        wanted = wanted ? c.intersect(wanted, inputs[o].interface.axes)
                        : c.support(inputs[o].interface.axes);
      for (std::size_t o = 0; o < inputs.size(); ++o)
        if (!incidence.contains(o))
          wanted = c.difference(wanted, inputs[o].interface.axes);
      wanted = reduced.contains(g)
                   ? c.difference(wanted, match.boundaries.at(id))
                   : c.intersect(wanted, match.boundaries.at(id));
      S order;
      // A fixed binding determines group order. Unknown paired fields follow
      // it.
      for (const auto& b : group.bindings) {
        auto p = lowerer.metadata(*b.positions, n, inputs, groups);
        if (!containsHole(c.resolve(p))) {
          c.require(call("is_valid_axis_list",
                         {p, call("len", {inputs[b.operand].interface.axes})}));
          order = c.select(inputs[b.operand].interface.axes, p);
          c.selectionFacts(inputs[b.operand].interface.axes, p, order);
          break;
        }
      }
      if (order)
        c.require(call("same_set", {order, wanted}));
      else {
        auto preferred = reduced.contains(g) ? reductions : source.axes;
        order =
            call("concat", {c.filter(preferred, wanted),
                            c.filter(inputs[*incidence.begin()].interface.axes,
                                     c.difference(wanted, preferred))});
      }
      for (const auto& b : group.bindings) {
        // Fixed selectors are checked by semantic lowering in their original
        // form. Only unresolved selectors need inversion into attributes.
        auto fixed =
            c.resolve(lowerer.metadata(*b.positions, n, inputs, groups));
        if (!containsHole(fixed)) continue;
        auto positions = inputs[b.operand].interface.lookup(order);
        if (!positions)
          positions =
              c.mappedPositions(inputs[b.operand].interface.axes, order);
        if (!positions)
          positions = c.positions(inputs[b.operand].interface.axes, order);
        assign(*b.positions, n, inputs, groups, positions);
      }
      groups.push_back(order);
    }
    if (infer_indices) {
      inferReadAttributes(n, inputs, groups);
      for (auto g : deferred_domains)
        assign(*n.definition->groups[g].domain, n, inputs, groups,
               c.extents(groups[g]));
    }
    // Generic inversion of output permutations. Preserve the source order at
    // the root and at internal reindexings with the same complete interface.
    std::function<void(const Meta&)> output = [&](const Meta& m) {
      if (m.kind != Meta::Kind::Call) return;
      if (m.name == "place") {
        auto left = lowerer.metadata(*m.args[0], n, inputs, groups);
        auto right = lowerer.metadata(*m.args[2], n, inputs, groups);
        auto desired = match.interfaces.at(id);
        c.require(call("same_set", {call("concat", {left, right}), desired}));
        auto p = lowerer.metadata(*m.args[1], n, inputs, groups);
        if (containsHole(p))
          assign(*m.args[1], n, inputs, groups, c.positions(desired, left));
      }
      if (m.name == "permute") {
        auto p = lowerer.metadata(*m.args[1], n, inputs, groups);
        if (containsHole(p)) {
          auto base = lowerer.metadata(*m.args[0], n, inputs, groups);
          auto wanted = id == c.plan.rhs ||
                                c.entails(call("same_set", {base, source.axes}))
                            ? source.axes
                            : base;
          c.require(call("same_set", {base, wanted}));
          assign(*m.args[1], n, inputs, groups, c.positions(base, wanted));
        }
      }
      for (const auto& a : m.args) output(*a);
    };
    output(*n.definition->output);
    // Target lowering checks every obligation before publishing this node's
    // maps. Rechecking after publication repeats proof work in a changed frame.
    // Trial metadata lowering can contain holes. The completed operation is
    // lowered afresh and checks its own obligations, without trial premises.
    lowerer.obligations.resize(trial_obligations);
    return lowerer.operation(id, std::move(inputs));
  }
};
}  // namespace

TensorMeaning construct(Context& c, Lowerer& lowerer,
                        const Correspondence& match,
                        const TensorMeaning& source, const S& reductions,
                        const Terms& source_terms) {
  return Constructor{c, lowerer, match, source, reductions, source_terms}.build(
      c.plan.rhs);
}
}  // namespace tepl::core::attr_synth
