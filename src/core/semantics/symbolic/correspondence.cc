#include <functional>

#include "src/core/semantics/symbolic/normalize.h"

namespace tepl::core::semantics {
namespace {
bool wholeOperand(const Binding& binding) {
  const auto& p = *binding.positions;
  return p.kind == Meta::Kind::Call && p.name == "all_axes" &&
         p.args.size() == 1 && p.args[0]->kind == Meta::Kind::Operand &&
         p.args[0]->index == binding.operand;
}
}  // namespace
Correspondence correspond(Context& c, const TensorMeaning& lhs, const Terms& lt,
                          const Terms& rt) {
  Correspondence result;
  // Use raw terms to establish coordinates. Coefficients and cancellation are
  // checked later in the fully lowered graph, not in the axis-free skeleton.
  std::map<std::string, std::vector<const Term*>> left;
  for (const auto& t : lt) left[footprint(t)].push_back(&t);
  std::vector<std::pair<const Term*, const Term*>> pairs;
  auto bindReads = [&](const std::vector<E>& reads, const Term& target) {
    std::map<std::size_t, S> axes;
    for (const auto& r : reads) {
      auto frame = r->access && r->access->kind == IndexExpr::Kind::Coordinates
                       ? r->access->axes
                       : r->axes;
      auto [it, inserted] = axes.emplace(r->capture, frame);
      if (!inserted && !c.entails(call("==", {it->second, frame})))
        throw ProofFailure(
            "ambiguous tensor access correspondence is unsupported");
    }
    for (const auto& r : target.reads) {
      if (!axes.contains(r->capture)) continue;
      auto [it, inserted] =
          result.anchors.emplace(r->occurrence, axes.at(r->capture));
      if (!inserted) c.require(call("==", {it->second, axes.at(r->capture)}));
    }
  };
  for (const auto& t : rt) {
    c.tick();
    auto k = footprint(t);
    if (!left.contains(k)) continue;
    std::vector<E> reads;
    for (const auto* source : left.at(k))
      reads.insert(reads.end(), source->reads.begin(), source->reads.end());
    bindReads(reads, t);
    const auto* source = left.at(k).front();
    for (std::size_t i = 0; i < t.reads.size(); ++i)
      if (t.reads[i]->capture != source->reads[i]->capture)
        c.note("commutative-matching");
    if (pairs.size() < lt.size() && k != footprint(lt[pairs.size()]))
      c.note("commutative-matching");
    pairs.emplace_back(source, &t);
  }
  // Shape-only operands and canceled subtrees still have coordinate interfaces.
  // Fall back to source occurrences only if their access frame is unambiguous.
  std::vector<E> source_reads;
  for (const auto& [id, axes] : lhs.occurrence_axes)
    source_reads.push_back(std::make_shared<Expr>(
        Expr{Expr::Kind::Read, c.plan.nodes[id].capture->value, id, axes, {}}));
  std::function<void(std::size_t)> fill = [&](std::size_t id) {
    const auto& n = c.plan.nodes[id];
    if (n.capture && !result.anchors.contains(id)) {
      std::vector<E> reads;
      for (const auto& r : source_reads)
        if (r->capture == n.capture->value) reads.push_back(r);
      Term target{call("empty_axes", {}),
                  {std::make_shared<Expr>(
                      Expr{Expr::Kind::Read, n.capture->value, id, {}, {}})}};
      bindReads(reads, target);
      if (!result.anchors.contains(id))
        throw ProofFailure(
            "shape-only capture has no established source interface");
    }
    for (auto child : n.inputs) fill(child);
  };
  fill(c.plan.rhs);
  bool has_domains = false;
  for (auto id : c.source().nodes)
    if (const auto& definition = c.plan.nodes[id].definition)
      for (const auto& group : definition->groups)
        has_domains |= bool(group.domain);
  auto independent_axes = has_domains
                              ? c.difference(lhs.axes, freeAxes(lhs.value))
                              : call("empty_axes", {});
  std::function<std::set<std::size_t>(std::size_t)> visit =
      [&](std::size_t id) {
        c.tick();
        const auto& n = c.plan.nodes[id];
        std::set<std::size_t> leaves;
        if (n.capture)
          leaves.insert(id);
        else
          for (auto child : n.inputs) {
            auto xs = visit(child);
            leaves.insert(xs.begin(), xs.end());
          }
        S boundary;
        for (const auto& [source, target] : pairs) {
          std::vector<S> inside, outside;
          bool active = false;
          for (const auto& r : target->reads) {
            bool local = leaves.contains(r->occurrence);
            active |= local;
            S dependencies = result.anchors.at(r->occurrence);
            for (const auto& sr : source->reads)
              if (sr->capture == r->capture && sr->access) {
                dependencies = freeAxes(sr);
                break;
              }
            (local ? inside : outside).push_back(dependencies);
          }
          if (!active) continue;
          auto b = c.intersect(
              call("union_axes", inside),
              call("union_axes", {call("union_axes", outside), lhs.axes}));
          // Additive terms may depend on different subsets of the output
          // coordinates, for example a tensor plus a broadcast scalar.
          boundary = boundary ? call("union_axes", {boundary, b}) : b;
        }
        if (!boundary) {
          std::vector<S> interfaces;
          for (auto leaf : leaves)
            interfaces.push_back(result.anchors.at(leaf));
          boundary = call("union_axes", std::move(interfaces));
        }
        if (has_domains && n.definition)
          boundary = call("union_axes", {boundary, independent_axes});
        result.boundaries[id] = boundary;
        std::vector<S> input_frames;
        for (auto leaf : leaves)
          input_frames.push_back(result.anchors.at(leaf));
        result.interfaces[id] =
            n.capture ? result.anchors.at(id)
            : id == c.plan.rhs
                ? lhs.axes
                : call("concat",
                       {c.filter(lhs.axes, boundary),
                        c.filter(call("union_axes", std::move(input_frames)),
                                 c.difference(boundary, lhs.axes))});
        // Internal whole-tensor operations can retain an operand's order.
        // Sorting that frame by the final output would unnecessarily permute
        // operands below an intervening broadcast or transpose.
        if (n.definition && id != c.plan.rhs &&
            n.definition->output->kind == Meta::Kind::Group) {
          const auto& group =
              n.definition->groups.at(n.definition->output->index);
          for (const auto& binding : group.bindings) {
            auto frame = result.interfaces.at(n.inputs.at(binding.operand));
            if (wholeOperand(binding)) {
              result.interfaces[id] = frame;
              break;
            }
          }
        }
        return leaves;
      };
  visit(c.plan.rhs);
  result.mandatory_boundaries = result.boundaries;
  // Value dependencies alone do not determine a broadcast's tensor interface.
  // Propagate the consumer's output frame through whole-operand bindings. This
  // follows the declared shape contract, including renamed/custom operations;
  // final lowering still proves every operand shape and coordinate obligation.
  auto propagate = [&](auto&& self, std::size_t id, const S& required) -> void {
    c.tick();
    const auto& n = c.plan.nodes[id];
    if (!n.definition) return;
    result.interfaces[id] = required;
    result.boundaries[id] =
        call("union_axes", {result.boundaries.at(id), required});
    const auto& output = *n.definition->output;
    std::vector<S> inputs;
    for (auto child : n.inputs) inputs.push_back(result.interfaces.at(child));
    if (output.kind == Meta::Kind::Group)
      for (const auto& binding :
           n.definition->groups.at(output.index).bindings) {
        if (wholeOperand(binding)) inputs.at(binding.operand) = required;
      }
    for (std::size_t operand = 0; operand < inputs.size(); ++operand)
      self(self, n.inputs[operand], inputs[operand]);
  };
  propagate(propagate, c.plan.rhs, lhs.axes);
  return result;
}
}  // namespace tepl::core::semantics
