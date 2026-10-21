#include "src/core/semantics/synthesis/ordered/requirements.h"

#include <functional>
#include <stdexcept>

namespace tepl::core::semantics::ordered {
Requirements deriveRequirements(Context& c, const Admission& admitted,
                                const Correspondence& match, const S& output) {
  if (!admitted.admitted())
    throw std::invalid_argument("requirements need an admitted ordered tree");
  Requirements result;
  std::function<void(std::size_t, S)> visit = [&](std::size_t id, S q) {
    c.tick();
    const auto& n = c.plan.nodes.at(id);
    auto& r = result.nodes[id];
    r.live = match.boundaries.at(id);
    r.required_order = std::move(q);
    if (n.capture) {
      result.order_constraints.push_back(call(
          "==", {r.required_order,
                 c.filter(match.anchors.at(id), c.support(r.required_order))}));
      return;
    }
    if (admitted.operations.at(id).kind == Contract::Kind::Permutation) {
      visit(n.inputs[0], call("empty_axes", {}));
      return;
    }
    auto left = match.boundaries.at(n.inputs[0]);
    auto right = match.boundaries.at(n.inputs[1]);
    auto shared = c.intersect(left, right);
    r.batch = c.intersect(shared, r.live);
    r.contract = c.difference(shared, r.live);
    r.left_free = c.difference(left, right);
    r.right_free = c.difference(right, left);
    auto qb = c.filter(r.required_order, r.batch);
    auto ql = c.filter(r.required_order, r.left_free);
    auto qr = c.filter(r.required_order, r.right_free);
    result.order_constraints.push_back(
        call("==", {r.required_order, call("concat", {qb, ql, qr})}));
    visit(n.inputs[0], ql);
    visit(n.inputs[1], qr);
  };
  visit(c.plan.rhs, output);
  return result;
}

}  // namespace tepl::core::semantics::ordered
