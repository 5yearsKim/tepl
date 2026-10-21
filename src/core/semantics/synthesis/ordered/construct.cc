#include "src/core/semantics/synthesis/ordered/construct.h"

#include <functional>
#include <stdexcept>

namespace tepl::core::semantics::ordered {

TensorMeaning construct(Context& c, Lowerer& lowerer, const Admission& admitted,
                        const Requirements& required,
                        const S& reduction_order) {
  if (!admitted.admitted())
    throw std::invalid_argument("construction needs an admitted ordered tree");
  for (const auto& constraint : required.order_constraints)
    c.require(constraint, GuardKind::CoordinateOrder);
  auto extend = [&](const S& q, const S& set, const S& traversal) {
    return call("concat", {q, c.filter(traversal, c.difference(set, q))});
  };
  std::function<TensorMeaning(std::size_t)> build = [&](std::size_t id) {
    c.tick();
    const auto& n = c.plan.nodes.at(id);
    if (n.capture) return lowerer.lower(id);
    std::vector<TensorMeaning> inputs;
    for (auto child : n.inputs) inputs.push_back(build(child));
    const auto& contract = admitted.operations.at(id);
    const auto& r = required.nodes.at(id);
    auto assign = [&](std::size_t field, const S& value) {
      c.assignAttribute(n.descriptor->value, field, value);
    };
    if (contract.kind == Contract::Kind::Permutation) {
      auto output = extend(r.required_order, r.live, inputs[0].axes);
      assign(contract.permutation, c.positions(inputs[0].axes, output));
    } else {
      auto qb = c.filter(r.required_order, r.batch);
      auto batches = extend(qb, r.batch, inputs[0].axes);
      auto contractions = extend(c.filter(reduction_order, r.contract),
                                 r.contract, inputs[0].axes);
      assign(contract.left_batch, c.positions(inputs[0].axes, batches));
      assign(contract.right_batch, c.positions(inputs[1].axes, batches));
      assign(contract.left_contract, c.positions(inputs[0].axes, contractions));
      assign(contract.right_contract,
             c.positions(inputs[1].axes, contractions));
    }
    return lowerer.operation(id, std::move(inputs));
  };
  auto result = build(c.plan.rhs);
  c.note("ordered-requirements-construction");
  return result;
}
}  // namespace tepl::core::semantics::ordered
