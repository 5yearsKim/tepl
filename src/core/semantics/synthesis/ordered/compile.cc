#include "src/core/semantics/synthesis/ordered/compile.h"

#include <functional>

#include "src/inference/evaluate.h"

namespace tepl::core::semantics::ordered {
Compilation compile(const Plan& plan, inference::Mode mode) {
  auto admission = admit(plan, mode);
  if (!admission.admitted()) return {{}, admission.reason};
  inference::Program out;
  out.mode = mode;
  auto block = [&](std::size_t root) {
    std::vector<inference::Instruction> code;
    std::function<std::size_t(std::size_t)> visit = [&](std::size_t id) {
      const auto& n = plan.nodes.at(id);
      inference::Instruction i;
      i.node = id;
      std::set<std::size_t> captures;
      for (auto child : n.inputs) {
        auto c = visit(child);
        i.inputs.push_back(c);
        captures.insert(code[c].captures.begin(), code[c].captures.end());
      }
      if (n.capture) {
        i.capture = n.capture->value;
        captures.insert(*i.capture);
      } else {
        i.descriptor = n.descriptor->value;
        if (n.operation) i.operation = n.operation->value;
        const auto& c = admission.operations.at(id);
        if (c.kind == Contract::Kind::Permutation) {
          i.kind = inference::Kind::Permutation;
          i.fields = {c.permutation};
        } else {
          i.kind = inference::Kind::Contraction;
          i.fields = {c.left_batch, c.right_batch, c.left_contract,
                      c.right_contract};
        }
      }
      i.captures.assign(captures.begin(), captures.end());
      code.push_back(std::move(i));
      return code.size() - 1;
    };
    visit(root);
    return code;
  };
  out.source = block(plan.lhs);
  out.replacement = block(plan.rhs);
  inference::validate(out);
  return {std::move(out), {}};
}
}  // namespace tepl::core::semantics::ordered
