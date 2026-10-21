#include "src/rewrite/inference.h"

#include <limits>

#include "src/inference/evaluate.h"

namespace tepl::rewrite {
bool inferAttributes(const executable::ProgramData& p,
                     const executable::Rule& r,
                     const core::ConcreteGraph& graph, Bindings& bindings,
                     Budget& budget) {
  if (!r.inference) return true;
  inference::Metadata m;
  for (auto capture : r.inference->source.back().captures) {
    budget.tick();
    const auto& node = graph.nodes.at(bindings.captures.at(capture));
    if (!node.shape) return false;
    budget.size(node.shape->size());
    m.shapes.emplace(capture, *node.shape);
  }
  auto descriptors = [&](const auto& code) {
    for (const auto& n : code)
      if (n.descriptor && !m.descriptors.contains(*n.descriptor)) {
        budget.tick();
        auto found = bindings.descriptors.find(*n.descriptor);
        if (found == bindings.descriptors.end())
          throw EvaluationError("missing inference descriptor");
        auto& fields = m.descriptors[*n.descriptor];
        for (const auto& field : found->second.fields) {
          const auto& values = list(field);
          budget.size(values.size());
          inference::List out;
          for (const auto& value : values) {
            budget.tick();
            auto i = integer(value);
            if (i < std::numeric_limits<std::int64_t>::min() ||
                i > std::numeric_limits<std::int64_t>::max())
              throw EvaluationError("inference position outside i64 range");
            out.push_back(static_cast<std::int64_t>(i));
          }
          fields.push_back(std::move(out));
        }
      }
  };
  descriptors(r.inference->source);
  if (r.inference->mode == inference::Mode::Validate)
    descriptors(r.inference->replacement);
  auto result =
      inference::evaluate(*r.inference, m, {budget.steps, budget.max_list});
  budget.charge(result.work);
  if (result.status == inference::Status::ResourceLimit)
    throw LimitError(result.reason);
  if (result.status != inference::Status::Solved) return false;
  if (r.inference->mode == inference::Mode::Validate) return true;
  for (const auto& [id, fields] : result.attributes) {
    auto schema = p.types.at(r.descriptors.at(id).type.value).schema;
    if (!schema) throw EvaluationError("inference descriptor has no schema");
    DescriptorValue out{schema->value, {}};
    for (const auto& field : fields) {
      List values;
      for (auto x : field) {
        budget.tick();
        values.emplace_back(Integer(x));
      }
      out.fields.emplace_back(std::move(values));
    }
    bindings.descriptors[id] = std::move(out);
  }
  return true;
}
}  // namespace tepl::rewrite
