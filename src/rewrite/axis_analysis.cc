#include "src/rewrite/axis_analysis.h"

#include <algorithm>
#include <limits>

#include "src/rewrite/evaluator.h"

namespace tepl::rewrite {
void AxisAnalysis::metadata(bool replacement) {
  if (!rule_.inference)
    throw EvaluationError("axis query requires a synthesis plan");
  const auto& plan = *rule_.inference;
  if (metadata_.shapes.empty())
    for (auto capture : plan.source.back().captures) {
      budget_.tick();
      const auto& node = graph_.nodes.at(bindings_.captures.at(capture));
      if (!node.shape)
        throw ConditionUnavailable("axis input shape unavailable");
      budget_.size(node.shape->size());
      metadata_.shapes.emplace(capture, *node.shape);
    }
  for (const auto& n : replacement ? plan.replacement : plan.source)
    if (n.descriptor && !metadata_.descriptors.contains(*n.descriptor)) {
      budget_.tick();
      auto found = bindings_.descriptors.find(*n.descriptor);
      if (found == bindings_.descriptors.end())
        throw EvaluationError("missing axis-analysis descriptor");
      auto& fields = metadata_.descriptors[*n.descriptor];
      for (const auto& field : found->second.fields) {
        const auto& values = list(field);
        budget_.size(values.size());
        inference::List out;
        for (const auto& value : values) {
          budget_.tick();
          auto i = integer(value);
          if (i < std::numeric_limits<std::int64_t>::min() ||
              i > std::numeric_limits<std::int64_t>::max())
            throw EvaluationError("axis position outside i64 range");
          out.push_back(static_cast<std::int64_t>(i));
        }
        fields.push_back(std::move(out));
      }
    }
}
void AxisAnalysis::charge(const inference::Result& r) {
  budget_.charge(r.work);
  if (r.status == inference::Status::ResourceLimit) throw LimitError(r.reason);
}
const inference::Result& AxisAnalysis::source() {
  if (!source_) {
    metadata(false);
    source_ = inference::analyze(*rule_.inference, metadata_,
                                 {budget_.steps, budget_.max_list});
    charge(*source_);
    for (const auto& n : rule_.inference->replacement) {
      budget_.tick();
      nodes_.emplace(n.node, &n);
    }
  }
  return *source_;
}
const inference::Result& AxisAnalysis::replacement() {
  if (!replacement_) {
    if (rule_.inference->mode != inference::Mode::Validate)
      throw EvaluationError(
          "replacement axis queries require explicit RHS attributes");
    metadata(true);
    replacement_ = inference::replayReplacement(
        *rule_.inference, metadata_, {budget_.steps, budget_.max_list});
    charge(*replacement_);
  }
  return *replacement_;
}
Value AxisAnalysis::axes(const inference::AxisSequence& a) {
  List out;
  budget_.size(a.size());
  for (auto id : a) {
    budget_.tick();
    // Canonical physical-port class IDs are scoped to this source network.
    out.emplace_back(
        Axis{0, 0, id, Integer(source().source.ports.at(id).extent)});
  }
  return out;
}
Value AxisAnalysis::query(core::builtins::Builtin b,
                          const std::vector<Value>& args) {
  using B = core::builtins::Builtin;
  const auto& s = source();
  if (b == B::kAxisDomain) return s.status == inference::Status::Analyzed;
  if (s.status != inference::Status::Analyzed)
    throw ConditionUnavailable(s.reason);
  if (b == B::kAxisRhsLegal)
    return replacement().status == inference::Status::Analyzed;
  if (b == B::kAxisCaptureOrder)
    return axes(s.source.inputs.at(position(args.at(0))));
  if (b == B::kAxisSourceNodeOrder || b == B::kAxisRhsNodeOrder) {
    const auto& r = b == B::kAxisSourceNodeOrder ? s : replacement();
    if (r.status != inference::Status::Analyzed)
      throw ConditionUnavailable(r.reason);
    return axes(r.replay_layouts.at(position(args.at(0))));
  }
  if (b == B::kAxisSourceGroupOrder || b == B::kAxisSourceGroupMembers ||
      b == B::kAxisRhsGroupOrder || b == B::kAxisRhsGroupMembers ||
      b == B::kAxisRhsGroupPositions) {
    const bool rhs = core::builtins::isReplacementAxisQuery(b);
    const auto& r = rhs ? replacement() : s;
    if (r.status != inference::Status::Analyzed)
      throw ConditionUnavailable(r.reason);
    auto id = position(args.at(0)), role = position(args.at(1));
    auto group = r.replay_groups.at(id).at(role);
    if (b == B::kAxisRhsGroupPositions) {
      const auto& order = r.replay_layouts.at(id);
      List out;
      for (auto a : group) {
        budget_.tick();
        auto it = std::find(order.begin(), order.end(), a);
        if (it == order.end())
          throw EvaluationError("semantic group is absent from the output");
        out.emplace_back(Integer(it - order.begin()));
      }
      return out;
    }
    if (b == B::kAxisSourceGroupMembers || b == B::kAxisRhsGroupMembers)
      std::sort(group.begin(), group.end());
    return axes(group);
  }
  if (b == B::kAxisSourceOrder) return axes(s.source.output);
  if (b == B::kAxisSourcePartition || b == B::kAxisRhsPartition) {
    const auto& r = b == B::kAxisSourcePartition ? s : replacement();
    if (r.status != inference::Status::Analyzed)
      throw ConditionUnavailable(r.reason);
    List out;
    for (auto id : r.source.classes) {
      budget_.tick();
      out.emplace_back(Integer(id));
    }
    return out;
  }
  if (b == B::kAxisRhsOrder) {
    const auto& r = replacement();
    if (r.status != inference::Status::Analyzed)
      throw ConditionUnavailable(r.reason);
    return axes(r.source.output);
  }
  const auto id = position(args.at(0));
  const auto found = nodes_.find(id);
  if (found == nodes_.end())
    throw EvaluationError("unknown replacement axis node");
  const auto* node = found->second;
  if (b == B::kAxisInputOrder) {
    if (!node->capture)
      throw EvaluationError("axis_input_order expects an input node");
    return axes(s.source.inputs.at(*node->capture));
  }
  auto it = s.requirements.find(id);
  if (it == s.requirements.end())
    throw EvaluationError("axis requirements unavailable");
  const auto& r = it->second;
  if (b == B::kAxisRequired) return axes(r.required_order);
  const inference::AxisSet* set = nullptr;
  if (b == B::kAxisBatch) set = &r.batch;
  if (b == B::kAxisContract) set = &r.contract;
  if (b == B::kAxisLive) set = &r.live;
  if (b == B::kAxisLeftFree) set = &r.left_free;
  if (b == B::kAxisRightFree) set = &r.right_free;
  if (!set) throw EvaluationError("unknown axis query");
  return axes(inference::AxisSequence(set->begin(), set->end()));
}
void AxisAnalysis::constructAttributes() {
  if (!rule_.inference || rule_.inference->mode == inference::Mode::Validate)
    return;
  auto result = inference::construct(*rule_.inference, source(),
                                     {budget_.steps, budget_.max_list});
  charge(result);
  if (result.status != inference::Status::Solved)
    throw EvaluationError(
        "attribute constructor invoked without established preconditions");
  for (const auto& [id, fields] : result.attributes) {
    auto schema = program_.types.at(rule_.descriptors.at(id).type.value).schema;
    if (!schema) throw EvaluationError("attribute constructor has no schema");
    DescriptorValue out{schema->value, {}};
    for (const auto& field : fields) {
      List values;
      for (auto x : field) {
        budget_.tick();
        values.emplace_back(Integer(x));
      }
      out.fields.emplace_back(std::move(values));
    }
    bindings_.descriptors[id] = std::move(out);
  }
}
}  // namespace tepl::rewrite
