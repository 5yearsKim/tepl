#include "src/rewrite/engine.h"

#include <functional>

#include "src/rewrite/builder.h"
#include "src/rewrite/matcher.h"

namespace tepl::rewrite {
std::string_view stopReasonName(StopReason s) {
  switch (s) {
    case StopReason::FixedPoint:
      return "fixed_point";
    case StopReason::StepLimit:
      return "step_limit";
    case StopReason::ResourceLimit:
      return "resource_limit";
    case StopReason::Cancelled:
      return "cancelled";
  }
  return "unknown";
}
Result run(const executable::ExecutableProgram& program,
           const core::ConcreteGraph& input, const Options& options) {
  // ExecutableProgram is an immutable snapshot validated by its only factories
  // (lowering and loading). Revalidating it here would repeat guard/DAG checks
  // on every application; only the caller-owned graph needs validation.
  executable::validateGraph(program.ir(), input);
  if (!options.rule.empty()) {
    std::size_t count = 0;
    for (const auto& rule : program.ir().rules)
      if (rule.name == options.rule) ++count;
    if (count != 1)
      throw std::invalid_argument(count
                                      ? "ambiguous rule '" + options.rule + "'"
                                      : "unknown rule '" + options.rule + "'");
  }
  Result out;
  out.graph = input;
  const auto& p = program.ir();
  auto budget = [&] {
    return Budget{options.max_expression_steps, options.max_list_items};
  };
  try {
    if (out.graph.nodes.size() > options.max_nodes)
      throw LimitError("graph node limit");
    auto initial = budget();
    for (std::size_t i = 0; i < out.graph.nodes.size(); ++i)
      infer(p, out.graph, i, initial);
    for (;;) {
      if (options.cancelled && options.cancelled()) {
        out.stop_reason = StopReason::Cancelled;
        return out;
      }
      std::set<std::size_t> reachable;
      std::vector<std::size_t> pending{out.graph.root};
      while (!pending.empty()) {
        auto id = pending.back();
        pending.pop_back();
        if (!reachable.insert(id).second) continue;
        for (auto a : out.graph.nodes.at(id).operands) pending.push_back(a);
      }
      bool changed = false;
      for (auto node : reachable) {
        for (const auto& rule : p.rules) {
          if (!options.rule.empty() && rule.name != options.rule) continue;
          if (options.cancelled && options.cancelled()) {
            out.stop_reason = StopReason::Cancelled;
            return out;
          }
          auto work = budget();
          Bindings bindings;
          try {
            if (!match(p, rule, out.graph, node, bindings, work)) continue;
            Evaluator evaluator(program, out.graph, rule.id.value, bindings,
                                work);
            if (!evaluator.evaluateWhere()) continue;
            evaluator.constructAttributes();
            auto candidate = out.graph;
            auto root = build(p, *rule.rhs, candidate, bindings, work,
                              options.max_nodes);
            const auto& old = out.graph.nodes[node];
            const auto& replacement = candidate.nodes[root];
            if ((old.shape && replacement.shape != old.shape) ||
                (old.dtype && replacement.dtype != old.dtype))
              continue;
            candidate = replace(candidate, node, root, out.graph.nodes.size());
            if (sameGraph(out.graph, candidate)) continue;
            // Infer again in topological order, including all surviving users.
            for (std::size_t i = 0; i < candidate.nodes.size(); ++i)
              infer(p, candidate, i, work);
            if (out.rewrite_count >= options.max_steps) {
              out.stop_reason = StopReason::StepLimit;
              return out;
            }
            out.graph = std::move(candidate);
            ++out.rewrite_count;
            if (options.trace) out.trace.push_back({rule.name, node});
            changed = true;
            break;
          } catch (const ConditionUnavailable&) {
            continue;
          }
        }
        if (changed) break;
      }
      if (!changed) return out;
    }
  } catch (const LimitError& e) {
    out.stop_reason = StopReason::ResourceLimit;
    out.detail = e.what();
    return out;
  }
}
}  // namespace tepl::rewrite
