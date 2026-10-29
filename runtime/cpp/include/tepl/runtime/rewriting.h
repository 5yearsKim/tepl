#pragma once
#include "matching.h"
namespace tepl_runtime {
template <Program P>
struct PreparedReplacement {
  std::optional<eggc::Id> existing;
  Node<P> node;
  std::vector<PreparedReplacement> children;
  TensorFacts facts;
  eggc::Id insert(BaseGraph<P>& graph) const {
    if (existing) return graph.find(*existing);
    auto n = node;
    for (std::size_t i = 0; i < children.size(); ++i)
      n.children_mut()[i] = children[i].insert(graph);
    return graph.add(std::move(n));
  }
};
template <Program P>
struct RuleContext {
  const BaseGraph<P>& graph;
  eggc::Id root;
  matching::Bindings bindings;
  Budget budget;
  std::map<std::pair<std::size_t, std::size_t>, Value> axis_cache;
  std::set<std::pair<std::size_t, std::size_t>> active_axes;
  RuleContext(const BaseGraph<P>& graph, eggc::Id root,
              matching::Bindings bindings)
      : graph(graph), root(root), bindings(std::move(bindings)) {}
  static Value lookup(const std::map<std::size_t, Value>& xs, std::size_t id,
                      const char* message) {
    auto it = xs.find(id);
    require(it != xs.end(), message);
    return it->second;
  }
  Value capture(std::size_t id) const {
    auto it = bindings.captures.find(id);
    require(it != bindings.captures.end(), "missing capture");
    return Value(Tensor{graph.find(it->second)});
  }
  Value descriptor(std::size_t id) const {
    return lookup(bindings.descriptors, id, "missing attributes");
  }
  Value dimension(std::size_t id) const {
    return lookup(bindings.dimensions, id, "missing dimension");
  }
  Value dtype_variable(std::size_t id) const {
    return lookup(bindings.dtypes, id, "missing dtype variable");
  }
  Value local(std::size_t id) const {
    return lookup(bindings.locals, id, "missing local");
  }
  const TensorFacts& facts(std::size_t id) const {
    auto it = bindings.captures.find(id);
    require(it != bindings.captures.end(), "missing capture");
    return graph.analysis_data(it->second);
  }
  Value shape(std::size_t id) const { return facts(id).shape_value(); }
  Value dtype(std::size_t id) const { return facts(id).dtype_value(); }
  std::optional<Value> cached_axis(std::size_t owner, std::size_t group) const {
    auto it = axis_cache.find({owner, group});
    return it == axis_cache.end() ? std::nullopt : std::optional(it->second);
  }
  void enter_axis(std::size_t owner, std::size_t group) {
    require(active_axes.emplace(owner, group).second, "cyclic axis definition");
  }
  Value finish_axis(std::size_t owner, std::size_t group,
                    const Value& extents) {
    Value::List xs;
    for (auto x : extents.dimensions())
      xs.emplace_back(Axis{owner, group, xs.size(), Integer(x)});
    Value out(xs);
    active_axes.erase({owner, group});
    axis_cache[{owner, group}] = out;
    return out;
  }
  PreparedReplacement<P> captured_rhs(std::size_t id) const {
    auto it = bindings.captures.find(id);
    require(it != bindings.captures.end(), "missing capture");
    return {it->second, {}, {}, graph.analysis_data(it->second)};
  }
  PreparedReplacement<P> prepared(
      Node<P> node, std::vector<PreparedReplacement<P>> children) {
    std::vector<TensorFacts> operands;
    for (const auto& child : children) operands.push_back(child.facts);
    auto facts = TensorAnalysis<P>::infer_node(node, operands, budget);
    return {{}, std::move(node), std::move(children), std::move(facts)};
  }
  PreparedReplacement<P> operation_rhs(
      std::size_t id, Value attrs,
      std::vector<PreparedReplacement<P>> children) {
    std::optional<typename P::Attrs> a;
    if (!attrs.none()) a = P::Attrs::from_value(attrs).value();
    auto node = Node<P>::operation(P::operation(id), std::move(a),
                                   std::vector<eggc::Id>(children.size(), 0))
                    .value();
    return prepared(std::move(node), std::move(children));
  }
  PreparedReplacement<P> literal_rhs(std::string text,
                                     std::optional<DType> dtype) {
    if (!dtype && graph.analysis_data(root).dtype.is_known())
      dtype = graph.analysis_data(root).dtype.value;
    return prepared(Node<P>::literal(std::move(text), dtype).value(), {});
  }
};
template <Program P>
using Rewrite = eggc::Rewrite<Node<P>, TensorAnalysis<P>>;
template <Program P>
struct RuleDefinition {
  std::string name, source;
  matching::Pattern lhs;
  std::vector<matching::Constraint> constraints;
  EvalResult<std::optional<PreparedReplacement<P>>> (*prepare)(RuleContext<P>&);
  EvalResult<Rewrite<P>> build() const {
    return evaluate([&] {
      auto definition = std::make_shared<const RuleDefinition>(*this);
      return Rewrite<P>(name, [definition](
                                  const BaseGraph<P>& graph,
                                  const typename Rewrite<P>::Sink& sink,
                                  const eggc::StopCheck& stop) {
        std::shared_ptr<AnalysisState> diagnostics;
        try {
          Budget budget;
          for (auto root : graph.classes()) {
            if (stop && stop()) return false;
            auto state = analysis_state<P>(graph, root);
            diagnostics = state;
            state->check();
            auto witnesses = matching::matches<P>(definition->lhs, root, graph,
                                                  budget, stop);
            auto batch = std::make_shared<
                std::optional<std::vector<matching::Bindings>>>();
            std::set<std::vector<std::pair<std::size_t, eggc::Id>>> seen;
            for (const auto& witness : witnesses) {
              std::vector<std::pair<std::size_t, eggc::Id>> key;
              for (auto [c, id] : witness.captures)
                key.emplace_back(c, graph.find(id));
              if (!seen.insert(key).second) continue;
              // The application owns only tensor substitutions. It recovers
              // concrete attribute witnesses from the current clean graph.
              if (!sink(
                      {root,
                       [definition, root, key, state, batch](
                           BaseGraph<P>& current) -> std::optional<eggc::Id> {
                         try {
                           current.rebuild();
                           state->check();
                           Budget budget;
                           if (!*batch)
                             *batch = matching::matches<P>(
                                 definition->lhs, root, current, budget);
                           std::vector<PreparedReplacement<P>> prepared;
                           for (auto binding : **batch) {
                             bool same = binding.captures.size() == key.size();
                             for (auto [c, id] : key) {
                               auto it = binding.captures.find(c);
                               same =
                                   same && it != binding.captures.end() &&
                                   current.find(it->second) == current.find(id);
                             }
                             if (!same) continue;
                             try {
                               if (!matching::constraints<P>(
                                       definition->constraints, binding,
                                       current))
                                 continue;
                               RuleContext<P> ctx(current, root,
                                                  std::move(binding));
                               auto rhs = definition->prepare(ctx).value();
                               if (rhs) {
                                 if (rhs->facts.conflicts())
                                   throw EvalError(
                                       ErrorKind::Conflict,
                                       "conflicting replacement metadata");
                                 if (!rhs->facts.disagrees(
                                         current.analysis_data(root)))
                                   prepared.push_back(std::move(*rhs));
                               }
                             } catch (const EvalError& error) {
                               if (error.kind == ErrorKind::Unavailable)
                                 continue;
                               if (error.kind == ErrorKind::Invalid) {
                                 state->reject(definition->source,
                                               definition->name, error);
                                 continue;
                               }
                               throw;
                             }
                           }
                           std::optional<eggc::Id> result;
                           for (const auto& rhs : prepared) {
                             if (rhs.facts.disagrees(
                                     current.analysis_data(root)))
                               continue;
                             // egg-c unions the returned ID with root after
                             // this callback. Earlier witnesses can already
                             // refine the replacement's facts in this batch.
                             if (result && rhs.facts.disagrees(
                                               current.analysis_data(*result)))
                               continue;
                             auto id = rhs.insert(current);
                             state->check();
                             if (result)
                               current.merge(
                                   *result, id,
                                   eggc::Justification{eggc::UnionKind::Rewrite,
                                                       definition->name,
                                                       {}});
                             else
                               result = id;
                             current.rebuild();
                             state->check();
                           }
                           return result;
                         } catch (const EvalError& error) {
                           state->report(error);
                           throw;
                         }
                       }}))
                return false;
            }
          }
          return true;
        } catch (const EvalError& error) {
          if (error.kind == ErrorKind::Limit && stop && stop()) return false;
          if (diagnostics) diagnostics->report(error);
          throw;
        }
      });
    });
  }
};
}  // namespace tepl_runtime
