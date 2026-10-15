#pragma once
#include <eggc/rewrite.hpp>

#include "checks.h"

namespace @TEPL_NAMESPACE@::rewriting {
namespace detail {
struct SearchState {
  TensorMatch matched;
  MetadataBindings shapes;
  ::std::size_t next_condition = 0;
};
// Continuation matching streams witnesses, preserving branch-local bindings
// and cooperative cancellation without materializing Cartesian products.
template <class A, class M>
class Matcher {
 public:
  using Graph = ::eggc::EGraph<OpNode, A>;
  using Visit = ::std::function<bool(SearchState)>;
  const Graph& graph;
  const MatchChecks<A>& checks;
  const M& metadata;
  const ::eggc::StopCheck& stop;
  bool cancelled() const { return stop && stop(); }
  bool bind(SearchState& state, BindingId capture, ::eggc::Id id) const {
    id = graph.find(id);
    auto [entry, added] = state.matched.tensors.emplace(capture, id);
    if (!added) return graph.find(entry->second) == id;
    return checks.tensors.check_capture(graph, capture, id, metadata,
                                        state.shapes) &&
           checks.advance(graph, state.matched, state.shapes,
                          state.next_condition);
  }
  bool children(const TensorPattern& pattern, const OpNode& node,
                ::std::size_t index, SearchState state,
                const Visit& emit) const {
    if (cancelled()) return false;
    if (index == pattern.children.size()) return emit(::std::move(state));
    return match(pattern.children[index], node.children()[index],
                 ::std::move(state), [&](SearchState next) {
                   return children(pattern, node, index + 1, ::std::move(next),
                                   emit);
                 });
  }
  bool match(const TensorPattern& pattern, ::eggc::Id id, SearchState state,
             const Visit& emit) const {
    if (cancelled()) return false;
    id = graph.find(id);
    if (pattern.kind == TensorPattern::Kind::Var ||
        pattern.kind == TensorPattern::Kind::Bind) {
      if (!bind(state, pattern.capture, id)) return true;
      if (pattern.kind == TensorPattern::Kind::Var)
        return emit(::std::move(state));
      return match(pattern.children.at(0), id, ::std::move(state), emit);
    }
    for (const auto& node : graph.nodes(id)) {
      if (cancelled()) return false;
      if (pattern.kind == TensorPattern::Kind::Literal) {
        if (node.op() != Op::Literal) continue;
        const auto& value = ::std::get<LiteralAttrs>(node.attrs().value);
        if (value.value == pattern.literal_value.value &&
            (!pattern.literal_value.dtype ||
             value.dtype == pattern.literal_value.dtype))
          if (!emit(state)) return false;
        continue;
      }
      if (node.op() != pattern.operation ||
          node.children().size() != pattern.children.size())
        continue;
      auto branch = state;
      if (pattern.attributes.binding) {
        auto [entry, added] = branch.matched.attrs.emplace(
            *pattern.attributes.binding, node.attrs());
        if (!added && entry->second != node.attrs()) continue;
      } else if (node.attrs() != pattern.attributes.exact)
        continue;
      if (!checks.advance(graph, branch.matched, branch.shapes,
                          branch.next_condition))
        continue;
      if (!children(pattern, node, 0, ::std::move(branch), emit)) return false;
    }
    return true;
  }
};
}  // namespace detail
template <class A, class M, class F>
bool for_each_match_at(const ::eggc::EGraph<OpNode, A>& graph, ::eggc::Id root,
                       const TensorPattern& pattern,
                       const MatchChecks<A>& checks, const M& metadata, F emit,
                       const ::eggc::StopCheck& stop = {}) {
  graph.require_clean();
  if (stop && stop()) return false;
  detail::Matcher<A, M> matcher{graph, checks, metadata, stop};
  detail::SearchState state{{graph.find(root), {}, {}}, {}, 0};
  if (!checks.advance(graph, state.matched, state.shapes, state.next_condition))
    return true;
  return matcher.match(pattern, root, ::std::move(state),
                       [&](detail::SearchState found) {
                         if (!checks.advance(graph, found.matched, found.shapes,
                                             found.next_condition) ||
                             found.next_condition != checks.conditions.size())
                           return true;
                         return emit(::std::move(found.matched));
                       });
}
template <class A, class M>
::std::vector<TensorMatch> matches_at_with_checks(
    const ::eggc::EGraph<OpNode, A>& graph, ::eggc::Id root,
    const TensorPattern& pattern, const MatchChecks<A>& checks,
    const M& metadata) {
  checks.validate(pattern);
  ::std::vector<TensorMatch> result;
  for_each_match_at(graph, root, pattern, checks, metadata,
                    [&](TensorMatch match) {
                      result.push_back(::std::move(match));
                      return true;
                    });
  return result;
}
template <class A>
::std::vector<TensorMatch> matches_at(const ::eggc::EGraph<OpNode, A>& graph,
                                      ::eggc::Id root,
                                      const TensorPattern& pattern) {
  return matches_at_with_checks(
      graph, root, pattern, MatchChecks<A>{},
      [](const auto&, ::eggc::Id) -> ::std::optional<TensorInfo> {
        return {};
      });
}
template <class A, class M>
::std::vector<TensorMatch> matches_at_with_constraints(
    const ::eggc::EGraph<OpNode, A>& graph, ::eggc::Id root,
    const TensorPattern& pattern, TensorConstraints constraints,
    const M& metadata) {
  return matches_at_with_checks(
      graph, root, pattern, MatchChecks<A>{::std::move(constraints), {}, {}},
      metadata);
}
}  // namespace @TEPL_NAMESPACE@::rewriting
