#pragma once
#include <memory>

#include "matcher.h"

namespace @TEPL_NAMESPACE@::pattern {
namespace detail {
struct PreparedRhs {
  ::std::optional<::eggc::Id> tensor;
  ::std::optional<OpNode> node;
  ::std::vector<PreparedRhs> children;
};
inline PreparedRhs prepare_rhs(const TensorExpr& expr,
                               const TensorMatch& matched,
                               const DerivedAttrs& derived) {
  if (expr.kind == TensorExpr::Kind::Var) {
    auto i = matched.tensors.find(expr.capture);
    if (i == matched.tensors.end())
      throw builtins::BuiltinError("unavailable tensor capture");
    return {i->second, {}, {}};
  }
  Op op = expr.operation;
  OpAttrs attrs;
  if (expr.kind == TensorExpr::Kind::Literal) {
    op = Op::Literal;
    attrs = expr.literal_value;
  } else {
    const auto& values = expr.attributes.kind == AttrExpr::Kind::Derived
                             ? derived
                             : matched.attrs;
    if (expr.attributes.kind == AttrExpr::Kind::Exact)
      attrs = expr.attributes.value;
    else {
      auto i = values.find(expr.attributes.binding);
      if (i == values.end())
        throw builtins::BuiltinError("unavailable descriptor");
      attrs = i->second;
    }
  }
  auto node = OpNode::from_parts(
      op, ::std::vector<::eggc::Id>(expr.children.size(), ::eggc::Id{}),
      ::std::move(attrs));
  PreparedRhs result{{}, ::std::move(node), {}};
  for (const auto& child : expr.children)
    result.children.push_back(prepare_rhs(child, matched, derived));
  return result;
}
template <class A, class M, class I>
TensorInfo infer_rhs(const ::eggc::EGraph<OpNode, A>& graph, PreparedRhs& rhs,
                     const M& metadata, const I& inference,
                     const TensorInfo& expected) {
  if (rhs.tensor)
    return builtins::require(metadata(graph, graph.find(*rhs.tensor)));
  ::std::vector<TensorInfo> operands;
  for (auto& child : rhs.children)
    operands.push_back(infer_rhs(graph, child, metadata, inference, expected));
  auto node = *rhs.node;
  ::std::optional<DType> literal_dtype;
  if (node.op() == Op::Literal) {
    auto literal = ::std::get<LiteralAttrs>(node.attrs().value);
    if constexpr (requires {
                    inference.infer_literal(literal.value, literal.dtype,
                                            expected);
                  })
      literal_dtype =
          inference.infer_literal(literal.value, literal.dtype, expected);
    else
      literal_dtype =
          literal.dtype ? literal.dtype : ::std::optional(expected.dtype);
    if (!literal_dtype || (literal.dtype && literal.dtype != literal_dtype))
      throw builtins::BuiltinError("invalid literal dtype");
    literal.dtype = literal_dtype;
    node = OpNode::from_parts(Op::Literal, {}, literal);
    rhs.node = node;
  }
  auto result = [&]() -> ::std::optional<TensorInfo> {
    if constexpr (requires {
                    inference.infer_output(
                        node.op(), ::std::span<const TensorInfo>(operands),
                        node.attrs());
                  })
      return inference.infer_output(
          node.op(), ::std::span<const TensorInfo>(operands), node.attrs());
    else
      return inference(node.op(), ::std::span<const TensorInfo>(operands),
                       node.attrs());
  }();
  auto info = builtins::require(::std::move(result));
  if (literal_dtype && (info.dtype != *literal_dtype || !info.shape.empty()))
    throw builtins::BuiltinError(
        "literal must have rank zero and its resolved dtype");
  return info;
}
template <class A>
::eggc::Id insert_rhs(::eggc::EGraph<OpNode, A>& graph, PreparedRhs rhs) {
  if (rhs.tensor) return graph.find(*rhs.tensor);
  auto node = ::std::move(*rhs.node);
  for (::std::size_t i = 0; i < rhs.children.size(); ++i)
    node.children_mut()[i] = insert_rhs(graph, ::std::move(rhs.children[i]));
  return graph.add(::std::move(node));
}
template <class A>
::std::vector<::eggc::Id> substitution_key(
    const ::eggc::EGraph<OpNode, A>& graph, const TensorMatch& match) {
  ::std::vector<::eggc::Id> result;
  for (auto [capture, id] : match.tensors) result.push_back(graph.find(id));
  return result;
}
inline void validate_rhs(const TensorExpr& rhs,
                         const ::std::set<BindingId>& tensors,
                         const ::std::set<BindingId>& attrs) {
  if (rhs.kind == TensorExpr::Kind::Var && !tensors.contains(rhs.capture))
    throw ::std::invalid_argument("unavailable RHS tensor");
  if (rhs.kind == TensorExpr::Kind::Literal &&
      (!valid_literal(rhs.literal_value.value) ||
       (rhs.literal_value.dtype &&
        !accepts_literal(*rhs.literal_value.dtype, rhs.literal_value.value))))
    throw ::std::invalid_argument("invalid RHS literal");
  if (rhs.kind == TensorExpr::Kind::Op) {
    if (!rhs.operation.arity().accepts(rhs.children.size()))
      throw ::std::invalid_argument("invalid RHS arity");
    if (rhs.attributes.kind == AttrExpr::Kind::Captured &&
        !attrs.contains(rhs.attributes.binding))
      throw ::std::invalid_argument("unavailable RHS attributes");
    if (rhs.attributes.kind == AttrExpr::Kind::Exact &&
        !rhs.operation.accepts_attrs(rhs.attributes.value))
      throw ::std::invalid_argument("invalid RHS attributes");
  }
  for (const auto& child : rhs.children) validate_rhs(child, tensors, attrs);
}
}  // namespace detail
template <class A, class M, class I, class F>
::eggc::Rewrite<OpNode, A> tensor_rewrite_checked_with_checks(
    ::std::string name, TensorPattern lhs, TensorExpr rhs,
    MatchChecks<A> checks, M metadata, I inference, F check_and_derive) {
  checks.validate(lhs);
  ::std::set<BindingId> tensors, attrs;
  lhs.bindings(tensors, attrs);
  detail::validate_rhs(rhs, tensors, attrs);
  struct State {
    ::std::string name;
    TensorPattern lhs;
    TensorExpr rhs;
    MatchChecks<A> checks;
    M metadata;
    I inference;
    F callback;
  };
  auto state = ::std::make_shared<State>(
      State{::std::move(name), ::std::move(lhs), ::std::move(rhs),
            ::std::move(checks), ::std::move(metadata), ::std::move(inference),
            ::std::move(check_and_derive)});
  using Key = ::std::vector<::eggc::Id>;
  struct MatchBatch {
    ::std::vector<Key> keys;
    ::std::map<Key, ::std::vector<TensorMatch>> groups;
    bool initialized = false;
  };
  using Rewrite = ::eggc::Rewrite<OpNode, A>;
  return Rewrite(state->name, [state](const ::eggc::EGraph<OpNode, A>& graph,
                                      const typename Rewrite::Sink& sink,
                                      const ::eggc::StopCheck& stop) {
    auto root_op = state->lhs.root_op();
    auto classes =
        root_op ? graph.classes_for_op(root_op->id()) : graph.classes();
    for (auto root : classes) {
      ::std::set<::std::vector<::eggc::Id>> seen;
      auto batch = ::std::make_shared<MatchBatch>();
      bool completed = for_each_match_at(
          graph, root, state->lhs, state->checks, state->metadata,
          [&](TensorMatch matched) {
            auto key = detail::substitution_key(graph, matched);
            if (!seen.insert(key).second) return true;
            auto index = batch->keys.size();
            batch->keys.push_back(::std::move(key));
            return sink(
                {root,
                 [state, root, batch, index](::eggc::EGraph<OpNode, A>& current)
                     -> ::std::optional<::eggc::Id> {
                   // The runner batches actions. Repair before querying current
                   // nodes; copied witnesses and prepared trees survive
                   // subsequent mutations.
                   if (!current.is_clean()) current.rebuild();
                   auto target = current.find(root);
                   // Like Rust's apply_matches, rematch once per e-class and
                   // retain all attribute witnesses for the searched keys.
                   // E-graph unions preserve these structural witnesses; each
                   // application rechecks metadata and semantic conditions.
                   // A new search owns a fresh batch and sees new witnesses.
                   if (!batch->initialized) {
                     for (auto& key : batch->keys) {
                       for (auto& id : key) id = current.find(id);
                       batch->groups.try_emplace(key);
                     }
                     for_each_match_at(
                         current, target, state->lhs, state->checks,
                         state->metadata, [&](TensorMatch candidate) {
                           auto group = batch->groups.find(
                               detail::substitution_key(current, candidate));
                           if (group != batch->groups.end())
                             group->second.push_back(::std::move(candidate));
                           return true;
                         });
                     batch->initialized = true;
                   }
                   const auto& candidates =
                       batch->groups.at(batch->keys.at(index));
                   ::std::vector<detail::PreparedRhs> prepared;
                   for (const auto& candidate : candidates) {
                     try {
                       auto shapes = state->checks.check_match(
                           current, candidate, state->metadata);
                       if (!shapes) continue;
                       auto derived =
                           state->callback(current, candidate, *shapes);
                       if (!derived) continue;
                       auto replacement =
                           detail::prepare_rhs(state->rhs, candidate, *derived);
                       auto expected =
                           builtins::require(state->metadata(current, target));
                       if (detail::infer_rhs(current, replacement,
                                             state->metadata, state->inference,
                                             expected) != expected)
                         continue;
                       prepared.push_back(::std::move(replacement));
                     } catch (const builtins::BuiltinError&) {
                     } catch (const NodeError&) {
                     }
                   }
                   if (prepared.empty()) return {};
                   ::std::vector<::eggc::Id> replacements;
                   for (auto& replacement : prepared)
                     replacements.push_back(
                         detail::insert_rhs(current, ::std::move(replacement)));
                   // The runner merges the final replacement; earlier witnesses
                   // use the same attribution. All semantic work finished
                   // before any insertion.
                   for (::std::size_t i = 0; i + 1 < replacements.size(); ++i)
                     current.merge(
                         target, replacements[i],
                         {::eggc::UnionKind::Rewrite, state->name, {}});
                   return replacements.back();
                 }});
          },
          stop);
      if (!completed) return false;
    }
    return !(stop && stop());
  });
}
template <class A, class M, class I, class F>
::eggc::Rewrite<OpNode, A> tensor_rewrite_checked_with_constraints(
    ::std::string name, TensorPattern lhs, TensorExpr rhs,
    TensorConstraints constraints, M metadata, I inference, F callback) {
  return tensor_rewrite_checked_with_checks<A>(
      ::std::move(name), ::std::move(lhs), ::std::move(rhs),
      MatchChecks<A>{::std::move(constraints), {}, {}}, ::std::move(metadata),
      ::std::move(inference), ::std::move(callback));
}
template <class A, class M, class I, class F>
::eggc::Rewrite<OpNode, A> tensor_rewrite_checked(::std::string name,
                                                  TensorPattern lhs,
                                                  TensorExpr rhs, M metadata,
                                                  I inference, F callback) {
  return tensor_rewrite_checked_with_constraints<A>(
      ::std::move(name), ::std::move(lhs), ::std::move(rhs), {},
      ::std::move(metadata), ::std::move(inference),
      [callback = ::std::move(callback)](const auto& graph, const auto& matched,
                                         const auto&) {
        return callback(graph, matched);
      });
}
}  // namespace @TEPL_NAMESPACE@::pattern
