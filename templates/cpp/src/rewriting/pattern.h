#pragma once

#include <map>
#include <set>

#include "../op_node.h"

namespace @TEPL_NAMESPACE@::rewriting {
struct NoFunctions {};
using BindingId = ::std::size_t;
struct AttrPattern {
  ::std::optional<BindingId> binding;
  OpAttrs exact;
  static AttrPattern bind(BindingId id) { return {id, {}}; }
  static AttrPattern value(OpAttrs attrs) { return {{}, ::std::move(attrs)}; }
};
struct TensorPattern {
  enum class Kind { Var, Bind, Op, Literal };
  Kind kind;
  BindingId capture = 0;
  ::@TEPL_NAMESPACE@::Op operation;
  AttrPattern attributes;
  ::std::vector<TensorPattern> children;
  LiteralAttrs literal_value;
  static TensorPattern var(BindingId id) {
    TensorPattern p{Kind::Var};
    p.capture = id;
    return p;
  }
  static TensorPattern bind(BindingId id, TensorPattern expression) {
    TensorPattern p{Kind::Bind};
    p.capture = id;
    p.children.push_back(::std::move(expression));
    return p;
  }
  static TensorPattern op(::@TEPL_NAMESPACE@::Op op, AttrPattern attrs,
                          ::std::vector<TensorPattern> children) {
    TensorPattern p{Kind::Op};
    p.operation = op;
    p.attributes = ::std::move(attrs);
    p.children = ::std::move(children);
    return p;
  }
  static TensorPattern literal(::std::string value,
                               ::std::optional<DType> dtype = {}) {
    TensorPattern p{Kind::Literal};
    p.literal_value = {::std::move(value), dtype};
    return p;
  }
  ::std::optional<::@TEPL_NAMESPACE@::Op> root_op() const {
    if (kind == Kind::Bind) return children.at(0).root_op();
    if (kind == Kind::Op) return operation;
    if (kind == Kind::Literal) return ::@TEPL_NAMESPACE@::Op::Literal;
    return {};
  }
  void bindings(::std::set<BindingId>& tensors,
                ::std::set<BindingId>& attrs) const {
    if (kind == Kind::Var || kind == Kind::Bind) tensors.insert(capture);
    if (kind == Kind::Op && attributes.binding)
      attrs.insert(*attributes.binding);
    for (const auto& child : children) child.bindings(tensors, attrs);
  }
  void validate() const {
    if (kind == Kind::Bind && children.size() != 1)
      throw ::std::invalid_argument("binder requires one expression");
    if ((kind == Kind::Var || kind == Kind::Literal) && !children.empty())
      throw ::std::invalid_argument("leaf pattern has children");
    if (kind == Kind::Op &&
        (!operation.arity().accepts(children.size()) ||
         (!attributes.binding && !operation.accepts_attrs(attributes.exact))))
      throw ::std::invalid_argument("invalid operation pattern");
    if (kind == Kind::Literal &&
        (!valid_literal(literal_value.value) ||
         (literal_value.dtype &&
          !accepts_literal(*literal_value.dtype, literal_value.value))))
      throw ::std::invalid_argument("invalid literal pattern");
    for (const auto& child : children) child.validate();
  }
};
struct AttrExpr {
  enum class Kind { Exact, Captured, Derived };
  Kind kind = Kind::Exact;
  BindingId binding = 0;
  OpAttrs value;
  static AttrExpr exact(OpAttrs value = {}) {
    return {Kind::Exact, 0, ::std::move(value)};
  }
  static AttrExpr captured(BindingId id) { return {Kind::Captured, id, {}}; }
  static AttrExpr derived(BindingId id) { return {Kind::Derived, id, {}}; }
};
struct TensorExpr {
  enum class Kind { Var, Op, Literal };
  Kind kind;
  BindingId capture = 0;
  ::@TEPL_NAMESPACE@::Op operation;
  AttrExpr attributes;
  ::std::vector<TensorExpr> children;
  LiteralAttrs literal_value;
  static TensorExpr var(BindingId id) {
    TensorExpr p{Kind::Var};
    p.capture = id;
    return p;
  }
  static TensorExpr op(::@TEPL_NAMESPACE@::Op op, AttrExpr attrs,
                       ::std::vector<TensorExpr> children) {
    TensorExpr p{Kind::Op};
    p.operation = op;
    p.attributes = ::std::move(attrs);
    p.children = ::std::move(children);
    return p;
  }
  static TensorExpr literal(::std::string value,
                            ::std::optional<DType> dtype = {}) {
    TensorExpr p{Kind::Literal};
    p.literal_value = {::std::move(value), dtype};
    return p;
  }
};
struct TensorMatch {
  ::eggc::Id root;
  ::std::map<BindingId, ::eggc::Id> tensors;
  ::std::map<BindingId, OpAttrs> attrs;
};
using DerivedAttrs = ::std::map<BindingId, OpAttrs>;
}  // namespace @TEPL_NAMESPACE@::rewriting
