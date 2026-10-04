#pragma once

#include <eggc/language.hpp>
#include <stdexcept>

#include "types.h"
// @tepl:dialect-includes

namespace @TEPL_NAMESPACE@ {
class NodeError : public ::std::invalid_argument {
 public:
  using ::std::invalid_argument::invalid_argument;
};
struct LiteralAttrs {
  ::std::string value;
  ::std::optional<DType> dtype;
  bool operator==(const LiteralAttrs&) const = default;
  ::std::size_t hash_value() const {
    auto seed = detail::hash_value(value);
    detail::hash_combine(seed, detail::hash_value(dtype));
    return seed;
  }
};
struct InputAttrs {
  ::std::string name;
  bool operator==(const InputAttrs&) const = default;
  ::std::size_t hash_value() const { return detail::hash_value(name); }
};
// @tepl:op-types

class OpNode {
 public:
  using Discriminant = ::std::size_t;
  static OpNode from_parts(Op op, ::std::vector<::eggc::Id> children,
                           OpAttrs attrs) {
    if (!op.arity().accepts(children.size()))
      throw NodeError("invalid operand count for " + ::std::string(op.name()));
    if (!op.accepts_attrs(attrs))
      throw NodeError("invalid attributes for " + ::std::string(op.name()));
    return OpNode(op, ::std::move(children), ::std::move(attrs));
  }
  template <class O>
    requires requires { typename DialectOp<O>::Attrs; }
  static OpNode make(O op, typename DialectOp<O>::Attrs attrs,
                     ::std::vector<::eggc::Id> children) {
    return from_parts(Op(op), ::std::move(children),
                      OpAttrs(::std::move(attrs)));
  }
  static OpNode input(::std::string name) {
    return from_parts(Op::Input, {}, InputAttrs{::std::move(name)});
  }
  static OpNode literal(::std::string value, DType dtype) {
    return from_parts(Op::Literal, {}, LiteralAttrs{::std::move(value), dtype});
  }
  Op op() const { return op_; }
  const OpAttrs& attrs() const { return attrs_; }
  Discriminant discriminant() const { return op_.id(); }
  const ::std::vector<::eggc::Id>& children() const { return children_; }
  ::std::vector<::eggc::Id>& children_mut() { return children_; }
  bool matches(const OpNode& other) const {
    return op_ == other.op_ && attrs_ == other.attrs_ &&
           children_.size() == other.children_.size();
  }
  bool operator==(const OpNode&) const = default;
  ::std::size_t hash() const {
    auto seed = op_.id();
    detail::hash_combine(seed, attrs_.hash_value());
    detail::hash_combine(seed, detail::hash_value(children_));
    return seed;
  }

 private:
  OpNode(Op op, ::std::vector<::eggc::Id> children, OpAttrs attrs)
      : op_(op), children_(::std::move(children)), attrs_(::std::move(attrs)) {}
  Op op_;
  ::std::vector<::eggc::Id> children_;
  OpAttrs attrs_;
};
static_assert(::eggc::Language<OpNode>);
}  // namespace @TEPL_NAMESPACE@
