#pragma once
#include <eggc/language.hpp>

#include "metadata.h"
namespace tepl_runtime {
template <class P>
concept Program = requires(
    typename P::Op op, const typename P::Attrs& attrs, const Value& value,
    const std::vector<TensorFacts>& operands, Budget& budget, std::size_t id) {
  requires std::copyable<typename P::Op>;
  requires std::equality_comparable<typename P::Op>;
  requires std::copyable<typename P::Attrs>;
  requires std::equality_comparable<typename P::Attrs>;
  { P::operation(id) } -> std::same_as<typename P::Op>;
  { P::operation_id(op) } -> std::same_as<std::size_t>;
  { P::operation_name(op) } -> std::same_as<std::string_view>;
  { P::arity(op) } -> std::same_as<Arity>;
  {
    P::accepts_attrs(op, std::optional<typename P::Attrs>(attrs))
  } -> std::same_as<bool>;
  {
    P::infer_shape(op, operands, value, budget)
  } -> std::same_as<EvalResult<std::optional<std::vector<std::uint64_t>>>>;
  {
    P::infer_dtype(op, operands, value, budget)
  } -> std::same_as<EvalResult<std::optional<DType>>>;
  { attrs.to_value() } -> std::same_as<Value>;
  { attrs.hash_value() } -> std::same_as<std::size_t>;
  {
    P::Attrs::from_value(value)
  } -> std::same_as<EvalResult<typename P::Attrs>>;
};
enum class NodeKind { Input, Literal, Operation };
template <Program P>
class Node {
  NodeKind kind_ = NodeKind::Input;
  std::string text_;
  std::optional<DType> dtype_;
  std::optional<typename P::Op> op_;
  std::optional<typename P::Attrs> attrs_;
  std::vector<eggc::Id> children_;

 public:
  using Discriminant = std::size_t;
  Node() = default;
  static Node input(std::string name) {
    Node out;
    out.text_ = std::move(name);
    return out;
  }
  static EvalResult<Node> literal(std::string text,
                                  std::optional<DType> dtype = {}) {
    return evaluate([&] {
      if (text.find('.') == text.npos)
        integer(text);
      else
        floating(text);
      require(!dtype || accepts_literal(*dtype, text),
              "literal incompatible with dtype");
      Node out;
      out.kind_ = NodeKind::Literal;
      out.text_ = std::move(text);
      out.dtype_ = dtype;
      return out;
    });
  }
  static EvalResult<Node> operation(typename P::Op op,
                                    std::optional<typename P::Attrs> attrs,
                                    std::vector<eggc::Id> children) {
    return evaluate([&] {
      require(P::arity(op).accepts(children.size()),
              "operation arity mismatch");
      require(P::accepts_attrs(op, attrs),
              "operation attribute schema mismatch");
      Node out;
      out.kind_ = NodeKind::Operation;
      out.op_ = op;
      out.attrs_ = std::move(attrs);
      out.children_ = std::move(children);
      return out;
    });
  }
  NodeKind kind() const { return kind_; }
  const std::string& text() const { return text_; }
  std::optional<DType> dtype() const { return dtype_; }
  typename P::Op op() const {
    require(op_.has_value(), "expected operation");
    return *op_;
  }
  const auto& attrs() const { return attrs_; }
  Value attribute_value() const {
    return attrs_ ? attrs_->to_value() : Value();
  }
  const std::vector<eggc::Id>& children() const { return children_; }
  std::vector<eggc::Id>& children_mut() { return children_; }
  Discriminant discriminant() const {
    return kind_ == NodeKind::Input     ? 0
           : kind_ == NodeKind::Literal ? 1
                                        : P::operation_id(*op_) + 2;
  }
  bool matches(const Node& other) const {
    return kind_ == other.kind_ && text_ == other.text_ &&
           dtype_ == other.dtype_ && op_ == other.op_ &&
           attrs_ == other.attrs_ && children_.size() == other.children_.size();
  }
  bool operator==(const Node&) const = default;
  std::size_t hash() const {
    auto seed = discriminant();
    detail::hash_combine(seed, detail::hash_value(text_));
    detail::hash_combine(seed, detail::hash_value(dtype_));
    if (attrs_) detail::hash_combine(seed, attrs_->hash_value());
    detail::hash_combine(seed, detail::hash_value(children_));
    return seed;
  }
};
}  // namespace tepl_runtime
