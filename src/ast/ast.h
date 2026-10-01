#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace tepl::ast {

// One-based lines and Unicode code-point columns, as in parser diagnostics.
struct SourcePosition {
  std::size_t line = 1;
  std::size_t column = 1;
};

// Half-open range: begin is included, end is excluded.
struct SourceSpan {
  SourcePosition begin;
  SourcePosition end;
};

struct NamedDimension {
  std::string name;
};

struct WildcardDimension {};

struct SequenceDimension {
  // Missing name represents anonymous "...".
  std::optional<std::string> name;
};

struct ShapeDimension {
  SourceSpan span;
  std::variant<NamedDimension, WildcardDimension, SequenceDimension> value;
};

struct TensorDecl {
  std::string name;
  // An empty shape represents a rank-zero tensor.
  std::vector<ShapeDimension> shape;
};

struct ScalarDecl {
  std::string name;
};

struct Declaration {
  SourceSpan span;
  std::variant<TensorDecl, ScalarDecl> value;
};

// References own their names. Attribute sigils ('@') are omitted from names.
// A plain name may refer to an input or a value introduced by `let`;
// symbol resolution happens after AST construction.
struct NameRef {
  std::string name;
};

struct AttributeRef {
  std::string name;
};

struct IntegerLiteral {
  // Preserve unsigned decimal spelling until numeric range validation.
  // Negative values are represented by UnaryExpr.
  std::string digits;
};

struct BooleanLiteral {
  bool value = false;
};

struct GraphExpr;
struct ConstraintExpr;

// Copyable handles can be returned through ANTLR's std::any visitor interface.
// Completed ASTs have non-null required children. Binders refer to names, so
// repeated references do not create pointer cycles in the syntax tree.
using GraphExprPtr = std::shared_ptr<GraphExpr>;
using ConstraintExprPtr = std::shared_ptr<ConstraintExpr>;

struct Operator {
  std::string name;
  std::optional<AttributeRef> attribute;
  std::vector<GraphExprPtr> operands;
};

struct Binding {
  NameRef binder;
  GraphExprPtr expression;
};

struct Projection {
  IntegerLiteral index;
  GraphExprPtr tuple;
};

struct GraphExpr {
  SourceSpan span;
  // Tuple construction uses Operator with name "tuple".
  std::variant<NameRef, Operator, Binding, Projection> value;
};

// makeConstraint expressions used in `where` conditions and `derive` values.
// They combine references, literals, host calls, and unary/binary operations:
//   !broadcastable(A, B), K % 128 == 0, infer_dot(X, W, @outer).
// Tree nesting preserves operator precedence. Semantic validation determines
// result types and requires each complete `where` condition to be boolean.
enum class UnaryOp {
  kPlus,
  kNegate,
  kLogicalNot,
};

enum class BinaryOp {
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  kRemainder,
  kLess,
  kLessEqual,
  kGreater,
  kGreaterEqual,
  kEqual,
  kNotEqual,
  kLogicalAnd,
  kLogicalOr,
};

struct Call {
  std::string callee;
  std::vector<ConstraintExprPtr> arguments;
};

struct UnaryExpr {
  UnaryOp op = UnaryOp::kPlus;
  ConstraintExprPtr operand;
};

struct BinaryExpr {
  BinaryOp op = BinaryOp::kAdd;
  ConstraintExprPtr lhs;
  ConstraintExprPtr rhs;
};

struct ConstraintExpr {
  SourceSpan span;
  std::variant<NameRef, AttributeRef, IntegerLiteral, BooleanLiteral, Call,
               UnaryExpr, BinaryExpr>
      value;
};

struct Derivation {
  SourceSpan span;
  AttributeRef target;
  ConstraintExprPtr value;
};

struct Rule {
  SourceSpan span;
  std::string name;
  std::vector<Declaration> declarations;
  GraphExprPtr lhs;
  GraphExprPtr rhs;
  // Empty vectors represent absent or empty where/derive sections.
  std::vector<ConstraintExprPtr> conditions;
  std::vector<Derivation> derivations;
};

struct Import {
  SourceSpan span;
  std::string path;
  // Empty for the original `import "path";` form.
  std::optional<std::string> dialect;
  std::optional<std::string> alias;
};

struct Use {
  SourceSpan span;
  std::string alias;
  // Empty means `use alias;`, which opens every operation in the dialect.
  std::vector<std::string> operations;
};

struct AttrField {
  SourceSpan span;
  std::string name;
  std::string type;
  bool list = false;
  bool empty_default = false;
};

struct AttrSchema {
  SourceSpan span;
  std::string name;
  std::vector<AttrField> fields;
};

struct OperandDecl {
  SourceSpan span;
  std::string name;
  std::string type;
  bool variadic = false;
};

struct SharedAttrs {
  std::string name;
};

struct InlineAttrs {
  std::vector<AttrField> fields;
};

using OpAttrs = std::variant<SharedAttrs, InlineAttrs>;

struct OpDecl {
  SourceSpan span;
  // The declared operation name is canonical; alias is another spelling.
  std::string name;
  std::vector<OperandDecl> operands;
  std::string result_type;
  std::optional<std::string> alias;
  std::optional<OpAttrs> attrs;
};

struct Dialect {
  SourceSpan span;
  std::string name;
  std::vector<AttrSchema> schemas;
  std::vector<OpDecl> operations;
  std::string source_name;
};

struct Program {
  SourceSpan span;
  // Root rules, imports, uses, and local dialects use this source file.
  // Imported dialects carry their own source_name.
  std::string source_name;
  std::vector<Rule> rules;
  std::vector<Import> imports;
  std::vector<Dialect> dialects;
  std::vector<Use> uses;
};

}  // namespace tepl::ast
