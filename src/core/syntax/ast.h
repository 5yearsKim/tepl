#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "src/shared/source/source.h"
#include "src/shared/types/operators.h"

namespace tepl::syntax {

using ::tepl::SourcePosition;
using ::tepl::SourceSpan;

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

struct DTypeAnnotation {
  SourceSpan span;
  std::string name;
};

struct Declaration {
  SourceSpan span;
  std::string name;
  // An empty shape represents a rank-zero tensor.
  std::vector<ShapeDimension> shape;
  std::optional<DTypeAnnotation> dtype;
};

// References own their names. Attribute sigils ('@') are omitted from names.
// A plain name may refer to an input or a value introduced by `let`;
// symbol resolution happens after AST construction.
struct NameRef {
  std::string name;
};

struct IntegerLiteral {
  // Preserve decimal spelling until numeric range validation. Graph literals
  // may include a sign; constraint signs are represented by UnaryExpr.
  std::string digits;
  // Set only for graph literals; constraint numbers are scalar values.
  std::optional<DTypeAnnotation> dtype;
};

struct FloatLiteral {
  // Preserve spelling without rounding or choosing a tensor element type.
  // Graph literals may include a sign, as with IntegerLiteral.
  std::string digits;
  // Set only for graph literals; constraint numbers are scalar values.
  std::optional<DTypeAnnotation> dtype;
};

struct BooleanLiteral {
  bool value = false;
};

struct GraphExpr;
struct ConstraintExpr;

// Copyable handles can be returned through ANTLR's std::any visitor interface.
// Completed expressions have non-null required children. Binders refer to
// names, so repeated references do not create pointer cycles in the syntax
// tree.
using GraphExprPtr = std::shared_ptr<const GraphExpr>;
using ConstraintExprPtr = std::shared_ptr<const ConstraintExpr>;

struct Operator {
  std::string name;
  std::vector<GraphExprPtr> operands;
  std::optional<std::string> label;
  std::optional<std::string> descriptor_node;
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
  std::variant<NameRef, IntegerLiteral, FloatLiteral, Operator, Binding,
               Projection>
      value;
  std::size_t resolution_id = 0;
};

using ::tepl::BinaryOp;
using ::tepl::UnaryOp;

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

struct NodeProperty {
  std::string node;
  std::vector<std::string> path;
};

struct StringLiteral {
  std::string value;
};
struct ListExpr {
  std::vector<ConstraintExprPtr> elements;
};
struct IndexExpr {
  ConstraintExprPtr value, index;
};
struct ConditionalExpr {
  ConstraintExprPtr condition, then_value, else_value;
};
struct ComprehensionExpr {
  ConstraintExprPtr element;
  std::string variable;
  ConstraintExprPtr iterable;
};
struct DescriptorConstructor {
  std::optional<std::string> spread;
  struct Field {
    SourceSpan span;
    std::string name;
    ConstraintExprPtr value;
  };
  std::vector<Field> fields;
};
// Statement-only node, never accepted as a nested expression.
struct LocalBinding {
  std::string name;
  ConstraintExprPtr value;
};

struct ConstraintExpr {
  SourceSpan span;
  std::variant<NameRef, NodeProperty, IntegerLiteral, FloatLiteral,
               BooleanLiteral, Call, UnaryExpr, BinaryExpr, StringLiteral,
               ListExpr, IndexExpr, ConditionalExpr, ComprehensionExpr,
               DescriptorConstructor, LocalBinding>
      value;
  std::size_t resolution_id = 0;
};

// Shape programs have their own expression tree. Unlike rule constraints,
// they cannot contain tensor captures, descriptor references, or rule-local
// bindings.
struct MetadataExpr;
using MetadataExprPtr = std::shared_ptr<const MetadataExpr>;

struct MetadataAttrs {};
struct MetadataCall {
  // Unresolved builtin name; the parser does not validate names or signatures.
  std::string callee;
  std::vector<MetadataExprPtr> arguments;
};
struct MetadataUnary {
  UnaryOp op;
  MetadataExprPtr operand;
};
struct MetadataBinary {
  BinaryOp op;
  MetadataExprPtr lhs;
  MetadataExprPtr rhs;
};
struct MetadataList {
  std::vector<MetadataExprPtr> elements;
};
struct MetadataIndex {
  MetadataExprPtr value;
  MetadataExprPtr index;
};
struct MetadataField {
  MetadataExprPtr value;
  std::string field;
};
struct MetadataConditional {
  MetadataExprPtr condition;
  MetadataExprPtr then_value;
  MetadataExprPtr else_value;
};
struct MetadataComprehension {
  MetadataExprPtr element;
  std::string variable;
  SourceSpan variable_span;
  MetadataExprPtr iterable;
};
struct MetadataExpr {
  SourceSpan span;
  std::variant<NameRef, IntegerLiteral, BooleanLiteral, MetadataAttrs,
               MetadataCall, MetadataUnary, MetadataBinary, MetadataList,
               MetadataIndex, MetadataField, MetadataConditional,
               MetadataComprehension>
      value;
};
struct MetadataParameter {
  SourceSpan span;
  std::string name;
  bool variadic = false;
};
struct MetadataLet {
  std::string name;
  SourceSpan name_span;
  MetadataExprPtr value;
};
struct MetadataAssert {
  MetadataExprPtr condition;
};
struct MetadataStatement {
  SourceSpan span;
  std::variant<MetadataLet, MetadataAssert> value;
};
struct MetadataYield {
  SourceSpan span;
  MetadataExprPtr value;
};
struct MetadataDefinition {
  SourceSpan span;
  std::vector<MetadataParameter> parameters;
  std::vector<MetadataStatement> statements;
  MetadataYield result;
};

struct AxisBinding {
  SourceSpan span;
  std::string operand;
  MetadataExprPtr positions;
};
struct AxisBind {
  SourceSpan span;
  std::string name;
  std::vector<AxisBinding> bindings;
  MetadataExprPtr domain;
};
struct AxisDefinition {
  SourceSpan span;
  std::vector<std::variant<MetadataStatement, AxisBind>> statements;
  MetadataExprPtr output;
};

// Semantics expressions own their scope and are separate from metadata
// programs.
struct SemanticExpr;
using SemanticExprPtr = std::shared_ptr<const SemanticExpr>;
struct SemanticExpr {
  enum class Kind { kName, kCall, kInteger, kAttribute };
  SourceSpan span;
  Kind kind;
  std::string text;
  std::vector<SemanticExprPtr> arguments;
};
struct SemanticDefinition {
  SourceSpan span;
  SemanticExprPtr result;
};

struct Derivation {
  SourceSpan span;
  std::string target;
  ConstraintExprPtr value;
  bool local = false;
};

struct SignatureType {
  SourceSpan span;
  std::string name;
};

struct RuleParameter {
  SourceSpan span;
  std::string name;
  std::vector<SignatureType> operand_types;
  SignatureType result_type;
};

struct RuleBinding {
  SourceSpan span;
  std::string parameter;
  // Unresolved operation name, possibly qualified.
  std::string value;
};

struct RuleInheritance {
  SourceSpan span;
  std::string base;
  std::vector<RuleBinding> bindings;
};

struct DTypeDeclaration {
  SourceSpan span;
  std::string name;
};

struct Rule {
  SourceSpan span;
  std::string name;
  std::vector<Declaration> declarations;
  // Inherited rules have no local graph; both pointers are null until
  // expansion.
  GraphExprPtr lhs;
  GraphExprPtr rhs;
  // Where bindings in source order, followed by its final Boolean expression.
  // An empty vector represents an absent where section.
  std::vector<ConstraintExprPtr> conditions;
  // An empty vector represents an absent or empty derive section.
  std::vector<Derivation> derivations;
  bool is_abstract = false;
  std::vector<RuleParameter> parameters;
  std::optional<RuleInheritance> inheritance;
  std::string source_name;
  std::vector<DTypeDeclaration> dtypes;
};

struct Import {
  SourceSpan span;
  std::string path;
  // Empty for the original `import "path";` form.
  std::optional<std::string> dialect;
  std::optional<std::string> alias;
  // Nonempty for `from "path" import {rule, ...};`. This form selects rules,
  // while dialect/alias describe the existing named dialect import form.
  std::vector<NameRef> rules;
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
  std::size_t list_depth = 0;
  bool optional = false;
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
  std::optional<MetadataDefinition> shape_definition;
  std::optional<MetadataDefinition> dtype_definition;
  std::vector<std::string> duplicate_properties;
  std::optional<SemanticDefinition> semantics;
  std::optional<AxisDefinition> axes;
};

struct Dialect {
  SourceSpan span;
  std::string name;
  std::vector<AttrSchema> schemas;
  std::vector<OpDecl> operations;
  std::string source_name;
};

// Concrete graph values own their data; bare names in attributes denote enums.
struct ConcreteValue {
  enum class Kind { kInteger, kDecimal, kBool, kString, kName, kList } kind;
  SourceSpan span;
  std::string text;
  std::vector<ConcreteValue> elements;
};
struct ConcreteField {
  SourceSpan span;
  std::string name;
  ConcreteValue value;
};
struct ConcreteExpr {
  enum class Kind { kReference, kLiteral, kOperation } kind;
  SourceSpan span;
  std::string name;
  bool decimal = false;
  std::optional<DTypeAnnotation> dtype;
  std::vector<ConcreteField> attributes;
  std::vector<ConcreteExpr> operands;
  std::size_t resolution_id = 0;
};
struct ConcreteInput {
  SourceSpan span;
  std::string name;
  std::optional<DTypeAnnotation> dtype;
  std::optional<std::vector<std::string>> shape;
};
struct ConcreteBinding {
  SourceSpan span;
  std::string name;
  ConcreteExpr expression;
};
struct ConcreteGraph {
  SourceSpan span;
  std::string name, source_name;
  std::vector<ConcreteInput> inputs;
  std::vector<ConcreteBinding> bindings;
  ConcreteExpr output;
};

struct ParsedModule {
  SourceSpan span;
  // Root rules, imports, uses, and local dialects use this source file.
  // Imported dialects carry their own source_name.
  std::string source_name;
  std::vector<Rule> rules;
  std::vector<Import> imports;
  std::vector<Dialect> dialects;
  std::vector<Use> uses;
  std::vector<ConcreteGraph> graphs;
};

}  // namespace tepl::syntax
