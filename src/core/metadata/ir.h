#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "src/core/builtins/catalog.h"
#include "src/core/ids.h"
#include "src/core/metadata/types.h"
#include "src/core/type/types.h"
#include "src/operators.h"
#include "src/source.h"

namespace tepl::core::metadata {

using TypeId = core::Id<struct MetadataTypeTag>;
using SymbolId = core::Id<struct MetadataSymbolTag>;
struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;

struct SymbolRef {
  SymbolId symbol;
};
struct Integer {
  std::string spelling;
};
struct AttributeRef {
  AttributeSchemaId schema;
  std::size_t field;
};
struct Call {
  builtins::Builtin builtin;
  std::vector<ExprPtr> arguments;
};
struct Unary {
  UnaryOp op;
  ExprPtr operand;
};
struct Binary {
  BinaryOp op;
  ExprPtr lhs;
  ExprPtr rhs;
};
struct List {
  std::vector<ExprPtr> elements;
};
struct Index {
  ExprPtr value;
  ExprPtr index;
};
struct Conditional {
  ExprPtr condition;
  ExprPtr then_value;
  ExprPtr else_value;
};
struct Comprehension {
  ExprPtr element;
  SymbolId variable;
  ExprPtr iterable;
};
struct Expr {
  SourceOrigin origin;
  TypeId type;
  std::variant<SymbolRef, Integer, bool, core::DType, AttributeRef, Call, Unary,
               Binary, List, Index, Conditional, Comprehension>
      value;
};
struct Symbol {
  SymbolId id;
  std::string name;
  TypeId type;
  SourceOrigin origin;
};
struct Parameter {
  SymbolId symbol;
  std::size_t operand;
  bool variadic;
};
struct Let {
  SymbolId symbol;
  ExprPtr value;
};
struct Assert {
  ExprPtr condition;
};
struct Statement {
  SourceOrigin origin;
  std::variant<Let, Assert> value;
};
struct Yield {
  SourceOrigin origin;
  ExprPtr value;
};
enum class ProgramKind { Shape, DType };

struct Program {
  SourceOrigin origin;
  // IDs are local to this operation's shape program. Every type is concrete
  // after checking; no AST nodes or unresolved builtin names survive.
  std::vector<Type> types;
  std::vector<Symbol> symbols;
  std::vector<Parameter> parameters;
  std::vector<Statement> statements;
  Yield result;
  ProgramKind kind = ProgramKind::Shape;
};

}  // namespace tepl::core::metadata
