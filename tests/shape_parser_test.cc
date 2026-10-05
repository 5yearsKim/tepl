#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

#include "src/ast/ast.h"
#include "src/ast/print.h"
#include "src/parse.h"

namespace {
namespace ast = tepl::ast;

void check(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename T>
const T& as(const ast::MetadataExprPtr& expression) {
  check(expression != nullptr, "Required expression must be present");
  const auto* value = std::get_if<T>(&expression->value);
  check(value != nullptr, "Unexpected shape expression kind");
  return *value;
}

ast::MetadataDefinition shape(std::string_view body) {
  const auto parsed =
      tepl::parse("dialect D { op test(inputs: tensor...) -> tensor { " +
                      std::string(body) + " } }",
                  "shape.tepl");
  for (const auto& diagnostic : parsed.diagnostics)
    std::cerr << diagnostic.line << ':' << diagnostic.column << ' '
              << diagnostic.message << '\n';
  check(parsed.ok() && parsed.program.has_value(), "Shape fixture must parse");
  const auto& definition =
      parsed.program->dialects[0].operations[0].shape_definition;
  check(definition.has_value(), "Operation must retain its shape definition");
  return *definition;
}

void testBuiltinCallSyntax() {
  // Names and types are intentionally unresolved until core checking.
  const std::pair<std::string_view, std::string_view> cases[] = {
      {"len", "len(s)"},
      {"range", "range(3)"},
      {"concat", "concat(s, s, [])"},
      {"gather", "gather(s, [0])"},
      {"exclude", "exclude(s, [0])"},
      {"slice", "slice(s, 0, 2)"},
      {"replace", "replace(s, 0, 5)"},
      {"sum", "sum(s)"},
      {"product", "product(s)"},
      {"all", "all([true, false])"},
      {"any", "any([false, true])"},
      {"contains", "contains(s, 1)"},
      {"is_valid_axis_list", "is_valid_axis_list([0], len(s))"},
      {"is_disjoint", "is_disjoint([0], [1])"},
      {"broadcast_shape", "broadcast_shape(s, [1])"},
      {"min", "min(-1, 3)"},
      {"max", "max(0, 3)"},
      {"floor_div", "floor_div(-7, 3)"},
      {"ceil_div", "ceil_div(7, 3)"},
  };
  for (auto [name, expression] : cases) {
    const auto definition =
        shape("shape(s) { yield " + std::string(expression) + "; }");
    check(as<ast::MetadataCall>(definition.result.value).callee == name,
          "Builtin call name must survive AST construction");
  }
  const auto unresolved =
      shape("shape(s) { assert future_builtin(); yield unknown(s, true); }");
  check(as<ast::MetadataCall>(unresolved.result.value).callee == "unknown",
        "Parser must leave builtin name checking to core");
  const auto& assertion =
      std::get<ast::MetadataAssert>(unresolved.statements[0].value);
  check(as<ast::MetadataCall>(assertion.condition).arguments.empty(),
        "Zero-argument native calls must parse");
}

void testStatementsParametersAndSpans() {
  const auto parsed = tepl::parse(R"(dialect D {
  op transpose(input: tensor) -> tensor {
    attrs { permutation: index[]; }
    shape(s) {
      let axes = attrs.permutation;
      assert len(axes) == len(s);
      yield gather(s, axes);
    }
  }
})",
                                  "transpose.tepl");
  check(parsed.ok() && parsed.program, "Transpose must parse");
  const auto& definition =
      *parsed.program->dialects[0].operations[0].shape_definition;
  check(definition.parameters.size() == 1 &&
            definition.parameters[0].name == "s" &&
            !definition.parameters[0].variadic &&
            definition.statements.size() == 2,
        "Parameters and statement order must be preserved");
  const auto& binding =
      std::get<ast::MetadataLet>(definition.statements[0].value);
  check(binding.name == "axes", "Let name must be preserved");
  const auto& field = as<ast::MetadataField>(binding.value);
  check(field.field == "permutation" &&
            std::holds_alternative<ast::MetadataAttrs>(field.value->value),
        "attrs field access must be structured");
  const auto& assertion =
      std::get<ast::MetadataAssert>(definition.statements[1].value);
  check(
      as<ast::MetadataBinary>(assertion.condition).op == ast::BinaryOp::kEqual,
      "Assert must retain its Boolean expression");
  const auto& result = as<ast::MetadataCall>(definition.result.value);
  check(result.callee == "gather" && result.arguments.size() == 2,
        "Yield must retain call arguments");
  check(definition.span.begin.line == 4 && definition.span.begin.column == 5 &&
            definition.span.end.line == 8 && definition.span.end.column == 6,
        "Shape definition span must cover the complete block");
  check(
      definition.parameters[0].span.begin.column == 11 &&
          definition.statements[0].span.begin.line == 5 &&
          binding.name_span.begin.column == 11 &&
          definition.result.span.begin.line == 7 &&
          definition.result.value->span.begin.column == 13,
      "Parameters, names, statements and yield expressions need source spans");
  const auto printed = tepl::formatAst(*parsed.program);
  for (const auto text : {"shape(s)", "let axes", "assert", "field permutation",
                          "yield", "call gather"})
    check(printed.find(text) != std::string::npos,
          "AST output must expose shape syntax");

  const auto variadic =
      shape("shape(a, b, rest...) { yield concat([a, b], rest); }");
  check(variadic.parameters.size() == 3 && !variadic.parameters[0].variadic &&
            !variadic.parameters[1].variadic &&
            variadic.parameters[2].variadic &&
            variadic.parameters[2].name == "rest",
        "Fixed and trailing variadic shape parameters must remain distinct");
  check(shape("shape(inputs...) { yield inputs[0]; }").parameters[0].variadic,
        "A variadic-only parameter must parse");
  const auto scalar = shape("shape() { yield []; }");
  check(scalar.parameters.empty() && scalar.statements.empty() &&
            as<ast::MetadataList>(scalar.result.value).elements.empty(),
        "A zero-operand scalar shape must parse");
}

void testPrecedenceAndPostfix() {
  const auto definition =
      shape("shape(s) { yield -s[0] + 2 * s[1] >= 3 && !false || true; }");
  const auto& disjunction = as<ast::MetadataBinary>(definition.result.value);
  check(disjunction.op == ast::BinaryOp::kLogicalOr, "OR must be outermost");
  const auto& conjunction = as<ast::MetadataBinary>(disjunction.lhs);
  check(conjunction.op == ast::BinaryOp::kLogicalAnd,
        "AND must bind more tightly than OR");
  const auto& comparison = as<ast::MetadataBinary>(conjunction.lhs);
  check(comparison.op == ast::BinaryOp::kGreaterEqual,
        "Comparison must bind more tightly than AND");
  const auto& sum = as<ast::MetadataBinary>(comparison.lhs);
  check(sum.op == ast::BinaryOp::kAdd &&
            as<ast::MetadataBinary>(sum.rhs).op == ast::BinaryOp::kMultiply &&
            as<ast::MetadataUnary>(sum.lhs).op == ast::UnaryOp::kNegate &&
            as<ast::MetadataUnary>(conjunction.rhs).op ==
                ast::UnaryOp::kLogicalNot,
        "Multiplication, postfix indexing and unary operations must keep "
        "precedence");
  check(as<ast::MetadataIndex>(as<ast::MetadataUnary>(sum.lhs).operand).value !=
            nullptr,
        "Indexing must bind more tightly than unary minus");
  const auto arithmetic =
      shape("shape(s) { yield 10 - 4 - 1 + +s[0] / 2 % 3; }");
  const auto& outer = as<ast::MetadataBinary>(arithmetic.result.value);
  check(
      as<ast::MetadataBinary>(outer.lhs).op == ast::BinaryOp::kSubtract &&
          as<ast::MetadataBinary>(as<ast::MetadataBinary>(outer.lhs).lhs).op ==
              ast::BinaryOp::kSubtract &&
          as<ast::MetadataBinary>(outer.rhs).op == ast::BinaryOp::kRemainder &&
          as<ast::MetadataBinary>(as<ast::MetadataBinary>(outer.rhs).lhs).op ==
              ast::BinaryOp::kDivide,
      "Arithmetic must associate left and retain division/remainder");
  const auto indexed = shape("shape(s) { yield (gather(s, attrs.axes))[0]; }");
  check(
      as<ast::MetadataCall>(as<ast::MetadataIndex>(indexed.result.value).value)
              .callee == "gather",
      "Parenthesized call results must support indexing");
}

void testListsConditionalsAndComprehensions() {
  const auto definition = shape(R"(shape(s) {
    let pairs = [[0, 1], [2, 3]];
    assert all([len(row) > 0 for row in pairs]);
    yield [if contains(attrs.axes, i) then -s[i] else s[i] * 2 + 1
           for i in range(len(s))];
  })");
  const auto& binding =
      std::get<ast::MetadataLet>(definition.statements[0].value);
  const auto& pairs = as<ast::MetadataList>(binding.value);
  check(pairs.elements.size() == 2 &&
            as<ast::MetadataList>(pairs.elements[1]).elements.size() == 2,
        "Nested list literals must retain their structure");
  const auto& comprehension =
      as<ast::MetadataComprehension>(definition.result.value);
  check(comprehension.variable == "i" &&
            as<ast::MetadataCall>(comprehension.iterable).callee == "range",
        "Comprehension binder and iterable must be retained");
  const auto& conditional = as<ast::MetadataConditional>(comprehension.element);
  check(as<ast::MetadataCall>(conditional.condition).callee == "contains" &&
            as<ast::MetadataUnary>(conditional.then_value).op ==
                ast::UnaryOp::kNegate &&
            as<ast::MetadataBinary>(conditional.else_value).op ==
                ast::BinaryOp::kAdd,
        "Conditional branches must preserve expression structure");
  const auto nested = shape(
      "shape(s) { yield [[s[i][j] for j in range(len(s[i]))] for i in "
      "range(len(s))]; }");
  const auto& inner = as<ast::MetadataComprehension>(
      as<ast::MetadataComprehension>(nested.result.value).element);
  check(inner.variable == "j" &&
            std::holds_alternative<ast::MetadataIndex>(
                as<ast::MetadataIndex>(inner.element).value->value),
        "Nested comprehensions and repeated indexing must parse");
  const auto branches = shape(
      "shape(s) { yield if true then [] else if false then [1] else s; }");
  check(std::holds_alternative<ast::MetadataConditional>(
            as<ast::MetadataConditional>(branches.result.value)
                .else_value->value),
        "Conditional else branches must nest correctly");
}

void testInvalidShapeSyntax() {
  const std::string_view cases[] = {
      "shape(s) {}",
      "shape(s) { assert true; }",
      "shape(s) { yield s; yield s; }",
      "shape(s) { yield s; let x = s; }",
      "shape(s) { let x = s yield x; }",
      "shape(s) { assert true yield s; }",
      "shape(s) { yield s }",
      "shape(s,) { yield s; }",
      "shape(a..., b) { yield a; }",
      "shape(a..., b...) { yield a; }",
      "shape(...) { yield []; }",
      "shape(s) { yield $f(s); }",
      "shape(s) { yield gather(s, $axes()); }",
      "shape(s) { yield @d; }",
      "shape(s) { yield 1.5; }",
      "shape(s) { yield [s,]; }",
      "shape(s) { yield [s s]; }",
      "shape(s) { yield s[]; }",
      "shape(s) { yield gather(s,); }",
      "shape(s) { yield attrs.; }",
      "shape(s) { yield if true then s; }",
      "shape(s) { yield [s for i range(3)]; }",
      "shape(s) { assert 0 < 1 < 2; yield s; }",
      "shape(s) { assert s == [] == s; yield s; }",
      "shape(s) { s = []; yield s; }",
      "shape(s) { { yield s; } }",
      "shape(s) { yield s...; }",
      "shape(s) { yield s; } shape(s) { yield s; }",
  };
  for (const auto body : cases) {
    const auto parsed =
        tepl::parse("dialect D { op test(x: tensor) -> tensor { " +
                    std::string(body) + " } }");
    check(!parsed.ok() && !parsed.program && parsed.tree.empty(),
          "Malformed shape syntax must fail without returning a recovered AST");
  }
  const auto diagnostic = tepl::parse(
      "dialect D {\n op x() -> tensor {\n  shape() {\n   yield $f();\n  }\n "
      "}\n}");
  check(!diagnostic.ok() && diagnostic.diagnostics[0].line == 4 &&
            diagnostic.diagnostics[0].column == 10,
        "Shape syntax diagnostics must identify the offending token");
}

void testBuiltinNamesRemainIdentifiers() {
  check(tepl::parse("dialect D { op len(shape: tensor) -> tensor { attrs { "
                    "shape: index[]; } shape(s) { yield attrs.shape; } } } "
                    "rule r { len => len where { gather(len); $len(len); } }")
            .ok(),
        "Builtin names and shape must remain usable as ordinary identifiers");
}
}  // namespace

int main() {
  try {
    testBuiltinCallSyntax();
    testStatementsParametersAndSpans();
    testPrecedenceAndPostfix();
    testListsConditionalsAndComprehensions();
    testInvalidShapeSyntax();
    testBuiltinNamesRemainIdentifiers();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
