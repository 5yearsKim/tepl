#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/ast/print.h"
#include "src/core/analyze.h"
#include "src/core/print.h"
#include "src/parse.h"

namespace {
namespace shape = tepl::core::shape;

void check(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

tepl::core::AnalysisResult analyze(std::string_view body,
                                   std::string_view signature = "x: tensor",
                                   std::string_view attrs = "") {
  auto parsed = tepl::parse("dialect D { op test(" + std::string(signature) +
                                ") -> tensor { " + std::string(attrs) + " " +
                                std::string(body) + " } }",
                            "shape.tepl");
  check(parsed.ok(), "Core fixture must have valid syntax");
  return tepl::core::analyze(*parsed.program);
}

const shape::Program& checkedShape(const tepl::core::AnalysisResult& result) {
  if (!result.ok()) {
    for (const auto& diagnostic : result.diagnostics)
      std::cerr << diagnostic.message << '\n';
    throw std::runtime_error("Expected successful shape checking");
  }
  const auto& program = *result.program;
  check(program.host_functions.empty(),
        "Shape builtins must not register host functions");
  check(program.operations[0].shape.has_value(),
        "Checked shape must be attached to the operation");
  return *program.operations[0].shape;
}

void expectError(std::string_view body, std::string_view message,
                 std::string_view signature = "x: tensor",
                 std::string_view attrs = "") {
  const auto result = analyze(body, signature, attrs);
  check(!result.ok() && !result.program,
        "Invalid shape program must not return checked IR");
  for (const auto& diagnostic : result.diagnostics) {
    check(diagnostic.origin.definition.source_name == "shape.tepl",
          "Shape diagnostic needs source identity");
    if (diagnostic.message.find(message) != std::string::npos) return;
  }
  for (const auto& diagnostic : result.diagnostics)
    std::cerr << diagnostic.message << '\n';
  throw std::runtime_error("Expected diagnostic: " + std::string(message));
}

void builtinSignatures() {
  // Each builtin is tested with correct types, wrong arity, and wrong types.
  struct Case {
    std::string_view name, valid, invalid;
    bool boolean;
  };
  const Case cases[] = {
      {"len", "len(s)", "len(1)", false},
      {"range", "range(3)", "range(s)", false},
      {"concat", "concat(s, [], s)", "concat(s, [true])", false},
      {"gather", "gather(s, [0])", "gather(s, [true])", false},
      {"exclude", "exclude(s, [])", "exclude(s, [true])", false},
      {"slice", "slice(s, 0, 2)", "slice(s, false, 2)", false},
      {"replace", "replace(s, 0, 5)", "replace(s, 0, true)", false},
      {"sum", "sum(s)", "sum([true])", false},
      {"product", "product(s)", "product([[1]])", false},
      {"all", "all([true, false])", "all([1])", true},
      {"any", "any([])", "any(s)", true},
      {"contains", "contains(s, 1)", "contains(s, true)", true},
      {"is_valid_axis_list", "is_valid_axis_list([0], len(s))",
       "is_valid_axis_list([true], 2)", true},
      {"is_disjoint", "is_disjoint([0], [1])", "is_disjoint([0], [false])",
       true},
      {"broadcast_shape", "broadcast_shape(s, [1])",
       "broadcast_shape(s, [[1]])", false},
      {"min", "min(-1, 3)", "min(s, 1)", false},
      {"max", "max(0, 3)", "max(0, false)", false},
      {"floor_div", "floor_div(-7, 3)", "floor_div(7, s)", false},
      {"ceil_div", "ceil_div(7, 3)", "ceil_div(true, 3)", false},
  };
  for (const auto& test : cases) {
    const std::string body =
        test.boolean
            ? "shape(s) { assert " + std::string(test.valid) + "; yield s; }"
            : "shape(s) { let result = " + std::string(test.valid) +
                  "; yield s; }";
    const auto result = analyze(body);
    const auto& program = checkedShape(result);
    const auto expression =
        test.boolean
            ? std::get<shape::Assert>(program.statements[0].value).condition
            : std::get<shape::Let>(program.statements[0].value).value;
    check(shape::builtinName(
              std::get<shape::Call>(expression->value).builtin) == test.name,
          "Calls must resolve to builtin IDs");
    expectError(
        "shape(s) { let result = " + std::string(test.invalid) + "; yield s; }",
        "type mismatch");
    expectError(
        "shape(s) { let result = " + std::string(test.name) + "(); yield s; }",
        "expects");
    if (test.name != "concat")
      expectError("shape(s) { let result = " + std::string(test.name) +
                      "(s, s, s, s); yield s; }",
                  "expects");
    else
      expectError("shape(s) { yield concat(s); }", "at least 2");
  }
}

void signaturesAndScopes() {
  expectError("shape() { yield []; }", "parameter count");
  expectError("shape(a, b) { yield a; }", "parameter count");
  expectError("shape(a...) { yield a[0]; }", "variadic parameter");
  expectError("shape(a) { yield a; }", "variadic parameter", "xs: tensor...");
  expectError("shape(a, a) { yield a; }", "duplicate shape binding",
              "x: tensor, y: tensor");
  expectError("shape(s) { let s = [1]; yield s; }", "duplicate shape binding");
  expectError("shape(s) { let x = [1]; let x = [2]; yield x; }",
              "duplicate shape binding");
  expectError("shape(s) { let x = x; yield s; }", "unknown shape name 'x'");
  expectError("shape(s) { assert x == 1; let x = 1; yield s; }",
              "unknown shape name 'x'");
  expectError("shape(s) { let xs = [i for i in range(3)]; yield [i]; }",
              "unknown shape name 'i'");
  expectError("shape(s) { yield [i for i in i]; }", "unknown shape name 'i'");
  expectError("shape(s) { yield unknown(s); }", "unknown shape builtin");
  expectError("shape(s) { yield x; }", "unknown shape name 'x'");
  const auto result = analyze(
      "shape(s) { let i = 1; let xs = [i + 1 for i in range(3)]; assert i == "
      "1; yield xs; }");
  const auto& program = checkedShape(result);
  const auto outer = std::get<shape::Let>(program.statements[0].value).symbol;
  const auto xs = std::get<shape::Let>(program.statements[1].value).value;
  const auto& comprehension = std::get<shape::Comprehension>(xs->value);
  check(comprehension.variable != outer,
        "Comprehension shadowing needs a distinct ID");
  const auto& assertion = std::get<shape::Binary>(
      std::get<shape::Assert>(program.statements[2].value).condition->value);
  check(std::get<shape::SymbolRef>(assertion.lhs->value).symbol == outer,
        "Outer binding must be restored after comprehension");
  const auto variadic = analyze(
      "shape(a, rest...) { yield concat(a, sum([len(s) for s in rest]) * [1]); "
      "}",
      "x: tensor, xs: tensor...");
  check(!variadic.ok(),
        "List arithmetic must not become implicit elementwise arithmetic");
  const auto scalar = analyze("shape() { yield []; }", "");
  check(checkedShape(scalar).parameters.empty(),
        "Zero-operand shape signatures must work");
}

void listsAndAttributes() {
  // Generic list helpers retain their element type, even for Boolean/nested
  // lists.
  const auto generic = analyze(R"(shape(s) {
    let rows = concat([s], [s], []);
    let selected = gather(rows, [0]);
    let tail = slice(selected, 0, 1);
    let replaced = replace(tail, 0, s);
    let remaining = exclude(replaced, [s]);
    assert contains(replaced, s);
    assert is_disjoint(remaining, [s]);
    let flags = concat([true], [false]);
    assert all(gather(flags, [0]));
    assert contains(flags, false);
    assert is_disjoint([true], [false]);
    yield replaced[0];
  })");
  checkedShape(generic);
  const auto result = analyze(R"(shape(a, rest...) {
    let empty = [];
    let shapes = concat([a], rest, empty);
    let flags = [];
    assert all(flags);
    let padding = attrs.padding;
    let ys = [[s[i] + padding[i][0] for i in range(len(s))] for s in shapes];
    yield if attrs.enabled then ys[0] else [];
  })",
                              "x: tensor, xs: tensor...",
                              "attrs { padding: i64[][]; enabled: bool; }");
  const auto& program = checkedShape(result);
  check(program.types.at(program.symbols[0].type.value) ==
            shape::Type{shape::Type::Kind::kInteger, 1},
        "Fixed parameters must be shapes");
  check(program.types.at(program.symbols[1].type.value) ==
            shape::Type{shape::Type::Kind::kInteger, 2},
        "Variadic parameters must be lists of shapes");
  check(program.types.at(program.symbols[2].type.value).list_depth == 2,
        "Nested empty list typing must come from its uses");
  const auto& field = std::get<shape::AttributeRef>(
      std::get<shape::Let>(program.statements[4].value).value->value);
  check(field.schema.value == 0 && field.field == 0,
        "Attribute field must resolve to schema and field indexes");
  check(program.types.at(program.result.value->type.value) ==
            shape::Type{shape::Type::Kind::kInteger, 1},
        "Yield type must be a shape");
  expectError("shape(s) { let xs = []; yield s; }", "cannot infer");
  expectError("shape(s) { let n = len([]); yield s; }", "cannot infer");
  expectError("shape(s) { yield [true]; }", "type mismatch");
  expectError("shape(s) { assert s; yield s; }", "type mismatch");
  expectError("shape(s) { yield [1, true]; }", "type mismatch");
  expectError("shape(s) { yield if true then s else false; }", "type mismatch");
  expectError("shape(s) { yield if 1 then s else []; }", "type mismatch");
  expectError("shape(s) { yield [1[0]]; }", "type mismatch");
  expectError("shape(s) { yield [s[true]]; }", "type mismatch");
  expectError("shape(s) { yield [i for i in 3]; }", "type mismatch");
  expectError("shape(s) { let xs = []; assert xs == [xs]; yield s; }",
              "contain itself");
  expectError("shape(s) { yield attrs.shape; }", "no attribute schema");
  expectError("shape(s) { yield attrs.missing; }", "unknown shape attribute",
              "x: tensor", "attrs { shape: index[]; }");
  expectError("shape(s) { yield attrs; }", "requires a field access");
  expectError("shape(s) { yield s.field; }", "requires attrs.field");
  for (const auto type : {"string", "precision", "elements", "region",
                          "dot_algorithm", "replica_groups", "index?"})
    expectError("shape(s) { let x = attrs.value; yield s; }",
                "optional or opaque", "x: tensor",
                "attrs { value: " + std::string(type) + "; }");
  const auto shared = tepl::parse(
      "dialect D { attrs A { shape: index[]; } op x(input: tensor) -> tensor { "
      "attrs: A; shape(s) { yield attrs.shape; } } }");
  check(shared.ok() && tepl::core::analyze(*shared.program).ok(),
        "Shared attribute schemas must work");
}

void integerPolicyAndRuntimeChecks() {
  for (const auto literal :
       {"18446744073709551615", "170141183460469231731687303715884105727",
        "-170141183460469231731687303715884105728", "+0001"}) {
    const auto result =
        analyze("shape(s) { yield [" + std::string(literal) + "]; }");
    checkedShape(result);
  }
  for (const auto literal : {"170141183460469231731687303715884105728",
                             "-170141183460469231731687303715884105729"})
    expectError("shape(s) { yield [" + std::string(literal) + "]; }",
                "128-bit range");
  // Value validity remains an execution-time responsibility; assertions and
  // short-circuit branches must survive core checking without being executed.
  const auto deferred = analyze(
      "shape(s) { assert false && s[999] > 0; yield if true then [] else [1 / "
      "0, -1]; }");
  const auto& program = checkedShape(deferred);
  check(std::holds_alternative<shape::Binary>(
            std::get<shape::Assert>(program.statements[0].value)
                .condition->value) &&
            std::holds_alternative<shape::Conditional>(
                program.result.value->value),
        "Runtime control flow must be preserved");
}

void diagnosticsAndPrinting() {
  auto parsed = tepl::parse(
      "dialect D {\n op x(input: tensor) -> tensor {\n  shape(s) {\n   yield "
      "missing;\n  }\n }\n}",
      "locations.tepl");
  check(parsed.ok(), "Diagnostic fixture must parse");
  const auto before = tepl::formatAst(*parsed.program);
  const auto invalid = tepl::core::analyze(*parsed.program);
  check(!invalid.ok() && invalid.diagnostics.size() == 1,
        "Unknown shape name should have one diagnostic");
  const auto& origin = invalid.diagnostics[0].origin.definition;
  check(origin.source_name == "locations.tepl" && origin.span.begin.line == 4 &&
            origin.span.begin.column == 10,
        "Diagnostic should point to the bad name");
  check(tepl::formatAst(*parsed.program) == before,
        "Shape checking must not mutate the AST");
  const auto result = analyze(
      "shape(s) { let axes = range(len(s)); assert all([i >= 0 for i in "
      "axes]); yield gather(s, axes); }");
  checkedShape(result);
  const auto text = tepl::core::formatProgram(*result.program);
  for (const auto fragment : {"parameter #", "List<Integer>", "Builtin(gather",
                              "Comprehension", "yield"})
    check(text.find(fragment) != std::string::npos,
          "Checked printing must expose resolved shape programs");
  const auto absent = analyze("");
  check(absent.ok() && !absent.program->operations[0].shape,
        "Missing shape definition must remain absent");
}

void checkExpression(const shape::ExprPtr& expression,
                     const shape::Program& shape,
                     const tepl::core::Program& program) {
  check(expression != nullptr && expression->type.value < shape.types.size(),
        "Every checked expression must have a resolved type");
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, shape::SymbolRef>) {
          check(
              value.symbol.value < shape.symbols.size() &&
                  shape.types[expression->type.value] ==
                      shape.types[shape.symbols[value.symbol.value].type.value],
              "Symbol refs must have consistent types");
        } else if constexpr (std::is_same_v<T, shape::AttributeRef>) {
          check(value.schema.value < program.attribute_schemas.size() &&
                    value.field < program.attribute_schemas[value.schema.value]
                                      .fields.size(),
                "Attributes must resolve");
        } else if constexpr (std::is_same_v<T, shape::Call>) {
          check(shape::resolveBuiltin(shape::builtinName(value.builtin)) !=
                    nullptr,
                "Builtins must resolve");
          for (const auto& argument : value.arguments)
            checkExpression(argument, shape, program);
        } else if constexpr (std::is_same_v<T, shape::Unary>) {
          checkExpression(value.operand, shape, program);
        } else if constexpr (std::is_same_v<T, shape::Binary>) {
          checkExpression(value.lhs, shape, program);
          checkExpression(value.rhs, shape, program);
        } else if constexpr (std::is_same_v<T, shape::List>) {
          for (const auto& element : value.elements)
            checkExpression(element, shape, program);
        } else if constexpr (std::is_same_v<T, shape::Index>) {
          checkExpression(value.value, shape, program);
          checkExpression(value.index, shape, program);
        } else if constexpr (std::is_same_v<T, shape::Conditional>) {
          checkExpression(value.condition, shape, program);
          checkExpression(value.then_value, shape, program);
          checkExpression(value.else_value, shape, program);
        } else if constexpr (std::is_same_v<T, shape::Comprehension>) {
          check(value.variable.value < shape.symbols.size(),
                "Comprehension symbol must resolve");
          checkExpression(value.iterable, shape, program);
          checkExpression(value.element, shape, program);
        }
      },
      expression->value);
}

void examples(const char* executable, const char* tensor, const char* scalar) {
  std::string error;
  const std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::Create(executable, &error));
  check(runfiles != nullptr, "Cannot load runfiles");
  std::size_t checked = 0;
  for (const char* file : {tensor, scalar}) {
    const auto path = runfiles->Rlocation(file);
    std::ifstream input(path);
    check(static_cast<bool>(input), "Cannot read example dialect");
    const std::string source{std::istreambuf_iterator<char>(input), {}};
    const auto parsed = tepl::parse(source, path);
    check(parsed.ok(), "Example dialect must parse");
    const auto result = tepl::core::analyze(*parsed.program);
    if (!result.ok())
      for (const auto& diagnostic : result.diagnostics)
        std::cerr << diagnostic.message << '\n';
    check(result.ok(), "Every example shape definition must check");
    for (const auto& operation : result.program->operations) {
      if (!operation.shape) continue;
      ++checked;
      const auto& shape = *operation.shape;
      check(shape.parameters.size() == operation.operands.size(),
            "Parameter/operand mapping must be complete");
      for (const auto& symbol : shape.symbols)
        check(symbol.type.value < shape.types.size(),
              "Every symbol needs a concrete type");
      for (const auto& statement : shape.statements) {
        std::visit(
            [&](const auto& value) {
              using T = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<T, shape::Let>)
                checkExpression(value.value, shape, *result.program);
              else
                checkExpression(value.condition, shape, *result.program);
            },
            statement.value);
      }
      checkExpression(shape.result.value, shape, *result.program);
    }
  }
  check(checked == 24, "Expected 22 tensor and 2 scalar shape programs");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 3, "Expected tensor and scalar fixture paths");
    builtinSignatures();
    signaturesAndScopes();
    listsAndAttributes();
    integerPolicyAndRuntimeChecks();
    diagnosticsAndPrinting();
    examples(argv[0], argv[1], argv[2]);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
