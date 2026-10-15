#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/pipeline.h"
#include "src/core/resolution/loading.h"
#include "src/core/semantics/report.h"
#include "src/core/syntax/parse.h"
#include "src/core/syntax/print.h"

namespace {

using namespace tepl::core;

constexpr std::string_view kDialect = R"(
dialect Tensor {
  attrs DotAttrs { axes: index[] = []; }
  op add(lhs: tensor, rhs: tensor) -> tensor;
  op multiply(lhs: tensor, rhs: tensor) -> tensor { alias: mul; }
  op negate(input: tensor) -> tensor;
  op concat(first: tensor, rest: tensor...) -> tensor;
  op dot(lhs: tensor, rhs: tensor) -> tensor { attrs: DotAttrs; }
  op reshape(input: tensor) -> tensor { attrs { shape: index[]; } }
}
)";

InspectionResult analyzeText(std::string_view text, bool dialect = true) {
  const auto parsed = tepl::parse(
      (dialect ? std::string(kDialect) : "") + std::string(text), "test.tepl");
  if (!parsed.ok()) {
    for (const auto& error : parsed.diagnostics)
      std::cerr << error.message << '\n';
  }
  assert(parsed.ok());
  return inspect(*parsed.program);
}

void expectError(std::string_view text, std::string_view message,
                 bool dialect = true) {
  const auto result = analyzeText(text, dialect);
  assert(!result.ok());
  assert(!result.program);
  for (const auto& error : result.diagnostics) {
    assert(
        std::filesystem::path(error.origin.definition.source_name).filename() ==
        "test.tepl");
    assert(error.origin.definition.span.begin.line > 0);
    if (error.message.find(message) != std::string::npos) return;
  }
  std::cerr << "Expected diagnostic containing: " << message
            << "\nSource: " << text << '\n';
  for (const auto& error : result.diagnostics)
    std::cerr << error.message << '\n';
  assert(false);
}

void builtinResolution() {
  expectError("rule r { X => X where { 1 } }", "type mismatch");
  expectError("rule r { X => X where { let n = 1; n } }", "Bool");
  expectError("rule r { X => X where { [true] } }", "Bool");
  const auto result = analyzeText(R"(
rule r {
  X: [Batch..., N]
  (negate X) => X
  where {
    len(concat(Batch, range(N), Batch)) >= N
    && contains(Batch, N)
    && is_valid_axis_list(range(len(Batch)), len(Batch))
    && min(-7, -1) <= max(-7, 3)
    && floor_div(-7, 3) == -3 && ceil_div(-7, 3) == -2
    && (all([true, false]) || any([true]))
  }
}
)");
  assert(result.ok());
  assert(formatProgram(*result.program).find("Builtin(len") !=
         std::string::npos);
  for (const auto& signature : builtins::catalog()) {
    assert(builtins::resolveBuiltin(signature.name) == &signature);
    assert(signature.availableIn(builtins::Context::kWhere));
    assert(signature.availableIn(builtins::Context::kDerive));
  }
  expectError("rule r { X => X where { len() } }", "expects exactly 1");
  expectError("rule r { X: [B...] X => X where { concat(B) } }",
              "expects at least 2");
  expectError("rule r { X => X where { len(X) > 0 } }", "expected a list");
  expectError("rule r { X => X where { all([1]) } }", "type mismatch");
  expectError("rule r { X => X where { mystery() } }", "unknown builtin");
}

void removedHostSyntax() {
  for (const auto* source :
       {"rule r { X => X where { $legal(X) } }",
        "abstract rule a(f: fn<(tensor) -> bool>) { X => X }",
        "rule r extends a(f = $legal);"}) {
    const auto parsed = tepl::parse(source, "removed.tepl");
    assert(!parsed.ok() && !parsed.program);
  }
}

void concreteRules() {
  const auto result = analyzeText(R"(
rule commute_small {
  X: f32[N]
  Y: f32[N]
  (add X Y) => (add Y X)
  where { N <= 1024 }
}
rule repeated { (mul X X) => (multiply X X) }
rule bound { let Y = (negate X) => Y where { true } }
rule variadic { (concat X Y Z) => (concat X) }
rule scalar_capture { S: [] S => S }
rule unrestricted { X => X }
)");
  assert(result.ok());
  const auto& program = *result.program;
  assert(program.rules.size() == 6);
  const auto& rule = program.rules[0];
  assert(rule.name == "commute_small");
  assert(rule.captures.size() == 2 && rule.captures[0].name == "X" &&
         rule.captures[1].name == "Y");
  assert(rule.dimensions.size() == 1 && rule.dimensions[0].name == "N");
  assert(rule.constraints.size() == 2);
  assert(std::get<DType>(*rule.constraints[0].dtype) == DType::kF32);
  assert(rule.constraints[0].shape[0].symbol ==
         rule.constraints[1].shape[0].symbol);
  const auto& rhs = std::get<BuildOperation>(rule.rhs->value);
  assert(std::get<CaptureRef>(rhs.operands[0]->value).capture ==
         rule.captures[1].id);
  assert(program.types[rule.conditions[0]->type.value].kind == TypeKind::kBool);
  assert(rule.lhs->origin.definition.source_name == "test.tepl");
  const auto& repeated = std::get<MatchOperation>(program.rules[1].lhs->value);
  assert(repeated.operation ==
         std::get<BuildOperation>(program.rules[1].rhs->value).operation);
  assert(std::get<CapturePattern>(repeated.operands[0]->value).capture ==
         std::get<CapturePattern>(repeated.operands[1]->value).capture);
  const auto& binding = std::get<BindPattern>(program.rules[2].lhs->value);
  assert(binding.capture ==
         std::get<CaptureRef>(program.rules[2].rhs->value).capture);
  assert(program.rules[4].constraints[0].shape.empty());
  assert(program.rules[5].constraints.empty());
  const auto dump = formatProgram(program);
  assert(dump.find("commute_small") != std::string::npos);
  assert(dump.find("dimension #0 N: Index") != std::string::npos);
  assert(dump.find("Tensor.multiply") != std::string::npos);

  const auto literals =
      analyzeText("rule r { (add X 1.00:f32) => (add 1.0:f32 X) }");
  assert(literals.ok());
  const auto& literal_rule = literals.program->rules.front();
  assert(
      std::get<GraphLiteral>(
          std::get<MatchOperation>(literal_rule.lhs->value).operands[1]->value)
          .spelling == "1.00");
  assert(
      std::get<GraphLiteral>(
          std::get<BuildOperation>(literal_rule.rhs->value).operands[0]->value)
          .spelling == "1.0");
}

void descriptorsAndTypes() {
  const auto anonymous = analyzeText("rule r { (dot (dot X Y) Z) => X }");
  assert(anonymous.ok());
  const auto& anonymous_rule = anonymous.program->rules[0];
  const auto& outer = std::get<MatchOperation>(anonymous_rule.lhs->value);
  const auto& inner = std::get<MatchOperation>(outer.operands[0]->value);
  assert(outer.descriptor && inner.descriptor &&
         outer.descriptor != inner.descriptor);
  assert(anonymous_rule.descriptors.size() == 2);
  for (const auto& descriptor : anonymous_rule.descriptors) {
    assert(descriptor.kind == Descriptor::Kind::kCaptured);
    assert(anonymous.program->types[descriptor.type.value].schema);
  }

  const auto result = analyzeText(R"(
rule r {
  X: [Batch..., M, K]
  Y: [Batch..., K, N]
  (dot[@d] X Y) => (dot[@out] X Y)
  where { M % 8 == 0 && N >= 1 }
  derive {
    let axes = gather(@d.axes, range(len(@d.axes)));
    @first = { ..@d, axes: axes };
    @out = { ..@first };
  }
}
)");
  assert(result.ok());
  const auto& program = *result.program;
  const auto& rule = program.rules[0];
  assert(rule.descriptors.size() == 3 && rule.derivations.size() == 3);
  assert(!rule.derivations[0].target);
  const auto& constructor =
      std::get<DescriptorConstruct>(rule.derivations[1].value->value);
  assert(constructor.fields.size() == 1);
  assert(std::holds_alternative<LocalRef>(constructor.fields[0]->value));
  const auto& copy =
      std::get<DescriptorConstruct>(rule.derivations[2].value->value);
  assert(std::get<DescriptorFieldRef>(copy.fields[0]->value).descriptor ==
         rule.derivations[1].target);
  assert(program.attribute_schemas[0].fields[0].empty_default);
  assert(program.types[rule.dimensions[0].type.value].kind ==
         TypeKind::kIndexList);
  // Schemas can flow backwards from later RHS uses without allowing forward
  // values.
  assert(analyzeText("rule r { X => (reshape[@out] X) derive { @first = { "
                     "shape: [] }; @out = { ..@first }; } }")
             .ok());
  assert(analyzeText("rule r { X => (dot[@out] X X) derive { @out = {}; } }")
             .ok());
  expectError("rule r { X => (reshape[@out] X) derive { @out = {}; } }",
              "missing required");
  expectError(
      "rule r { X => (reshape[@out] X) derive { @out = { shape: [], shape: [] "
      "}; } }",
      "duplicate constructor field");
  expectError(
      "rule r { X => (reshape[@out] X) derive { @out = { wrong: [] }; } }",
      "unknown constructor field");
  expectError(
      "rule r { X => (reshape[@out] X) derive { @out = { shape: true }; } }",
      "type mismatch");
  expectError("rule r { X => (reshape[@out] X) derive { @out = { ..@out }; } }",
              "not available here");
  expectError(
      "rule r { X => (reshape[@out] X) derive { @out = { ..@later }; @later = "
      "{ shape: [] }; } }",
      "not available here");
  expectError(
      "rule r { (dot[@d] X X) => (reshape[@out] X) derive { @out = { ..@d }; } "
      "}",
      "type mismatch");
  expectError(
      "rule r { X => (reshape[@out] X) where { len(@out.shape) == 0 } derive { "
      "@out = { shape: [] }; } }",
      "not available here");
}

void constructorFieldTypes() {
  const auto result = analyzeText(R"(
dialect D {
  attrs A { name: string; padding: i64[][]; flags: bool[]; dtype: dtype;
            precision: precision[]; axis: index?; dims: index[] = []; region: region?; extra: index[]? = []; }
  op copy(x: tensor) -> tensor { attrs: A; }
}
rule r {
  (copy[@a] X) => (copy[@b] X)
  where { @a.region == none && none == @a.axis && len(@a.flags) >= 0 }
  derive {
    @b = { name: "tensor", padding: [[-1, 2], [0, 3]], flags: [true, false],
           dtype: f32, precision: [default, high], axis: if true then 1 else none };
  }
}
)",
                                  false);
  if (!result.ok())
    for (const auto& d : result.diagnostics) std::cerr << d.message << '\n';
  assert(result.ok());
  const auto& ctor = std::get<DescriptorConstruct>(
      result.program->rules[0].derivations[0].value->value);
  assert(ctor.fields.size() == 9);
  assert(std::holds_alternative<ConditionalExpr>(ctor.fields[5]->value));
  assert(std::get<ListExpr>(ctor.fields[6]->value).elements.empty());
  assert(std::holds_alternative<NoneConstant>(ctor.fields[7]->value));
  assert(std::holds_alternative<OptionalValue>(ctor.fields[8]->value));
  expectError("rule r { X => X derive { @a = { axes: [] }; } }",
              "requires a schema");
  expectError("rule r { X => (dot[@a] X X) derive { @a = { axes: [true] }; } }",
              "type mismatch");
  expectError("rule r { X => (dot[@a] X X) derive { @a = {}; @a = {}; } }",
              "duplicate descriptor");
  expectError("rule r { X => (dot[@a] (reshape[@a] X) X) derive { @a = {}; } }",
              "type mismatch");
  expectError(
      "dialect D { op copy(x:tensor)->tensor { attrs { axes:index[]?; } } } "
      "rule r { (copy[@a] X) => X where { @a.axes[0] == 0 } }",
      "expected a list", false);
}

void expressionTypesAndScopes() {
  const auto result = analyzeText(R"(
rule r {
  X: [Dims...]
  X => X
  where {
    let axes = range(len(Dims));
    let dimensions = [Dims[i] for i in axes];
    all([d >= 0 for d in dimensions])
    && (if len(Dims) == 0 then [] else dimensions) == Dims
    && gather([[1], [2]], [0]) == [[1]]
    && concat([true], [false]) == [true, false]
    && concat([1], [-1]) == [1, -1]
    && min(-9223372036854775808, -1) <= -1
    && 18446744073709551615 > 0
    && 1.5 + 0.5 == 2.0
  }
}
)");
  if (!result.ok())
    for (const auto& d : result.diagnostics) std::cerr << d.message << '\n';
  assert(result.ok());
  assert(result.program->rules[0].locals.size() == 4);

  assert(analyzeText("abstract rule a() { X => X where { let n = 1; n > 0 } } "
                     "rule r extends a() { where { let n = 2; n > 0 } }")
             .ok());
  expectError(
      "abstract rule a() { X => X where { let n = 1; n > 0 } } rule r extends "
      "a() { where { n > 0 } }",
      "unknown constraint name");
  expectError("rule r { X => X where { let a = a; true } }",
              "unknown constraint name");
  expectError("rule r { X => X where { let a = 1; let a = 2; true } }",
              "local name");
  expectError("rule r { X => X where { let X = 1; true } }", "local name");
  expectError(
      "rule r { X => (reshape[@out] X) where { let dims = [1]; true } derive { "
      "@out = { shape: dims }; } }",
      "unknown constraint name");
  expectError("rule r { X => X where { all([true for i in [1]]) && i > 0 } }",
              "unknown constraint name");
  expectError("rule r { X => X where { [1, true] == [1] } }",
              "operator does not support Bool");
  expectError("rule r { X => X where { [X] == [X] } }",
              "operator does not support Tensor[]");
  expectError("rule r { X => X where { 1[0] == 0 } }", "expected a list");
  expectError("rule r { X => X where { [1][true] == 0 } }", "type mismatch");
  expectError("rule r { X => X where { 18446744073709551616 > 0 } }",
              "invalid for Index");
  expectError("rule r { X => X where { -9223372036854775809 < 0 } }",
              "invalid for I64");
}

void inheritance() {
  auto parsed = tepl::parse(std::string(kDialect) + R"(
abstract rule commute(F: op<(tensor, tensor) -> tensor>) {
  X: [N]
  (F X Y) => (F Y X)
  where { N > 0 }
}
rule a extends commute(F = add) { X: f32[N] }
rule b extends commute(F = mul) { where { N < 32 } }
)",
                            "test.tepl");
  assert(parsed.ok());
  const auto before = tepl::formatAst(*parsed.program);
  const auto template_lhs = parsed.program->rules[0].lhs;
  auto result = inspect(*parsed.program);
  assert(result.ok());
  assert(tepl::formatAst(*parsed.program) == before);
  assert(parsed.program->rules[0].lhs == template_lhs);
  assert(std::get<tepl::syntax::Operator>(template_lhs->value).name == "F");
  // The IR owns its contents independently of the parser's AST lifetime.
  parsed.program.reset();
  const auto& rules = result.program->rules;
  assert(rules.size() == 2);
  assert(rules[0].constraints.size() == 2);
  assert(std::get<DType>(*rules[0].constraints[1].dtype) == DType::kF32);
  assert(rules[1].conditions.size() == 2);
  assert(std::get<MatchOperation>(rules[0].lhs->value).operation !=
         std::get<MatchOperation>(rules[1].lhs->value).operation);
  assert(rules[0].lhs->origin.expansions.size() == 1);
  assert(rules[0].lhs->origin.definition.span.begin.line !=
         rules[0].lhs->origin.expansions[0].span.begin.line);
}

void inheritedRanks() {
  // A sequence may be empty, and fixed dimensions on either side count
  // toward the minimum rank. The order of restrictions must not matter.
  for (const auto& [base, derived] : {
           std::pair{"[N, ...]", "[M]"},
           {"[N]", "[M, ...]"},
           {"[N, ..., M]", "[A, B]"},
           {"[]", "[]"},
           {"f32[]", "[]"},
           {"[]", "f32[]"},
           {"[...]", "[]"},
           {"[]", "[...]"},
           {"[N, ...]", "[A, B, ...]"},
           {"[A, B, ...]", "[N, ...]"},
       }) {
    const auto result =
        analyzeText(std::string("abstract rule a() { X: ") + base +
                    " X => X } rule r extends a() { X: " + derived + " }");
    assert(result.ok());
    assert(result.program->rules[0].constraints.size() == 2);
  }
  for (const auto& [base, derived] : {
           std::pair{"[N, ...]", "[]"},
           {"[]", "[N, ...]"},
           {"[N, ..., M]", "[A]"},
           {"[A]", "[N, ..., M]"},
           {"[_, ...]", "[]"},
           {"[]", "[_]"},
           {"[_]", "[]"},
       }) {
    const auto result =
        analyzeText(std::string("abstract rule a() { X: ") + base +
                        " X => X }\nrule r extends a() { X: " + derived + " }",
                    false);
    assert(!result.ok());
    assert(!result.program);
    assert(result.diagnostics.size() == 1);
    const auto& error = result.diagnostics[0];
    assert(error.message.find("conflicting rank") != std::string::npos);
    assert(error.origin.definition.span.begin.line == 2);
    assert(error.related.size() == 1);
    assert(error.related[0].span.begin.line == 1);
  }
}

void tensorDTypes() {
  for (const auto* name : {"bool", "i8", "i16", "i32", "i64", "u8", "u16",
                           "u32", "u64", "f16", "bf16", "f32", "f64"}) {
    const auto dtype = resolveDType(name);
    assert(dtype && dtypeName(*dtype) == name);
    const auto result = analyzeText("rule r { X: " + std::string(name) +
                                    "[Batch..., M, _] X => X where { true } }");
    assert(result.ok());
    const auto& rule = result.program->rules.front();
    assert(std::get<DType>(*rule.constraints.front().dtype) == *dtype);
    assert(rule.constraints.front().shape.size() == 3);
  }

  for (const auto* type : {"f32[]", "[]", "bf16[...]", "f16[_, S..., N]"}) {
    assert(analyzeText("rule r { X: " + std::string(type) + " X => X }").ok());
  }
  for (const auto* source :
       {"rule r { -128:i8 => 127:i8 }",
        "rule r { -9223372036854775808:i64 => 9223372036854775807:i64 }",
        "rule r { 18446744073709551615:u64 => 000:u64 }",
        "rule r { +1:bool => 0:bool }", "rule r { 1:f16 => -0.5:bf16 }",
        "rule r { Y: f32[] let Y = (negate X) => Y }",
        "abstract rule a(F: op<(tensor) -> tensor>) { X: f32[N] (F X) => X } "
        "rule r extends a(F = negate);"}) {
    assert(analyzeText(source).ok());
  }

  // Abstract body annotations are checked when a concrete rule instantiates it.
  assert(analyzeText("abstract rule a() { X: typo[N] X => X }").ok());
  expectError("abstract rule a() { X: typo[N] X => X } rule r extends a();",
              "unknown dtype");
}

void invalidRules() {
  for (const auto& [source, diagnostic] : {
           std::pair{"rule r { X => Y }", "unknown RHS capture"},
           {"rule r { (missing X) => X }", "unknown operation"},
           {"rule r { (add X) => X }", "expects exactly 2"},
           {"rule r { (concat) => X }", "expects at least 1"},
           {"rule r { X: [N] X: [N] X => X }", "duplicate tensor declaration"},
           {"rule r { Y: [N] X => X }", "does not refer to an LHS capture"},
           {"rule r { X: typo[N] X => X }", "unknown dtype"},
           {"rule r { X: [X] X => X }", "conflicts with a tensor capture"},
           {"rule r { X: [N] Y: [N...] (add X Y) => X }",
            "both an index and a sequence"},
           {"rule r { let X = (negate X) => X }",
            "conflicts with an existing LHS capture"},
           {"rule r { X => X where { X } }", "type mismatch"},
           {"rule r { X => X where { 1 } }", "type mismatch"},
           {"rule r { X => X where { missing < 1 } }",
            "unknown constraint name"},
           {"rule r { X => X where { X == X } }",
            "operator does not support Tensor"},
           {"rule r { X => X where { true < false } }",
            "operator does not support Bool"},
           {"rule r { X => X where { 1.5 % 1.0 == 0 } }",
            "operator does not support F64"},
           {"rule r { 128:i8 => 0:i8 }", "invalid for dtype"},
           {"rule r { -129:i8 => 0:i8 }", "invalid for dtype"},
           {"rule r { 18446744073709551616:u64 => 0:u64 }",
            "invalid for dtype"},
           {"rule r { 2:bool => 0:bool }", "invalid for dtype"},
           {"rule r { -1:u64 => 0:u64 }", "invalid for dtype"},
           {"rule r { 1.0:i32 => 1:i32 }", "invalid for dtype"},
           {"rule r { 1:unknown => X }", "unknown dtype"},
           {"rule r { (add[@d] X Y) => X }", "has no attributes"},
           {"rule r { (dot[@d] X Y) => (dot[@missing] X Y) }",
            "unknown RHS descriptor"},
           {"rule r { (dot[@d] X Y) => (reshape[@d] X) }", "type mismatch"},
           {"rule r { X => X derive { @d = true; } }", "type mismatch"},
           {"rule r { (tuple X Y) => X }", "tuple semantics"},
           {"rule r { (get[0] X) => X }", "projection semantics"},
           {"rule r { X => X } rule r { Y => Y }", "duplicate rule"},
           {"rule r extends missing();", "unknown base rule"},
           {"rule a { X => X } rule r extends a();", "must be an abstract"},
           {"abstract rule a(F: op<(tensor) -> tensor>) { (F X) => X } rule r "
            "extends a();",
            "missing binding"},
           {"abstract rule a() { X => X } rule r extends a(F = add);",
            "unknown parameter binding"},
           {"abstract rule a(F: op<(tensor) -> tensor>) { (F X) => X } rule r "
            "extends a(F = negate, F = negate);",
            "duplicate binding"},
           {"abstract rule a(F: op<(tensor) -> tensor>) { (F X) => X } rule r "
            "extends a(F = add);",
            "does not match signature"},
           {"abstract rule a(F: op<(tensor) -> tensor>) { (F X) => X } rule r "
            "extends a(F = concat);",
            "does not match signature"},
           {"abstract rule a(F: op<(unknown) -> tensor>) { X => X }",
            "unsupported parameter signature"},
           {"abstract rule a(F: op<() -> tensor>, F: op<() -> tensor>) { X => "
            "X }",
            "duplicate rule parameter"},
           {"abstract rule a() { X: f32[N] X => X } rule r extends a() { X: "
            "bf16[N] }",
            "conflicting dtype"},
           {"abstract rule a() { X: [N] X => X } rule r extends a() { X: [M, "
            "N] }",
            "conflicting rank"},
       })
    expectError(source, diagnostic);
  expectError("rule r { (add X Y) => X }", "visible dialect declaration",
              false);
  expectError("import \"unloaded.tepl\"; rule r { X => X }",
              "cannot open import", false);
  expectError("from \"unloaded.tepl\" import D; rule r { X => X }",
              "cannot open import", false);
  expectError("from \"unloaded.tepl\" import {template}; rule r { X => X }",
              "cannot open import", false);
  expectError("rule r { X: [N] X => X where { N + 1.0 > 0.0 } }",
              "type mismatch");
  expectError("rule r { X: [N] X => X where { -N > -1 } }",
              "operator does not support Index");
  expectError("dialect D { op bad(x: unknown) -> tensor; }",
              "unsupported operation operand", false);
  expectError("dialect D { op bad() -> unknown; }",
              "unsupported operation result", false);
  expectError("dialect D { op a() -> tensor; op b() -> tensor { alias: a; } }",
              "duplicate operation", false);
  expectError("dialect D { op a() -> tensor { attrs: Missing; } }",
              "unknown attribute schema", false);
  expectError("dialect D { attrs A { x: unknown; } }", "unknown attribute type",
              false);
  expectError("dialect D { attrs A { x: index = []; } }",
              "requires a list type", false);
}

void imports() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(std::getenv("TEST_TMPDIR")) / "core_imports";
  fs::create_directories(dir);
  const auto write = [&](const char* file, std::string_view text) {
    std::ofstream output(dir / file);
    output << text;
    assert(output.good());
  };
  write("private.tepl", "dialect Private { op negate(x: tensor) -> tensor; }");
  write("template.tepl", R"(
from "private.tepl" import Private as p;
abstract rule template(F: op<(tensor, tensor) -> tensor>) {
  (F (p.negate X) Y) => (F Y (p.negate X))
}
abstract rule unselected() { X => X }
)");
  auto parsed = tepl::parse(R"(
from "template.tepl" import {template};
dialect Public {
  op add(x: tensor, y: tensor) -> tensor;
  op negate(x: tensor) -> tensor;
}
rule r extends template(F = add);
)",
                            (dir / "root.tepl").string());
  assert(parsed.ok());
  const auto loaded = tepl::loadImports(*parsed.program);
  assert(loaded.ok());
  const auto again = tepl::loadImports(*parsed.program);
  assert(again.ok() &&
         again.program->modules.size() == loaded.program->modules.size());
  const auto result = inspect(*parsed.program);
  assert(result.ok());
  const auto& rule = result.program->rules[0];
  const auto& root = std::get<MatchOperation>(rule.lhs->value);
  const auto& private_op = std::get<MatchOperation>(root.operands[0]->value);
  assert(result.program->operations[root.operation.value].dialect == "Public");
  assert(result.program->operations[private_op.operation.value].dialect ==
         "Private");
  assert(rule.lhs->origin.definition.source_name ==
         (dir / "template.tepl").string());
  assert(rule.lhs->origin.expansions[0].source_name ==
         (dir / "root.tepl").string());

  const auto invisible = [&](std::string_view body, std::string_view expected) {
    auto input = tepl::parse(
        "from \"template.tepl\" import {template}; " + std::string(body),
        (dir / "root.tepl").string());
    assert(input.ok());
    assert(tepl::loadImports(*input.program).ok());
    const auto result = inspect(*input.program);
    assert(!result.ok());
    bool found = false;
    for (const auto& diagnostic : result.diagnostics)
      found |= diagnostic.message.find(expected) != std::string::npos;
    assert(found);
  };
  invisible("rule r { (p.negate X) => X }", "unknown operation");
  invisible("rule r extends unselected();", "unknown base rule");

  write("aggregate.tepl", "import \"private.tepl\";");
  auto aggregate =
      tepl::parse("import \"aggregate.tepl\"; rule r { (negate X) => X }",
                  (dir / "root.tepl").string());
  assert(aggregate.ok());
  assert(tepl::loadImports(*aggregate.program).ok());
  assert(inspect(*aggregate.program).ok());
}

void examples(const char* executable, const char* lora, const char* inherited) {
  using rules_cc::cc::runfiles::Runfiles;
  std::string error;
  const std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(executable, &error));
  assert(runfiles);
  for (const char* file : {lora, inherited}) {
    const auto path = runfiles->Rlocation(file);
    std::ifstream input(path);
    assert(input);
    const std::string source{std::istreambuf_iterator<char>(input), {}};
    auto parsed = tepl::parse(source, path);
    assert(parsed.ok());
    assert(tepl::loadImports(*parsed.program).ok());
    const auto result = inspect(*parsed.program);
    assert(result.ok());
    assert(result.program->rules.size() == (file == lora ? 1 : 7));
    assert(!formatProgram(*result.program).empty());
  }
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 3);
  removedHostSyntax();
  builtinResolution();
  concreteRules();
  descriptorsAndTypes();
  expressionTypesAndScopes();
  constructorFieldTypes();
  inheritance();
  inheritedRanks();
  tensorDTypes();
  invalidRules();
  imports();
  examples(argv[0], argv[1], argv[2]);
}
