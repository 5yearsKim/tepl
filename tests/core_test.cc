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
#include "src/ast/print.h"
#include "src/core/analyze.h"
#include "src/core/print.h"
#include "src/imports.h"
#include "src/parse.h"

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

AnalysisResult analyzeText(std::string_view text, bool dialect = true) {
  const auto parsed = tepl::parse(
      (dialect ? std::string(kDialect) : "") + std::string(text), "test.tepl");
  if (!parsed.ok()) {
    for (const auto& error : parsed.diagnostics)
      std::cerr << error.message << '\n';
  }
  assert(parsed.ok());
  return analyze(*parsed.program);
}

void expectError(std::string_view text, std::string_view message,
                 bool dialect = true) {
  const auto result = analyzeText(text, dialect);
  assert(!result.ok());
  assert(!result.program);
  for (const auto& error : result.diagnostics) {
    assert(error.origin.definition.source_name == "test.tepl");
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
  const auto result = analyzeText(R"(
rule r {
  X: [Batch..., N]
  (negate X) => X
  where {
    len(concat(Batch, range(N), Batch)) >= N;
    contains($axes(X), N);
    is_valid_axis_list(range(len(Batch)), len(Batch));
    min($signed(X), -1) <= max(-7, 3);
    floor_div(-7, 3) == -3 && ceil_div(-7, 3) == -2;
  }
}
abstract rule a(len: fn<(tensor) -> bool>) { X => X where { len(X); } }
rule bound extends a(len = $legal);
)");
  assert(result.ok());
  const auto& program = *result.program;
  assert(std::holds_alternative<BuiltinCall>(
      std::get<BinaryExpr>(program.rules[0].conditions[0]->value).lhs->value));
  assert(
      std::holds_alternative<HostCall>(program.rules[1].conditions[0]->value));
  assert(program.host_functions.size() == 3);  // Builtins add no host entries.
  assert(formatProgram(program).find("Builtin(len") != std::string::npos);
  for (const auto& signature : builtins::catalog()) {
    assert(builtins::resolveBuiltin(signature.name) == &signature);
    assert(signature.availableIn(builtins::Context::kShape) ==
           (signature.domain != builtins::Domain::kDType));
    assert(signature.availableIn(builtins::Context::kDType));
    const bool boolean_list = signature.builtin == builtins::Builtin::kAll ||
                              signature.builtin == builtins::Builtin::kAny;
    assert(signature.availableIn(builtins::Context::kWhere) == !boolean_list);
    assert(signature.availableIn(builtins::Context::kDerive) == !boolean_list);
  }
  expectError("rule r { X: [B...] X => X where { len(); } }",
              "expects exactly 1");
  expectError("rule r { X: [B...] X => X where { concat(B); } }",
              "expects at least 2");
  expectError("rule r { X: [B...] X => X where { contains(B, true); } }",
              "type mismatch");
  expectError("rule r { X: [B...] X => X where { len(B); } }", "type mismatch");
  expectError("rule r { X: [B...] X => X where { all(B); } }",
              "not available in where");
  expectError(
      "rule r { X: [B...] X => X derive { @d = $infer_attrs(any(B)); } }",
      "not available in derive");
  expectError("rule r { X => X where { min(1.0, 2.0) > 0.0; } }",
              "does not support F64");
  expectError("rule r { X: [N] X => X where { min(N, -1) == N; } }",
              "invalid for Index");
}

void hostCallResolution() {
  expectError("rule r { X => X where { legal(X); } }",
              "use '$legal(...)' to call a host function");
  expectError("rule r { X => X where { $legal(len(X)); } }",
              "type mismatch: Tensor versus IndexList");
  expectError(
      "abstract rule a(f: fn<(tensor) -> bool>) { X => X where { f(X); } } "
      "rule r extends a(f = legal);",
      "requires '$' before 'legal'");
  expectError(
      "abstract rule a(F: op<(tensor) -> tensor>) { (F X) => X } "
      "rule r extends a(F = $negate);",
      "host function cannot bind operation parameter 'F'");
  // Explicit host calls bypass both operation and function parameters.
  const auto result = analyzeText(R"(
abstract rule a(F: op<(tensor) -> tensor>, p: fn<(tensor) -> bool>) {
  (F X) => X where { p(X); $p(X); $F(X); $len(X); }
}
rule r extends a(F = negate, p = $legal);
)");
  assert(result.ok());
  const auto& program = *result.program;
  const auto& conditions = program.rules[0].conditions;
  const char* names[] = {"legal", "p", "F", "len"};
  assert(conditions.size() == 4);
  for (std::size_t i = 0; i < conditions.size(); ++i) {
    const auto& call = std::get<HostCall>(conditions[i]->value);
    assert(program.host_functions[call.function.value].name == names[i]);
  }
  const auto printed = formatProgram(program);
  assert(printed.find("$legal #") != std::string::npos);
  assert(printed.find("$p #") != std::string::npos);
}

void concreteRules() {
  const auto result = analyzeText(R"(
rule commute_small {
  X: f32[N]
  Y: f32[N]
  (add X Y) => (add Y X)
  where { N <= 1024; }
}
rule repeated { (mul X X) => (multiply X X) }
rule bound { let Y = (negate X) => Y where { $allowed(Y); } }
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
  const auto result = analyzeText(R"(
rule r {
  X: [Batch..., M, K]
  Y: [Batch..., K, N]
  (dot[@d] X Y) => (dot[@out] X Y)
  where { $allowed(X, @d); M % 8 == 0 && N >= 1; }
  derive {
    @first = $infer(X, Y, @d);
    @out = $update(@first);
  }
}
)");
  assert(result.ok());
  const auto& program = *result.program;
  const auto& rule = program.rules[0];
  assert(rule.descriptors.size() == 3);
  assert(rule.descriptors[0].kind == Descriptor::Kind::kCaptured);
  assert(rule.descriptors[1].kind == Descriptor::Kind::kDerived);
  assert(rule.derivations.size() == 2);
  assert(rule.derivations[0].target == rule.descriptors[1].id);
  const auto& update = std::get<HostCall>(rule.derivations[1].value->value);
  assert(std::get<DescriptorRef>(update.arguments[0]->value).descriptor ==
         rule.derivations[0].target);
  assert(
      program.types[rule.derivations[1].value->type.value].schema ==
      program
          .operations[std::get<BuildOperation>(rule.rhs->value).operation.value]
          .attributes);
  assert(program.attribute_schemas[0].fields[0].empty_default);
  assert(program.types[rule.dimensions[0].type.value].kind ==
         TypeKind::kIndexList);
  assert(program.host_functions[0].fallible);

  const auto nested = analyzeText(R"(
rule r {
  X => X
  where {
    $outer($inner(X));
    $inner(X) < 10;
    $float_value(X) > -1.5;
    $check(-9223372036854775808, 18446744073709551615, +1.0);
  }
}
)");
  if (!nested.ok())
    for (const auto& error : nested.diagnostics)
      std::cerr << error.message << '\n';
  assert(nested.ok());
  const auto& functions = nested.program->host_functions;
  assert(functions[0].name == "outer");
  assert(
      nested.program->types[functions[0].signature.arguments[0].value].kind ==
      TypeKind::kIndex);
  assert(nested.program->types[functions[1].signature.result.value].kind ==
         TypeKind::kIndex);
  assert(nested.program->types[functions[2].signature.result.value].kind ==
         TypeKind::kF64);

  // Literal defaulting happens after all calls contribute their constraints.
  for (const auto* conditions :
       {"$score(X) == 1; $score(X) > -1;", "$score(X) > -1; $score(X) == 1;"}) {
    const auto signed_result = analyzeText(
        std::string("rule r { X => X where { ") + conditions + " } }");
    assert(signed_result.ok());
    const auto& program = *signed_result.program;
    assert(
        program.types[program.host_functions[0].signature.result.value].kind ==
        TypeKind::kI64);
  }
  for (const auto* conditions :
       {"$check(1); $check(1.0);", "$check(1.0); $check(1);"}) {
    const auto floating = analyzeText(std::string("rule r { X => X where { ") +
                                      conditions + " } }");
    assert(floating.ok());
    const auto& program = *floating.program;
    assert(program.types[program.host_functions[0].signature.arguments[0].value]
               .kind == TypeKind::kF64);
  }
}

void inferenceAcrossRules() {
  // Later rules constrain earlier nested host calls and equality expressions.
  const auto result = analyzeText(R"(
rule first { X => X where { $accept($measure(X)); } }
rule second { Y => Y where { $measure(Y) == $other(Y); } }
rule third { Z => Z where { $other(Z) > 1.0; } }
)");
  assert(result.ok());
  const auto& program = *result.program;
  assert(program.types[program.host_functions[0].signature.arguments[0].value]
             .kind == TypeKind::kF64);
  assert(program.types[program.host_functions[1].signature.result.value].kind ==
         TypeKind::kF64);
  assert(program.types[program.host_functions[2].signature.result.value].kind ==
         TypeKind::kF64);

  // Large sets of equations must not rely on recursive union-find traversal.
  std::string many_calls = "rule r { X => X where { ";
  for (int i = 0; i < 4096; ++i) many_calls += "$accept(1); ";
  many_calls += "$accept(1.0); } }";
  const auto large = analyzeText(many_calls);
  assert(large.ok());
  assert(large.program->rules[0].conditions.size() == 4097);
  assert(
      large.program
          ->types[large.program->host_functions[0].signature.arguments[0].value]
          .kind == TypeKind::kF64);
}

void inheritance() {
  auto parsed = tepl::parse(std::string(kDialect) + R"(
abstract rule commute(F: op<(tensor, tensor) -> tensor>,
                      allowed: fn<(tensor, tensor) -> bool>) {
  X: [N]
  (F X Y) => (F Y X)
  where { allowed(X, Y); }
}
rule a extends commute(F = add, allowed = $legal) { X: f32[N] }
rule b extends commute(F = mul, allowed = $legal) { where { N < 32; } }
)",
                            "test.tepl");
  assert(parsed.ok());
  const auto before = tepl::formatAst(*parsed.program);
  const auto template_lhs = parsed.program->rules[0].lhs;
  auto result = analyze(*parsed.program);
  assert(result.ok());
  assert(tepl::formatAst(*parsed.program) == before);
  assert(parsed.program->rules[0].lhs == template_lhs);
  assert(std::get<tepl::ast::Operator>(template_lhs->value).name == "F");
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

  const auto signature = analyzeText(R"(
abstract rule guarded(f: fn<(tensor) -> i64>) {
  X => X where { f(X) >= -1; }
}
rule r extends guarded(f = $score);
)");
  assert(signature.ok());
  assert(
      signature.program
          ->types[signature.program->host_functions[0].signature.result.value]
          .kind == TypeKind::kI64);

  // Substitution applies only inside the template, and bound function names
  // are not reinterpreted as parameter names after substitution.
  const auto functions = analyzeText(R"(
abstract rule guarded(F: op<(tensor) -> tensor>,
                      predicate: fn<(tensor) -> bool>) {
  (F X) => X where { !predicate(X) || predicate(X); }
}
rule r extends guarded(F = negate, predicate = $F) {
  where { $predicate(X); }
}
)");
  assert(functions.ok());
  const auto& program = *functions.program;
  const auto& conditions = program.rules[0].conditions;
  const auto& disjunction = std::get<BinaryExpr>(conditions[0]->value);
  const auto& negation = std::get<UnaryExpr>(disjunction.lhs->value);
  const auto& left = std::get<HostCall>(negation.operand->value);
  const auto& right = std::get<HostCall>(disjunction.rhs->value);
  const auto& local = std::get<HostCall>(conditions[1]->value);
  assert(left.function == right.function);
  assert(program.host_functions[left.function.value].name == "F");
  assert(program.host_functions[local.function.value].name == "predicate");
  assert(negation.operand->origin.expansions.size() == 1);
  assert(conditions[1]->origin.expansions.empty());
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
    const auto result =
        analyzeText("rule r { X: " + std::string(name) +
                    "[Batch..., M, _] X => X where { $check(X); } }");
    assert(result.ok());
    const auto& rule = result.program->rules.front();
    assert(std::get<DType>(*rule.constraints.front().dtype) == *dtype);
    assert(rule.constraints.front().shape.size() == 3);
    const auto& host = result.program->host_functions.front();
    assert(result.program->types[host.signature.arguments[0].value].kind ==
           TypeKind::kTensor);
    assert(result.program->types[host.signature.result.value].kind ==
           TypeKind::kBool);
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

  // Graph dtypes do not alter scalar host argument types.
  const auto result = analyzeText(
      "rule r { S: f32[] S => 1.0:f32 where { $check(S, 1, 1.0); } }");
  assert(result.ok());
  const auto& host = result.program->host_functions.front();
  assert(result.program->types[host.signature.arguments[0].value].kind ==
         TypeKind::kTensor);
  assert(result.program->types[host.signature.arguments[1].value].kind ==
         TypeKind::kIndex);
  assert(result.program->types[host.signature.arguments[2].value].kind ==
         TypeKind::kF64);
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
           {"rule r { X => X where { X; } }", "type mismatch"},
           {"rule r { X => X where { 1; } }", "type mismatch"},
           {"rule r { X => X where { missing < 1; } }",
            "unknown constraint name"},
           {"rule r { X => X where { X == X; } }",
            "operator does not support Tensor"},
           {"rule r { X => X where { true < false; } }",
            "operator does not support Bool"},
           {"rule r { X => X where { $f(X) == $g(X); } }", "cannot infer"},
           {"rule r { X => X where { $f(X); $f(1); } }", "type mismatch"},
           {"rule r { X => X where { $f(X); $f(X, X); } }",
            "conflicting arity"},
           {"rule r { X => X where { 1.5 % 1.0 == 0; } }",
            "operator does not support F64"},
           {"rule r { X => X where { $f(-9223372036854775809); } }",
            "invalid for I64"},
           {"rule r { X => X where { $f(18446744073709551616); } }",
            "invalid for Index"},
           {"rule r { 128:i8 => 0:i8 }", "invalid for dtype"},
           {"rule r { -129:i8 => 0:i8 }", "invalid for dtype"},
           {"rule r { 18446744073709551616:u64 => 0:u64 }",
            "invalid for dtype"},
           {"rule r { 2:bool => 0:bool }", "invalid for dtype"},
           {"rule r { -1:u64 => 0:u64 }", "invalid for dtype"},
           {"rule r { 1.0:i32 => 1:i32 }", "invalid for dtype"},
           {"rule r { 1:unknown => X }", "unknown dtype"},
           {"rule r { (dot X Y) => X }", "requires an attribute descriptor"},
           {"rule r { (add[@d] X Y) => X }", "has no attributes"},
           {"rule r { (dot[@d] X Y) => (dot[@missing] X Y) }",
            "unknown RHS descriptor"},
           {"rule r { (dot[@d] X Y) => (reshape[@d] X) }", "type mismatch"},
           {"rule r { X => X where { $check(@missing); } }",
            "not available here"},
           {"rule r { X => X derive { @d = $make(@d); } }",
            "not available here"},
           {"rule r { X => X derive { @a = $make(@b); @b = $make(X); } }",
            "not available here"},
           {"rule r { X => X where { $check(@d); } derive { @d = $make(X); } }",
            "not available here"},
           {"rule r { (dot[@d] X Y) => X derive { @d = $make(X); } }",
            "duplicate descriptor definition"},
           {"rule r { X => X derive { @d = $make(X); @d = $make(X); } }",
            "duplicate descriptor definition"},
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
  expectError("import \"unloaded.tepl\"; rule r { X => X }", "is not loaded",
              false);
  expectError("from \"unloaded.tepl\" import D; rule r { X => X }",
              "is not loaded", false);
  expectError("from \"unloaded.tepl\" import {template}; rule r { X => X }",
              "is not loaded", false);
  expectError("rule r { X: [N] X => X where { N + 1.0 > 0.0; } }",
              "type mismatch");
  expectError("rule r { X: [N] X => X where { -N > -1; } }",
              "operator does not support Index");
  expectError(
      "rule a { X => (dot[@d] X X) derive { @d = $create(X); } } "
      "rule b { Y => (reshape[@d] Y) derive { @d = $create(Y); } }",
      "type mismatch");
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
  assert(tepl::resolveImports(*parsed.program).empty());
  const auto scope_count = parsed.program->imported_scopes.size();
  assert(tepl::resolveImports(*parsed.program).empty());
  assert(parsed.program->imported_scopes.size() == scope_count);
  const auto result = analyze(*parsed.program);
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
    assert(tepl::resolveImports(*input.program).empty());
    const auto result = analyze(*input.program);
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
  assert(tepl::resolveImports(*aggregate.program).empty());
  assert(analyze(*aggregate.program).ok());
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
    assert(tepl::resolveImports(*parsed.program).empty());
    const auto result = analyze(*parsed.program);
    assert(result.ok());
    assert(result.program->rules.size() == (file == lora ? 1 : 7));
    assert(!formatProgram(*result.program).empty());
  }
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 3);
  hostCallResolution();
  builtinResolution();
  concreteRules();
  descriptorsAndTypes();
  inferenceAcrossRules();
  inheritance();
  inheritedRanks();
  tensorDTypes();
  invalidRules();
  imports();
  examples(argv[0], argv[1], argv[2]);
}
