#include <cassert>
#include <string>
#include <variant>

#include "src/ast/print.h"
#include "src/host_codegen.h"
#include "src/parse.h"
#include "src/semantic.h"
#include "src/simple_rewrite.h"

int main() {
  for (const auto* name : {"bool", "i8", "i16", "i32", "i64", "u8", "u16",
                           "u32", "u64", "f16", "bf16", "f32", "f64"}) {
    const auto source = "rule r { X: " + std::string(name) +
                        "[Batch..., M, _] X => X where { check(X); } }";
    const auto parsed = tepl::parse(source);
    assert(parsed.ok());
    const auto& rule = parsed.program->rules.front();
    const auto& tensor =
        std::get<tepl::ast::TensorDecl>(rule.declarations[0].value);
    assert(tensor.dtype && tensor.dtype->name == name);
    assert(tensor.dtype->span.begin.column == 13);
    assert(tepl::resolveDType(name));
    assert(tepl::validateTensorTypes(rule).empty());
    assert(tepl::formatAst(*parsed.program)
               .find(std::string(name) + "[Batch..., M, _]") !=
           std::string::npos);
    const auto host = tepl::generateHostTemplate(*parsed.program, false);
    assert(host.ok());
    assert(host.source.find("arg0: &TensorInfo") != std::string::npos);
  }

  for (const auto* type :
       {"f32[]", "[]", "scalar", "bf16[...]", "f16[_, S..., N]"}) {
    const auto parsed =
        tepl::parse("rule r { X: " + std::string(type) + " X => X }");
    assert(parsed.ok());
    assert(tepl::validateTensorTypes(parsed.program->rules.front()).empty());
  }
  for (const auto* source :
       {"rule r { X: float32[N] X => X }",
        "rule r { X: f32[N] X: bf16[N] X => X }", "rule r { Y: f32[N] X => X }",
        "rule r { 1:foo => 1:i32 }", "rule r { 128:i8 => 0:i8 }",
        "rule r { -129:i8 => 0:i8 }", "rule r { -1:u64 => 0:u64 }",
        "rule r { 18446744073709551616:u64 => 0:u64 }",
        "rule r { 1.0:i32 => 1:i32 }", "rule r { 2:bool => 0:bool }",
        "rule r extends base(F = t.add) { X: typo[N] }"}) {
    const auto parsed = tepl::parse(source);
    assert(parsed.ok());
    assert(!tepl::validateTensorTypes(parsed.program->rules.front()).empty());
    assert(!tepl::generateHostTemplate(*parsed.program, false).ok());
  }
  for (const auto* source :
       {"rule r { -128:i8 => 127:i8 }",
        "rule r { -9223372036854775808:i64 => 9223372036854775807:i64 }",
        "rule r { 18446744073709551615:u64 => 000:u64 }",
        "rule r { +1:bool => 0:bool }", "rule r { 1:f16 => -0.5:bf16 }",
        "rule r { Y: f32[] let Y = (negate X) => Y }",
        "abstract rule r(F: op<(tensor) -> tensor>) { X: f32[N] (F X) => X "
        "}"}) {
    const auto parsed = tepl::parse(source);
    assert(parsed.ok());
    assert(tepl::validateTensorTypes(parsed.program->rules.front()).empty());
  }

  const auto typed = tepl::parse("rule r { (add X 1:i32) => (add 1:i32 X) }");
  assert(typed.ok());
  const auto& rule = typed.program->rules.front();
  for (const auto* suffix : {"i32", "f32", "u32"}) {
    const auto input = tepl::parse(
        "rule input { (add A 1:" + std::string(suffix) + ") => A }");
    const auto result =
        tepl::rewriteOnce(rule, *input.program->rules.front().lhs);
    assert(result.has_value() == (std::string(suffix) == "i32"));
    if (result) {
      const auto& op = std::get<tepl::ast::Operator>((*result)->value);
      const auto& literal =
          std::get<tepl::ast::IntegerLiteral>(op.operands[0]->value);
      assert(literal.digits == "1" && literal.dtype->name == "i32");
    }
  }
  const auto bare = tepl::parse("rule input { (add A 1) => A }");
  assert(!tepl::rewriteOnce(rule, *bare.program->rules.front().lhs));

  // Constraint numbers remain host values; graph dtype never alters their type.
  const auto host_source = tepl::parse(
      "rule r { S: f32[] S => 1.0:f32 where { check(S, 1, 1.0); } }");
  const auto host = tepl::generateHostTemplate(*host_source.program, false);
  assert(host.ok());
  assert(host.source.find("arg0: &TensorInfo, arg1: usize, arg2: f64") !=
         std::string::npos);
}
