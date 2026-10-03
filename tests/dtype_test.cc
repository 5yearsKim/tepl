#include <cassert>
#include <string>
#include <variant>

#include "src/ast/print.h"
#include "src/parse.h"

int main() {
  for (const auto* name : {"bool", "i8", "i16", "i32", "i64", "u8", "u16",
                           "u32", "u64", "f16", "bf16", "f32", "f64"}) {
    const auto source = "rule r { X: " + std::string(name) +
                        "[Batch..., M, _] X => X where { $check(X); } }";
    const auto parsed = tepl::parse(source);
    assert(parsed.ok());
    const auto& rule = parsed.program->rules.front();
    const auto& tensor = rule.declarations[0];
    assert(tensor.dtype && tensor.dtype->name == name);
    assert(tensor.dtype->span.begin.column == 13);
    assert(tepl::formatAst(*parsed.program)
               .find(std::string(name) + "[Batch..., M, _]") !=
           std::string::npos);
  }

  for (const auto* type : {"f32[]", "[]", "bf16[...]", "f16[_, S..., N]"}) {
    const auto parsed =
        tepl::parse("rule r { X: " + std::string(type) + " X => X }");
    assert(parsed.ok());
  }

  // The frontend preserves annotations and literal spelling for core to check.
  for (const auto* dtype : {"i32", "f32", "u32", "unknown"}) {
    const auto parsed =
        tepl::parse("rule r { 001:" + std::string(dtype) + " => -1.2500:f32 }");
    assert(parsed.ok());
    const auto& rule = parsed.program->rules.front();
    const auto& integer = std::get<tepl::ast::IntegerLiteral>(rule.lhs->value);
    assert(integer.digits == "001" && integer.dtype->name == dtype);
    const auto& decimal = std::get<tepl::ast::FloatLiteral>(rule.rhs->value);
    assert(decimal.digits == "-1.2500" && decimal.dtype->name == "f32");
    const auto printed = tepl::formatAst(*parsed.program);
    assert(printed.find("integer 001:" + std::string(dtype)) !=
           std::string::npos);
    assert(printed.find("float -1.2500:f32") != std::string::npos);
  }
}
