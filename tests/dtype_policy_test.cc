#include <algorithm>
#include <array>
#include <cassert>
#include <string>
#include <variant>

#include "src/ast/print.h"
#include "src/core/analyze.h"
#include "src/core/print.h"
#include "src/parse.h"

namespace {

tepl::ParseResult parseBody(const std::string& body) {
  return tepl::parse(
      "dialect D { attrs A { dtype: index; } "
      "op test(dtype: tensor, rest: tensor...) -> tensor { " +
          body + " } }",
      "dtype_policy.tepl");
}

void policies() {
  using tepl::core::CommonDTypePolicy;
  for (const auto* name :
       {"same", "same_numeric", "same_float", "bool", "i8", "i16", "i32", "i64",
        "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64"}) {
    const auto parsed = parseBody("dtype: " + std::string(name) + ";");
    assert(parsed.ok());
    const auto before = tepl::formatAst(*parsed.program);
    const auto& syntax = parsed.program->dialects[0].operations[0].dtype_policy;
    assert(syntax && syntax->name == name);
    assert(before.find("dtype " + std::string(name)) != std::string::npos);
    const auto checked = tepl::core::analyze(*parsed.program);
    assert(checked.ok());
    assert(tepl::formatAst(*parsed.program) == before);
    const auto& policy = checked.program->operations[0].dtype_policy;
    assert(policy);
    if (auto dtype = tepl::core::resolveDType(name)) {
      assert(std::get<tepl::core::DType>(*policy) == *dtype);
    } else {
      const auto expected = std::string(name) == "same"
                                ? CommonDTypePolicy::kSame
                            : std::string(name) == "same_numeric"
                                ? CommonDTypePolicy::kSameNumeric
                                : CommonDTypePolicy::kSameFloat;
      assert(std::get<CommonDTypePolicy>(*policy) == expected);
    }
    assert(tepl::core::formatProgram(*checked.program)
               .find("dtype=" + std::string(name)) != std::string::npos);
  }
}

void optionalAndContextual() {
  const auto parsed = tepl::parse(R"(
    dialect D {
      op plain(x: tensor) -> tensor;
      op empty() -> tensor {}
      op shaped(x: tensor) -> tensor { shape(s) { yield s; } }
      op fixed() -> tensor { dtype: f32; }
      op no_inputs() -> tensor { dtype: same; }
      op dtype(dtype: tensor) -> tensor {
        dtype: same;
        attrs { dtype: index; }
        shape(dtype) { assert attrs.dtype >= 0; yield dtype; }
      }
    }
  )");
  assert(parsed.ok());
  const auto checked = tepl::core::analyze(*parsed.program);
  assert(checked.ok());
  for (int i = 0; i < 3; ++i) {
    assert(!parsed.program->dialects[0].operations[i].dtype_policy);
    assert(!checked.program->operations[i].dtype_policy);
  }
  assert(checked.program->operations[3].dtype_policy);
  assert(checked.program->operations[4].dtype_policy);
  assert(checked.program->operations[5].dtype_policy);
  assert(checked.program->operations[5].shape);
}

void propertyOrders() {
  // Every subset and ordering, with both forms of attrs and optional shape.
  for (const auto* attrs : {"attrs: A;", "attrs { dtype: index; }"}) {
    const std::array<std::string, 3> properties = {"alias: renamed;", attrs,
                                                   "dtype: same;"};
    for (unsigned mask = 0; mask < 8; ++mask) {
      std::array<int, 3> order = {0, 1, 2};
      do {
        std::string body;
        for (int index : order)
          if (mask & (1u << index)) body += properties[index] + " ";
        for (const auto* shape : {"", "shape(s, rest...) { yield s; }"}) {
          const auto parsed = parseBody(body + shape);
          assert(parsed.ok());
          const auto checked = tepl::core::analyze(*parsed.program);
          assert(checked.ok());
          const auto& op = checked.program->operations[0];
          assert(op.dtype_policy.has_value() == bool(mask & 4));
          assert(op.alias.has_value() == bool(mask & 1));
          assert(op.attributes.has_value() == bool(mask & 2));
        }
      } while (std::next_permutation(order.begin(), order.end()));
    }
  }
}

void errors() {
  for (const auto* body :
       {"dtype: same; dtype: f32;", "dtype: same; alias: a; dtype: same;",
        "dtype same;", "dtype: ;", "dtype: same", "dtype: 32;", "type: same;",
        "dtype: same(x);", "dtype: same; alias: a; alias: b;",
        "attrs: A; dtype: same; attrs: A;",
        "shape(s, rest...) { yield s; } dtype: same;",
        "dtype: same; shape(s, rest...) { yield s; } shape(s) { yield s; }"}) {
    const auto parsed = parseBody(body);
    assert(!parsed.ok() && !parsed.program);
  }
  for (const auto* name : {"unknown", "promote", "float", "f128"}) {
    const auto parsed =
        tepl::parse("dialect D { op test(x: tensor) -> tensor {\n  dtype: " +
                        std::string(name) + ";\n} }",
                    "invalid_dtype.tepl");
    assert(parsed.ok());
    const auto& syntax =
        *parsed.program->dialects[0].operations[0].dtype_policy;
    assert(syntax.name == name && syntax.span.begin.line == 2);
    assert(syntax.span.begin.column == 10);
    const auto checked = tepl::core::analyze(*parsed.program);
    assert(!checked.ok() && !checked.program);
    assert(checked.diagnostics.size() == 1);
    const auto& diagnostic = checked.diagnostics[0];
    assert(diagnostic.message ==
           "unknown operation dtype policy '" + std::string(name) + "'");
    assert(diagnostic.origin.definition.source_name == "invalid_dtype.tepl");
    assert(diagnostic.origin.definition.span.begin.line == 2);
    assert(diagnostic.origin.definition.span.begin.column == 10);
  }
}

}  // namespace

int main() {
  policies();
  optionalAndContextual();
  propertyOrders();
  errors();
}
