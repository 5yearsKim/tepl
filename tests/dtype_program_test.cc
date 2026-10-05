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
void valid(const std::string& source) {
  auto parsed = tepl::parse(source, "dtype.tepl");
  assert(parsed.ok());
  auto checked = tepl::core::analyze(*parsed.program);
  assert(checked.ok());
}
void invalid(const std::string& source, const std::string& message) {
  auto parsed = tepl::parse(source, "dtype.tepl");
  assert(parsed.ok());
  auto checked = tepl::core::analyze(*parsed.program);
  assert(!checked.ok());
  assert(std::any_of(checked.diagnostics.begin(), checked.diagnostics.end(),
                     [&](const auto& d) {
                       return d.message.find(message) != std::string::npos &&
                              d.origin.definition.source_name == "dtype.tepl";
                     }));
}
void programs() {
  valid(R"(
    dialect D {
      op unknown(x: tensor) -> tensor;
      op equal(x: tensor, y: tensor) -> tensor {
        dtype(a, b) { assert a == b; assert is_numeric(a); yield bool; }
        shape(a, b) { assert a == b; yield a; }
      }
      op cast(x: tensor) -> tensor {
        attrs { to: dtype; choices: dtype[]; }
        dtype(src) {
          let target = attrs.to;
          assert is_float(src) || is_integer(src);
          assert is_signed_integer(i32) && is_unsigned_integer(u8);
          assert contains(attrs.choices, target);
          yield if target == bool then f32 else target;
        }
        shape(s) { yield s; }
      }
      op zero() -> tensor { dtype() { yield f32; } }
      op concat(first: tensor, rest: tensor...) -> tensor {
        dtype(a, tail...) { assert all([t == a for t in tail]); yield a; }
      }
      op variadic(inputs: tensor...) -> tensor {
        dtype(types...) { assert len(types) > 0; yield types[0]; }
      }
    }
  )");
  auto parsed = tepl::parse(
      "dialect D { op zero() -> tensor { dtype() { yield f32; } } }");
  assert(parsed.ok());
  auto checked = tepl::core::analyze(*parsed.program);
  assert(checked.ok());
  const auto& dtype = *checked.program->operations[0].dtype;
  assert(dtype.kind == tepl::core::metadata::ProgramKind::DType);
  assert(std::get<tepl::core::DType>(dtype.result.value->value) ==
         tepl::core::DType::kF32);
  assert(tepl::formatAst(*parsed.program).find("dtype(") != std::string::npos);
  assert(tepl::core::formatProgram(*checked.program).find("DType(f32)") !=
         std::string::npos);
}
void propertyOrders() {
  const std::array<std::string, 3> properties = {
      "alias: renamed;", "attrs { to: dtype; }", "dtype(t) { yield t; }"};
  std::array<int, 3> order = {0, 1, 2};
  do {
    std::string source = "dialect D { op copy(x: tensor) -> tensor { ";
    for (int index : order) source += properties[index] + " ";
    valid(source + "shape(s) { yield s; } } }");
  } while (std::next_permutation(order.begin(), order.end()));
}
void errors() {
  for (const auto* legacy :
       {"same", "same_numeric", "same_float", "f32", "bool"})
    assert(!tepl::parse("dialect D { op copy(x: tensor) -> tensor { dtype: " +
                        std::string(legacy) + "; } }")
                .ok());
  const auto body = [](const std::string& block) {
    return "dialect D { op copy(x: tensor) -> tensor { " + block + " } }";
  };
  invalid(body("dtype() { yield f32; }"), "parameter count");
  invalid(body("dtype(t...) { yield f32; }"), "variadic");
  invalid(body("dtype(t) { yield true; }"), "type mismatch");
  invalid(body("dtype(t) { assert t; yield t; }"), "type mismatch");
  invalid(body("dtype(t) { yield t + t; }"), "type mismatch");
  invalid(body("dtype(t) { assert t < f32; yield t; }"), "type mismatch");
  invalid(body("dtype(t) { yield unknown; }"), "unknown");
  invalid(body("dtype(t) { assert is_float(1); yield t; }"), "type mismatch");
  invalid(body("dtype(t) { yield attrs.to; }"), "no attribute schema");
  invalid(body("dtype(t) { yield t; } dtype(a) { yield a; }"),
          "duplicate operation property");
  invalid(body("alias: a; alias: b;"), "duplicate operation property");
  invalid(body("attrs { to: dtype; } attrs { other: dtype; }"),
          "duplicate operation property");
  invalid(
      "dialect D { attrs S { to: dtype; } op copy(x: tensor) -> tensor { "
      "attrs: S; attrs { other: dtype; } } }",
      "duplicate operation property");
  invalid(body("dtype(f32) { yield f32; }"), "reserved");
  invalid(body("dtype(t) { let f32 = t; yield f32; }"), "reserved");
  invalid(body("dtype(t) { let xs = [f32 for f32 in [t]]; yield xs[0]; }"),
          "reserved");
  invalid(
      "dialect D { op cast(x: tensor) -> tensor { attrs { to: dtype?; } "
      "dtype(t) { yield attrs.to; } } }",
      "optional");
  assert(!tepl::parse(body("dtype(t) { yield t; } shape(s) { yield s; } "
                           "dtype(a) { yield a; }"))
              .ok());
}
void rules() {
  const std::string dialect = R"(
    dialect D {
      op convert(x: tensor) -> tensor { attrs { to: dtype; } dtype(t) { yield attrs.to; } shape(s) { yield s; } }
      op pair(x: tensor, y: tensor) -> tensor { dtype(a,b) { assert a == b; yield a; } shape(a,b) { assert a == b; yield a; } }
    }
  )";
  valid(dialect + R"(
    abstract rule identity() { dtype T; X: T[...] (convert[@c] X) => X where { is_numeric(T); @c.to == T; } }
    rule specialized extends identity() { X: f32[N] }
    rule shared { dtype T; X: T[N] Y: T[N] (pair X Y) => (pair Y X) where { T != bool; } }
  )");
  invalid(dialect + "rule r { X: T[...] X => X }", "unknown dtype");
  invalid(dialect + "rule r { dtype T; X => X }", "not bound");
  invalid(dialect + "rule r { dtype T; dtype T; X: T[...] X => X }",
          "duplicate dtype variable");
  invalid(dialect + "rule r { dtype X; X: X[...] X => X }", "conflicts");
  invalid(dialect + "rule r { dtype T; X: T[T] X => X }", "conflicts");
  invalid(dialect + "rule r { f32 => f32 }", "conflicts");
  invalid(dialect +
              "rule r { X: [...] (convert[@c] X) => X where { @c.missing == "
              "f32; } }",
          "unknown descriptor field");
  invalid(
      dialect +
          "rule r { X: [...] (convert[@c] X) => X where { @c.to == true; } }",
      "type mismatch");
}
}  // namespace
int main() {
  programs();
  propertyOrders();
  errors();
  rules();
}
