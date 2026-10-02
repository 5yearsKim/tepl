#include "src/host_codegen.h"

#include <cassert>
#include <string>

#include "src/parse.h"

int main() {
  const auto parsed = tepl::parse(R"(
rule example {
  X: [Batch..., M]
  Y: [Batch..., M]
  (add X Y) => (add X Y)
  where { broadcastable(Batch, Batch) && !forbidden(X); M == M; }
  derive { @out = infer_dot(X, Y, @source); }
}
)");
  assert(parsed.ok());
  const auto generated = tepl::generateHostTemplate(*parsed.program, false);
  assert(generated.ok());
  assert(generated.source.find("fn broadcastable(&self, arg0: &[usize], "
                               "arg1: &[usize]) -> Option<bool>;") !=
         std::string::npos);
  assert(generated.source.find("pub mod example {") != std::string::npos);
  assert(generated.source.find("pub trait Functions: Send + Sync") !=
         std::string::npos);
  assert(generated.source.find("fn forbidden(&self, arg0: &TensorInfo) "
                               "-> Option<bool>;") != std::string::npos);
  assert(generated.source.find("fn infer_dot(&self, arg0: &TensorInfo, "
                               "arg1: &TensorInfo, arg2: &OpAttrs) "
                               "-> Option<OpAttrs>;") != std::string::npos);

  const auto implementation = tepl::generateHostTemplate(*parsed.program, true);
  assert(implementation.ok());
  assert(implementation.source.find(
             "impl example::Functions for UserFunctions") != std::string::npos);
  assert(implementation.source.find("todo!(\"implement infer_dot\")") !=
         std::string::npos);

  const auto bindings = tepl::parse(R"(
rule lhs_binding {
  let Y = (dot X W) => Y
  where { reusable(Y); }
}
rule derived_from_binding {
  let Z = (dot X W) => Z
  derive { @out = infer(Z); }
}
)");
  assert(bindings.ok());
  const auto with_bindings =
      tepl::generateHostTemplate(*bindings.program, false);
  assert(with_bindings.ok());
  assert(with_bindings.source.find("pub mod lhs_binding {") !=
         std::string::npos);
  assert(with_bindings.source.find("pub mod derived_from_binding {") !=
         std::string::npos);
  assert(with_bindings.source.find(
             "fn reusable(&self, arg0: &TensorInfo) -> Option<bool>;") !=
         std::string::npos);
  assert(with_bindings.source.find(
             "fn infer(&self, arg0: &TensorInfo) -> Option<OpAttrs>;") !=
         std::string::npos);

  const auto captures = tepl::parse(R"(
rule captures {
  (add X (let Z = (negate Y))) => (add Z X)
  where { allowed(X, Y, Z); }
}
rule bound_capture {
  let Y = (negate X) => Y
  derive { @d = infer(Y, X); }
}
)");
  assert(captures.ok());
  const auto capture_host =
      tepl::generateHostTemplate(*captures.program, false);
  assert(capture_host.ok());
  assert(capture_host.source.find(
             "fn allowed(&self, arg0: &TensorInfo, arg1: &TensorInfo, "
             "arg2: &TensorInfo) -> Option<bool>;") != std::string::npos);
  assert(capture_host.source.find(
             "fn infer(&self, arg0: &TensorInfo, arg1: &TensorInfo) "
             "-> Option<OpAttrs>;") != std::string::npos);

  for (const auto* source :
       {"rule r { X => X where { check(Y); } }",
        "rule r { X => (add X Y) derive { @d = infer(Y); } }"}) {
    const auto unknown = tepl::parse(source);
    assert(unknown.ok());
    const auto unknown_host =
        tepl::generateHostTemplate(*unknown.program, false);
    assert(!unknown_host.ok());
    assert(unknown_host.diagnostics[0].message.find(
               "unknown host argument 'Y'") != std::string::npos);
  }

  const auto descriptor_only =
      tepl::parse("rule r { X => X derive { @d = descriptor(); } }");
  assert(descriptor_only.ok());
  const auto descriptor_host =
      tepl::generateHostTemplate(*descriptor_only.program, false);
  assert(descriptor_host.ok());
  assert(descriptor_host.source.find("use rust_egg::ir::OpAttrs;") !=
         std::string::npos);
  assert(descriptor_host.source.find(
             "fn descriptor(&self) -> Option<OpAttrs>;") != std::string::npos);

  const auto conflicting = tepl::parse(R"(
rule example {
  X: [Batch...]
  (add X X) => (add X X)
  where { check(X); check(Batch); }
}
)");
  assert(conflicting.ok());
  const auto rejected = tepl::generateHostTemplate(*conflicting.program, false);
  assert(!rejected.ok());
  assert(rejected.diagnostics[0].message.find("conflicting signatures") !=
         std::string::npos);

  const auto scalar = tepl::parse(R"(
rule example {
  S: scalar
  S => S
  where { check(S); }
}
)");
  assert(scalar.ok());
  const auto untyped = tepl::generateHostTemplate(*scalar.program, false);
  assert(untyped.ok());
  assert(untyped.source.find("fn check(&self, arg0: &TensorInfo)") !=
         std::string::npos);

  const auto numbers = tepl::parse(R"(
rule numbers {
  X => 1.0
  where { check(1, 1.0, -0.5, -1); threshold() + 0.25 < 1.0; }
  derive { @out = infer(+1.5); }
}
)");
  assert(numbers.ok());
  const auto numeric_host = tepl::generateHostTemplate(*numbers.program, false);
  assert(numeric_host.ok());
  assert(numeric_host.source.find("fn check(&self, arg0: usize, arg1: f64, "
                                  "arg2: f64, arg3: i64) -> Option<bool>;") !=
         std::string::npos);
  assert(numeric_host.source.find("fn threshold(&self) -> Option<f64>;") !=
         std::string::npos);
  assert(numeric_host.source.find(
             "fn infer(&self, arg0: f64) -> Option<OpAttrs>;") !=
         std::string::npos);

  for (const auto* source :
       {"abstract rule r(F: op<(tensor) -> tensor>) { (F X) => X }",
        "rule r extends base(F = t.add);"}) {
    const auto template_rule = tepl::parse(source);
    assert(template_rule.ok());
    const auto unsupported =
        tepl::generateHostTemplate(*template_rule.program, false);
    assert(!unsupported.ok() && unsupported.source.empty());
    assert(unsupported.diagnostics[0].message.find("inheritance expansion") !=
           std::string::npos);
  }
}
