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
  assert(generated.source.find("pub mod example {") !=
         std::string::npos);
  assert(generated.source.find("pub trait Functions: Send + Sync") !=
         std::string::npos);
  assert(generated.source.find("fn forbidden(&self, arg0: &TensorInfo) "
                               "-> Option<bool>;") != std::string::npos);
  assert(generated.source.find("fn infer_dot(&self, arg0: &TensorInfo, "
                               "arg1: &TensorInfo, arg2: &OpAttrs) "
                               "-> Option<InferredTensor>;") !=
         std::string::npos);

  const auto implementation = tepl::generateHostTemplate(*parsed.program, true);
  assert(implementation.ok());
  assert(implementation.source.find("impl example::Functions for UserFunctions") !=
         std::string::npos);
  assert(implementation.source.find("todo!(\"implement infer_dot\")") !=
         std::string::npos);

  const auto bindings = tepl::parse(R"(
rule lhs_binding {
  let Y = (dot X W) => Y
  where { reusable(Y); }
}
rule rhs_binding {
  X => (let Z = (dot X W))
  derive { @out = infer(Z); }
}
)");
  assert(bindings.ok());
  const auto with_bindings =
      tepl::generateHostTemplate(*bindings.program, false);
  assert(with_bindings.ok());
  assert(with_bindings.source.find("pub mod lhs_binding {") != std::string::npos);
  assert(with_bindings.source.find("pub mod rhs_binding {") != std::string::npos);
  assert(with_bindings.source.find(
             "fn reusable(&self, arg0: &TensorInfo) -> Option<bool>;") !=
         std::string::npos);
  assert(with_bindings.source.find(
             "fn infer(&self, arg0: &TensorInfo) -> Option<InferredTensor>;") !=
         std::string::npos);

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
  assert(!untyped.ok());
  assert(untyped.diagnostics[0].message.find("explicit type") !=
         std::string::npos);
}
