#include <cassert>
#include <string>

#include "src/codegen/common/rule_plan.h"
#include "src/codegen/generate.h"
#include "src/core/analyze.h"
#include "src/parse.h"

namespace {
tepl::core::Program checkedProgram() {
  auto parsed = tepl::parse(R"(
dialect Custom { op negate(input: tensor) -> tensor; }
rule test {
  X: [N]
  (negate X) => X
  where { outer(inner(N) + 1, X); }
}
)");
  assert(parsed.ok());
  auto checked = tepl::core::analyze(*parsed.program);
  assert(checked.ok());
  return std::move(*checked.program);
}

// A future language backend can implement the exact same public interface.
class PythonBackend final : public tepl::codegen::Generator {
 public:
  tepl::codegen::Target target() const override {
    return tepl::codegen::Target::kPython;
  }
  tepl::codegen::GenerationResult generate(
      const tepl::core::Program& program,
      const tepl::codegen::Options&) const override {
    tepl::codegen::GenerationResult result;
    result.files.push_back(
        {"rules.py",
         "rule_count = " + std::to_string(program.rules.size()) + "\n"});
    return result;
  }
};
}  // namespace

int main() {
  const auto program = checkedProgram();  // The source AST has been destroyed.
  const auto plan = tepl::codegen::planRule(program.rules.front());
  assert(plan.host_functions.size() == 2);  // Includes the nested call.
  assert(plan.metadata_captures.size() == 1);
  assert(plan.metadata_captures.front() ==
         program.rules.front().captures.front().id);
  const auto generated = tepl::codegen::generate(program);
  assert(generated.ok() && !generated.files.empty());
  const auto repeated = tepl::codegen::generate(program);
  assert(repeated.ok() && repeated.files.size() == generated.files.size());
  for (std::size_t i = 0; i < generated.files.size(); ++i) {
    assert(generated.files[i].path == repeated.files[i].path);
    assert(generated.files[i].contents == repeated.files[i].contents);
  }
  for (const auto target :
       {tepl::codegen::Target::kCpp, tepl::codegen::Target::kPython}) {
    tepl::codegen::Options options;
    options.target = target;
    const auto unavailable = tepl::codegen::generate(program, options);
    assert(!unavailable.ok() && unavailable.files.empty());
    assert(unavailable.diagnostics.front().message.find("not implemented") !=
           std::string::npos);
  }
  tepl::codegen::Options options;
  options.package_name = "bad\"\n[dependencies]";
  const auto invalid = tepl::codegen::generate(program, options);
  assert(!invalid.ok() && invalid.files.empty());
  for (const auto* source :
       {"dialect Collision { op foo_bar(x: tensor) -> tensor; op fooBar(x: "
        "tensor) -> tensor; }",
        "dialect None { op add(x: tensor, y: tensor) -> tensor; }"}) {
    auto parsed_collision = tepl::parse(source);
    assert(parsed_collision.ok());
    auto checked_collision = tepl::core::analyze(*parsed_collision.program);
    assert(checked_collision.ok());
    auto collision = tepl::codegen::generate(*checked_collision.program);
    assert(!collision.ok() && collision.files.empty());
    assert(collision.diagnostics.front().message.find("collision") !=
           std::string::npos);
  }
  const PythonBackend python;
  assert(python.generate(program, {}).files.front().path == "rules.py");
}
