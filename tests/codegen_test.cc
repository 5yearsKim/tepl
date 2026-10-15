#include <cassert>
#include <stdexcept>
#include <string>

#include "src/codegen/common/project_plan.h"
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
  where { sum(range(N + 1)) >= 0 }
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
      const tepl::program::Program& program,
      const tepl::codegen::Options&) const override {
    tepl::codegen::GenerationResult result;
    result.files.push_back(
        {"rules.py",
         "module_count = " +
             std::to_string(
                 tepl::codegen::planProject(program.ir()).modules.size()) +
             "\n"});
    return result;
  }
};
}  // namespace

int main() {
  const auto program = checkedProgram();  // The source AST has been destroyed.
  const auto plan = tepl::codegen::planRule(program.rules.front());
  assert(plan.early_conditions.size() == 1);
  const auto generated = tepl::codegen::generate(program);
  assert(generated.ok() && !generated.files.empty());
  bool found_builtins = false;
  for (const auto& file : generated.files) {
    assert(file.contents.find("#[test]") == std::string::npos);
    assert(file.contents.find("#[cfg(test)]") == std::string::npos);
    assert(file.contents.find("SEARCH_VISITS") == std::string::npos);
    assert(file.contents.find("use super::*") == std::string::npos);
    assert(file.contents.find("::dialects::*") == std::string::npos);
    assert(file.contents.find("::rewriting::*") == std::string::npos);
    assert(file.contents.find("fn $") == std::string::npos);
    assert(file.contents.find("functions.$") == std::string::npos);
    assert(file.contents.find("HostFunctions") == std::string::npos);
    if (file.path != "rules/input.rs") continue;
    found_builtins = true;
    assert(file.contents.find("::builtins::common::sum(") != std::string::npos);
    assert(file.contents.find("::builtins::common::add(") != std::string::npos);
  }
  assert(found_builtins);
  const auto repeated = tepl::codegen::generate(program);
  assert(repeated.ok() && repeated.files.size() == generated.files.size());
  for (std::size_t i = 0; i < generated.files.size(); ++i) {
    assert(generated.files[i].path == repeated.files[i].path);
    assert(generated.files[i].contents == repeated.files[i].contents);
  }
  for (const auto& file : generated.files) {
    assert(file.path != "Cargo.toml" && file.path != "lib.rs");
    assert(!file.path.starts_with("src/"));
    assert(file.contents.find("crate::ir") == std::string::npos);
  }
  auto project = program;
  project.rules[0].source_name = "project/rules/one.tepl";
  auto second = project.rules[0];
  second.id = tepl::core::RuleId{1};
  second.source_name = "project/rules/nested/two.tepl";
  project.rules.push_back(second);
  auto project_plan = tepl::codegen::planProject(project, "project/rules");
  assert(project_plan.modules.size() == 2);
  assert(project_plan.dialects.size() == 1);
  assert(project_plan.dialects[0].operations.size() == 1);
  assert(project_plan.modules[0].path ==
         std::vector<std::string>({"nested", "two"}));
  assert(project_plan.modules[0].rules[0] == second.id);
  tepl::codegen::Options project_options;
  project_options.rules_root = "project/rules";
  const auto project_files = tepl::codegen::generate(project, project_options);
  assert(project_files.ok());
  bool found_nested = false;
  for (const auto& file : project_files.files) {
    if (file.path != "rules/nested/two.rs") continue;
    found_nested = true;
    assert(file.contents.find("use super::super::super::super::rewriting::{") !=
           std::string::npos);
  }
  assert(found_nested);
  bool rejected_source = false;
  try {
    tepl::codegen::planProject(project, "another/rules");
  } catch (const std::invalid_argument&) {
    rejected_source = true;
  }
  assert(rejected_source);
  for (const auto target : {tepl::codegen::Target::kPython}) {
    tepl::codegen::Options options;
    options.target = target;
    const auto unavailable = tepl::codegen::generate(program, options);
    assert(!unavailable.ok() && unavailable.files.empty());
    assert(unavailable.diagnostics.front().message.find("not implemented") !=
           std::string::npos);
  }
  for (const auto* source :
       {"dialect Collision { op foo_bar(x: tensor) -> tensor; op fooBar(x: "
        "tensor) -> tensor; }",
        "dialect None { op add(x: tensor, y: tensor) -> tensor; }",
        "dialect Input { op add(x: tensor, y: tensor) -> tensor; }"}) {
    auto parsed_collision = tepl::parse(source);
    assert(parsed_collision.ok());
    auto checked_collision = tepl::core::analyze(*parsed_collision.program);
    assert(checked_collision.ok());
    auto collision = tepl::codegen::generate(*checked_collision.program);
    assert(!collision.ok() && collision.files.empty());
    assert(collision.diagnostics.front().message.find("collision") !=
           std::string::npos);
  }
  // These source names are valid TEPL but have no Rust raw-identifier spelling.
  for (const auto* name : {"self", "Self", "super", "crate"}) {
    const std::string field_source =
        "dialect D { attrs A { " + std::string(name) +
        ": index; } op node() -> tensor { attrs: A; } }";
    for (const auto& source : {field_source}) {
      auto parsed = tepl::parse(source);
      assert(parsed.ok());
      auto checked = tepl::core::analyze(*parsed.program);
      assert(checked.ok());
      const auto rejected = tepl::codegen::generate(*checked.program);
      assert(!rejected.ok() && rejected.files.empty());
      assert(
          !rejected.diagnostics.front().origin.definition.source_name.empty());
      assert(rejected.diagnostics.front().message.find(
                 "does not allow r#" + std::string(name)) != std::string::npos);
    }
  }
  // Validation runs after target normalization, before creating output.
  for (const auto* source :
       {"dialect self_ { op copy(x: tensor) -> tensor; }",
        "dialect D { op self_(x: tensor) -> tensor; }",
        "dialect D { attrs self_ { size: index; } op copy(x: tensor) -> "
        "tensor; }",
        "dialect Mod { op copy(x: tensor) -> tensor; }",
        "dialect FooBar { op copy(x: tensor) -> tensor; } dialect foo_bar { op "
        "other(x: tensor) -> tensor; }",
        "dialect D { attrs None { size: index; } op copy(x: tensor) -> tensor; "
        "}"}) {
    auto parsed = tepl::parse(source);
    assert(parsed.ok());
    auto checked = tepl::core::analyze(*parsed.program);
    assert(checked.ok());
    const auto rejected = tepl::codegen::generate(*checked.program);
    assert(!rejected.ok() && rejected.files.empty());
    assert(!rejected.diagnostics.front().origin.definition.source_name.empty());
  }
  for (const auto* source :
       {"project/rules/self.tepl", "project/rules/mod.tepl"}) {
    auto invalid = program;
    invalid.rules[0].source_name = source;
    const auto rejected = tepl::codegen::generate(invalid, project_options);
    assert(!rejected.ok() && rejected.files.empty());
  }
  for (const auto* second_source :
       {"project/rules/foo_bar.tepl", "project/rules/FooBar/child.tepl"}) {
    auto invalid = project;
    invalid.rules[0].source_name = "project/rules/FooBar.tepl";
    invalid.rules[1].source_name = second_source;
    const auto rejected = tepl::codegen::generate(invalid, project_options);
    assert(!rejected.ok() && rejected.files.empty());
    assert(rejected.diagnostics.front().message.find("collision") !=
           std::string::npos);
  }
  // Parent directory normalization is checked even when child filenames differ.
  auto directory_collision = project;
  directory_collision.rules[0].source_name = "project/rules/FooBar/one.tepl";
  directory_collision.rules[1].source_name = "project/rules/foo_bar/two.tepl";
  assert(!tepl::codegen::generate(directory_collision, project_options).ok());
  auto keywords = program;
  keywords.rules[0].source_name = "project/rules/type/match.tepl";
  const auto keyword_output =
      tepl::codegen::generate(keywords, project_options);
  assert(keyword_output.ok());
  bool keyword_index = false, keyword_file = false;
  for (const auto& file : keyword_output.files) {
    if (file.path == "rules/type/mod.rs") {
      keyword_index =
          file.contents.find("pub mod r#match;") != std::string::npos;
    }
    if (file.path == "rules/type/match.rs") keyword_file = true;
    assert(file.path.find("r#") == std::string::npos);
  }
  assert(keyword_index && keyword_file);
  auto where_parsed = tepl::parse(R"(
    dialect D { op copy(x: tensor) -> tensor; }
    rule early {
      X: [Batch..., N]
      (copy X) => X
      where { len(Batch) < 4 && N > 0 && (true || N == 1) && N < 128 }
    }
  )");
  assert(where_parsed.ok());
  auto where_checked = tepl::core::analyze(*where_parsed.program);
  assert(where_checked.ok());
  const auto where_plan =
      tepl::codegen::planRule(where_checked.program->rules[0]);
  assert(where_plan.early_conditions.size() == 1);
  assert(where_plan.early_conditions[0].dimensions.size() == 2);
  assert(where_checked.program->rules[0]
             .dimensions[where_plan.early_conditions[0].dimensions[0].value]
             .sequence);
  assert(!where_checked.program->rules[0]
              .dimensions[where_plan.early_conditions[0].dimensions[1].value]
              .sequence);
  const auto where_output = tepl::codegen::generate(*where_checked.program);
  assert(where_output.ok());
  bool where_emitted = false;
  for (const auto& file : where_output.files) {
    if (file.path != "rules/input.rs") continue;
    where_emitted = true;
    assert(file.contents.find("fn condition_0") != std::string::npos);
    assert(file.contents.find("fn condition_1") == std::string::npos);
    assert(file.contents.find(" && ") != std::string::npos);
    assert(file.contents.find(" || ") != std::string::npos);
    assert(file.contents.find("MatchBinding::Sequence(") != std::string::npos);
    assert(file.contents.find("MatchBinding::Dimension(") != std::string::npos);
  }
  assert(where_emitted);
  // Both backends reject deferred expressions before producing any files.
  for (const auto* source : {"dialect D { op copy(x: tensor) -> tensor; } rule "
                             "r { (copy X) => X where { let n = 1; n > 0 } }",
                             "dialect D { op copy(x: tensor) -> tensor { attrs "
                             "{ axes: index[]; } } } rule r { (copy[@a] X) => "
                             "(copy[@b] X) derive { @b = { ..@a }; } }"}) {
    auto parsed = tepl::parse(source);
    assert(parsed.ok());
    auto checked = tepl::core::analyze(*parsed.program);
    assert(checked.ok());
    for (auto target :
         {tepl::codegen::Target::kRust, tepl::codegen::Target::kCpp}) {
      tepl::codegen::Options options;
      options.target = target;
      auto result = tepl::codegen::generate(*checked.program, options);
      assert(!result.ok() && result.files.empty());
      assert(result.diagnostics[0].message.find("not implemented") !=
             std::string::npos);
    }
  }
  auto anonymous_parsed = tepl::parse(
      "dialect D { op copy(x: tensor) -> tensor { attrs { n: index; } } } "
      "rule r { (copy X) => X }");
  assert(anonymous_parsed.ok());
  auto anonymous_checked = tepl::core::analyze(*anonymous_parsed.program);
  assert(anonymous_checked.ok());
  for (auto target :
       {tepl::codegen::Target::kRust, tepl::codegen::Target::kCpp}) {
    tepl::codegen::Options options;
    options.target = target;
    assert(tepl::codegen::generate(*anonymous_checked.program, options).ok());
  }
  const PythonBackend python;
  assert(
      python.generate(tepl::program::compile(program), {}).files.front().path ==
      "rules.py");
}
