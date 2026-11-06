#include "src/core/resolution/loading.h"

#include <cassert>
#include <filesystem>
#include <map>
#include <string>

#include "src/core/pipeline.h"
#include "src/core/syntax/parse.h"
#include "src/support/project_config.h"

namespace {
struct MemoryLoader : tepl::SourceLoader {
  std::map<std::string, std::string> files;
  std::map<std::string, std::size_t> reads;
  std::optional<std::string> read(const std::string& name) override {
    ++reads[name];
    if (const auto found = files.find(name); found != files.end())
      return found->second;
    return {};
  }
  std::optional<std::string> findProjectConfig(
      const std::string& source) override {
    auto directory = std::filesystem::path(source).parent_path();
    for (;;) {
      const auto config = (directory / "tepl.toml").string();
      if (files.contains(config)) return config;
      const auto parent = directory.parent_path();
      if (parent == directory) return {};
      directory = parent;
    }
  }
};
void configParsing() {
  using tepl::support::parseProjectConfig;
  const auto empty = parseProjectConfig({"/virtual/tepl.toml", ""});
  assert(empty.config && empty.config->root == "/virtual");
  assert(!empty.config->tepl_version && empty.diagnostics.empty());
  assert(empty.config->sources.dialects ==
         std::vector<std::string>{"dialects"});
  assert(empty.config->sources.rules == std::vector<std::string>{"rules"});
  assert(empty.config->sources.graphs == std::vector<std::string>{"graphs"});
  assert(empty.config->sources.exclude.empty());
  assert(empty.config->semantic_policy == "strict");
  const auto configured = parseProjectConfig({"/virtual/tepl.toml", R"(
    [sources]
    dialects = ["schema", "shared/schema"]
    rules = []
    exclude = ["**/fixtures/**"]
    [check]
    semantic-policy = "algebraic"
  )"});
  assert(configured.config);
  assert((configured.config->sources.dialects ==
          std::vector<std::string>{"schema", "shared/schema"}));
  assert(configured.config->sources.rules.empty());
  assert(configured.config->sources.graphs ==
         std::vector<std::string>{"graphs"});
  assert(configured.config->semantic_policy == "algebraic");
  for (const auto& text :
       {"sources = []", "check = true", "[sources]\nrules = 'rules'",
        "[sources]\ndialects = [1]", "[sources]\ngraphs = ['']",
        "[sources]\nexclude = [false]", "[sources]\nrule = []",
        "[check]\nsemantic-policy = 'fast'", "[check]\nsemantic-policy = 1",
        "[check]\nsemantic-polciy = 'strict'", "unknown = true",
        "tepl-version = '0.1.0'"}) {
    const auto bad = parseProjectConfig({"/virtual/tepl.toml", text});
    assert(!bad.config && bad.diagnostics.size() == 1);
    assert(bad.diagnostics[0].origin.definition.source_name ==
           "/virtual/tepl.toml");
  }
  const auto supported = parseProjectConfig(
      {"/virtual/tepl.toml", "# project\ntepl-version = '0.2.0' # compiler\n"});
  assert(supported.config && supported.config->tepl_version == "0.2.0");
  for (const auto& text :
       {"tepl-version = false", "tepl-version = 1", "tepl-version = ['0.2.0']",
        "tepl-version = '99.0.0'", "tepl-version = '>=0.2.0'",
        "tepl-version = '0.2.0'\ntepl-version = '0.2.0'", "invalid toml"}) {
    const auto bad = parseProjectConfig({"/virtual/tepl.toml", text});
    assert(!bad.config && bad.diagnostics.size() == 1);
    assert(bad.diagnostics[0].origin.definition.source_name ==
           "/virtual/tepl.toml");
  }
}
void importResolution() {
  MemoryLoader loader;
  loader.files = {
      {"/virtual/tepl.toml",
       "tepl-version = '0.2.0'\n[check]\nsemantic-policy = 'algebraic'"},
      {"/virtual/dialects/base.tepl",
       "dialect D { op negate(x: tensor) -> tensor }"},
      {"/virtual/dialects/export.tepl", "import \"dialects/base.tepl\""},
      {"/virtual/rules/template.tepl", "abstract rule identity() { X => X }"},
  };
  const tepl::SourceFile source{"/virtual/rules/nested/main.tepl", R"(
    from "dialects/base.tepl" import D as d
    from "../../dialects/./base.tepl" import D as same
    import "dialects/export.tepl"
    from "rules/template.tepl" import {identity}
    rule inherited extends identity()
    rule direct { (d.negate X) => (same.negate X) }
    rule legacy { (negate X) => X }
  )"};
  auto parsed = tepl::parse(source);
  assert(parsed.ok());
  auto loaded = tepl::loadImports(*parsed.program, loader);
  assert(loaded.ok() && loaded.program->modules.size() == 4);
  assert(loaded.config && loaded.config->semantic_policy == "algebraic");
  assert(loader.reads["/virtual/dialects/base.tepl"] == 1);
  assert(loader.reads["/virtual/tepl.toml"] == 1);
  assert(loaded.program->import_targets.at(
             {source.identity, "dialects/base.tepl"}) ==
         "/virtual/dialects/base.tepl");
  assert(parsed.program->imports[0].path == "dialects/base.tepl");
  // Changes after loading cannot change the identities used by resolution.
  loader.files.clear();
  assert(tepl::core::compile(*loaded.program).ok());

  // Different roots in one compilation each retain their own project root.
  loader.files = {
      {"/one/tepl.toml", ""},
      {"/two/tepl.toml", ""},
      {"/one/dialects/d.tepl", "dialect D { op first(x:tensor)->tensor }"},
      {"/two/dialects/d.tepl", "dialect D { op second(x:tensor)->tensor }"},
  };
  auto project = tepl::loadProject(
      std::vector<tepl::SourceFile>{
          {"/one/rules/first.tepl", R"(
        from "dialects/d.tepl" import D as d
        rule first { (d.first X) => X }
      )"},
          {"/two/rules/second.tepl", R"(
        from "dialects/d.tepl" import D as d
        rule second { (d.second X) => X }
      )"},
      },
      loader);
  assert(project.ok() && tepl::core::compile(*project.program).ok());

  // Cycle detection uses resolved identities across relative and root imports.
  loader.files = {
      {"/virtual/tepl.toml", ""},
      {"/virtual/dialects/a.tepl", "import \"dialects/b.tepl\""},
      {"/virtual/dialects/b.tepl", "import \"./a.tepl\""},
  };
  parsed = tepl::parse("import \"dialects/a.tepl\"", source.identity);
  assert(parsed.ok());
  loaded = tepl::loadImports(*parsed.program, loader);
  assert(!loaded.ok() && loaded.diagnostics.size() == 1);
  assert(loaded.diagnostics[0].message.find("cyclic import") !=
         std::string::npos);
}
}  // namespace

int main() {
  configParsing();
  importResolution();
}
