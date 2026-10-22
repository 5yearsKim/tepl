#include <cassert>
#include <iostream>
#include <set>
#include <string>

#include "src/codegen/generate.h"
#include "src/core/analyze.h"
#include "src/parse.h"

namespace {
tepl::codegen::GenerationResult generate(const std::string& text,
                                         std::string ns = "tepl_generated") {
  auto parsed =
      tepl::parse(text.empty() ? "abstract rule unused() { X => X }" : text,
                  "rules/nested/example.tepl");
  if (!parsed.ok()) {
    std::cerr << text << '\n';
    for (const auto& diagnostic : parsed.diagnostics)
      std::cerr << diagnostic.message << '\n';
  }
  assert(parsed.ok());
  auto checked = tepl::core::analyze(*parsed.program);
  assert(checked.ok());
  tepl::codegen::Options options;
  options.target = tepl::codegen::Target::kCpp;
  options.rules_root = "rules";
  options.cpp_namespace = std::move(ns);
  return tepl::codegen::generate(*checked.program, options);
}
}  // namespace
int main() {
  auto result = generate(
      "dialect T { op copy(x: tensor) -> tensor { dtype(t0) { yield t0; } "
      "shape(s) { "
      "yield s; } } } rule identity { (copy X) => X }");
  assert(result.ok());
  std::set<std::string> paths;
  for (const auto& file : result.files) {
    assert(paths.insert(file.path).second);
    assert(file.contents.find("TEPL_NAMESPACE.attrs") == std::string::npos);
    assert(file.contents.find("// tepl.attrs:") == std::string::npos);
  }
  for (auto path :
       {"generated.h", "op_node.h", "types.h", "rules/nested/example.h",
        "rules/nested/rules.h", "analysis/shape.h", "analysis/tensor_info.h",
        "rewriting/rewrite.h", "rewriting/rewrite_analysis.h", "dialects/t.h"})
    assert(paths.contains(path));
  auto guarded = generate(
      "dialect T { op copy(x: tensor) -> tensor; } "
      "rule guarded { (copy X) => X "
      "where { (true || len(range(18446744073709551615)) == 0) && true } }");
  assert(guarded.ok());
  bool guard_emitted = false;
  for (const auto& file : guarded.files) {
    if (file.path != "rules/nested/example.h") continue;
    guard_emitted = true;
    assert(file.contents.find("true || ") != std::string::npos);
    assert(file.contents.find(" && true)") != std::string::npos);
    assert(file.contents.find("::common::range(") != std::string::npos);
  }
  assert(guard_emitted);
  assert(generate("").ok());
  assert(generate("", "app::optimizer").ok());
  for (auto ns : {"", "class", "std", "std::optimizer", "app::", "::app",
                  "app::_reserved", "a__b", "1bad", "a:b"}) {
    auto bad = generate("", ns);
    assert(!bad.ok() && bad.files.empty());
  }
  for (auto source : {"dialect T { attrs A { class: index; } op copy(x: "
                      "tensor) -> tensor { attrs: A; } }",
                      "dialect T { op foo_bar(x: tensor) -> tensor; op "
                      "fooBar(x: tensor) -> tensor; }",
                      "dialect T { attrs OpAttrs { x: index; } }",
                      "dialect T { attrs A { hash_value: index; } }"}) {
    auto bad = generate(source);
    assert(!bad.ok() && bad.files.empty());
    assert(!bad.diagnostics.front().origin.definition.source_name.empty());
  }
}
