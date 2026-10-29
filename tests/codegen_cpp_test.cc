#include <cassert>
#include <iostream>
#include <set>
#include <string>

#include "src/codegen/generate.h"
#include "src/core/pipeline.h"
#include "src/core/syntax/parse.h"

namespace {
tepl::codegen::GenerationResult generate(const std::string& text,
                                         std::string ns = "tepl_generated") {
  auto parsed =
      tepl::parse(text.empty() ? "abstract rule unused() {X=>X}" : text,
                  "rules/nested/example.tepl");
  assert(parsed.ok());
  auto compiled = tepl::core::compile(*parsed.program);
  for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
  assert(compiled.ok());
  tepl::codegen::Options options;
  options.target = tepl::codegen::Target::kCpp;
  options.rules_root = "rules";
  options.cpp_namespace = std::move(ns);
  auto a = tepl::codegen::generate(*compiled.program, options);
  auto b = tepl::codegen::generate(
      tepl::executable::load(tepl::executable::serialize(*compiled.program)),
      options);
  assert(a.ok() == b.ok() && a.files.size() == b.files.size());
  for (std::size_t i = 0; i < a.files.size(); ++i)
    assert(a.files[i].path == b.files[i].path &&
           a.files[i].contents == b.files[i].contents);
  return a;
}
}  // namespace
int main() {
  auto result = generate(
      "dialect T {op copy(x:tensor)->tensor {dtype(t){yield t;}shape(s){yield "
      "s;}}} rule identity {(T.copy X)=>X}");
  assert(result.ok());
  std::set<std::string> paths;
  for (const auto& file : result.files) {
    assert(paths.insert(file.path).second);
    assert(file.contents.find("@TEPL_") == std::string::npos);
  }
  for (auto path :
       {"generated.h", "program.h", "metadata.h", "dialects/t.h",
        "rules/nested/example.h", "rules/nested/rules.h", "graphs/graphs.h"})
    assert(paths.contains(path));
  assert(generate("").ok());
  assert(generate("", "app::optimizer").ok());
  for (auto ns : {"", "class", "std", "std::optimizer", "tepl_runtime",
                  "app::", "::app", "app::_reserved", "a__b", "1bad", "a:b"}) {
    auto bad = generate("", ns);
    assert(!bad.ok() && bad.files.empty());
  }
  for (auto source :
       {"dialect T {attrs A {signed:u64;signed_:u64;}}",
        "dialect T {op foo_bar(x:tensor)->tensor;op fooBar(x:tensor)->tensor;}",
        "dialect T {attrs OpAttrs {x:u64;}}",
        "dialect T {attrs A {to_value:u64;}}"}) {
    auto bad = generate(source);
    assert(!bad.ok() && bad.files.empty());
    assert(!bad.diagnostics.front().origin.definition.source_name.empty());
  }
}
