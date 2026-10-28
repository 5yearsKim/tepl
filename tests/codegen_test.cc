#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/codegen/generate.h"
#include "src/codegen/write.h"
#include "src/core/pipeline.h"
#include "src/core/syntax/parse.h"
#include "src/support/temporary_directory.h"
namespace {
tepl::executable::ExecutableProgram compile(const std::string& source) {
  auto parsed = tepl::parse(source, "fixture.tepl");
  assert(parsed.ok());
  auto result = tepl::core::compile(*parsed.program);
  for (const auto& d : result.diagnostics) std::cerr << d.message << '\n';
  assert(result.ok());
  return std::move(*result.program);
}
void equivalentInputs() {
  auto p = compile(R"(
    dialect D {op copy(x:tensor)->tensor {dtype(t){yield t;}shape(s){yield s;}}}
    rule copy {(D.copy X)=>X where {true || 1/0==0}}
    graph g {input X:f32[2,3];yield (D.copy X);}
  )");
  auto a = tepl::codegen::generate(p);
  auto b = tepl::codegen::generate(
      tepl::executable::load(tepl::executable::serialize(p)));
  assert(a.ok() && b.ok() && !a.files.empty());
  assert(a.files.size() == b.files.size());
  for (std::size_t i = 0; i < a.files.size(); ++i) {
    assert(a.files[i].path == b.files[i].path);
    assert(a.files[i].contents == b.files[i].contents);
  }
  tepl::support::TemporaryDirectory directory("tepl-codegen-test-");
  tepl::codegen::WriteOptions options;
  options.format = false;
  tepl::codegen::synchronize(a.files, directory.path(), options);
  options.check = true;
  assert(
      tepl::codegen::synchronize(b.files, directory.path(), options).current());
  a.files.front().contents += "// drift\n";
  assert(!tepl::codegen::synchronize(a.files, directory.path(), options)
              .current());
  assert(
      tepl::codegen::synchronize(b.files, directory.path(), options).current());
  tepl::codegen::Options target;
  target.target = tepl::codegen::Target::kCpp;
  auto unsupported = tepl::codegen::generate(p, target);
  assert(!unsupported.ok() && unsupported.files.empty());
}
void namingFailuresAreAtomic() {
  for (const auto& source :
       {"dialect D {attrs Op {x: u64;}op copy(x:tensor)->tensor {attrs:Op;}}",
        "dialect D {op self_(x:tensor)->tensor;}",
        "dialect D {op foo_bar(x:tensor)->tensor;op "
        "FooBar(x:tensor)->tensor;}"}) {
    auto result = tepl::codegen::generate(compile(source));
    assert(!result.ok() && result.files.empty());
  }
}
}  // namespace
int main() {
  try {
    equivalentInputs();
    namingFailuresAreAtomic();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
