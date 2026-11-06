#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_verifier.h"
namespace xla {
void SetAlgebraicRewriteCounter(void (*counter)(int, const char*));
namespace {
std::mutex counter_mutex;
std::map<std::pair<std::string, int>, int64_t> counts;
int64_t pass_invocations = 0;
void count(int line, const char* function) {
  std::lock_guard<std::mutex> lock(counter_mutex);
  if (!line)
    ++pass_invocations;
  else
    ++counts[{function, line}];
}
template <class T>
T take(absl::StatusOr<T> value) {
  if (!value.ok()) throw std::runtime_error(value.status().ToString());
  return std::move(value).value();
}
void require(bool good, const std::string& message) {
  if (!good) throw std::runtime_error(message);
}
std::string read(const char* path) {
  std::ifstream input(path);
  require(bool(input), std::string("Missing ") + path);
  return {std::istreambuf_iterator<char>(input), {}};
}
std::string canonical(const HloModule& m) {
  return m.ToString(HloPrintOptions::Canonical()
                        .set_print_large_constants(true)
                        .set_print_only_essential_constants(false)
                        .set_print_ids(false)
                        .set_print_backend_config(true)
                        .set_print_control_dependencies(true)
                        .set_canonicalize_computations(true));
}
int64_t total() {
  int64_t result = 0;
  for (const auto& entry : counts) result += entry.second;
  return result;
}
void selfTest() {
  const char* positive = R"(HloModule counted_add
ENTRY main {
 a = f32[2] parameter(0)
 z = f32[] constant(0)
 zs = f32[2] broadcast(z), dimensions={}
 x = f32[2] add(a,zs)
 y = f32[2] add(a,zs)
 ROOT result = (f32[2],f32[2]) tuple(x,y)
})";
  const char* negative = R"(HloModule unchanged_add
ENTRY main {
 a = f32[2] parameter(0)
 b = f32[2] parameter(1)
 ROOT result = f32[2] add(a,b)
})";
  SetAlgebraicRewriteCounter(count);
  for (int control = 0; control < 2; ++control) {
    counts.clear();
    pass_invocations = 0;
    auto module =
        take(ParseAndReturnUnverifiedModule(control ? negative : positive));
    take(HloVerifier(false, false).Run(module.get()));
    bool changed = take(
        AlgebraicSimplifier(AlgebraicSimplifierOptions{}).Run(module.get()));
    require(changed == !control, "Unexpected control changed flag");
    require(total() == (control ? 0 : 2),
            "Incorrect successful application count");
    require(counts.size() == (control ? 0 : 1),
            "Incorrect distinct rewrite-site count");
    require(pass_invocations == 1, "Incorrect pass invocation count");
    take(HloVerifier(false, false).Run(module.get()));
  }
  counts.clear();
  pass_invocations = 0;
  SetAlgebraicRewriteCounter(nullptr);
  std::cout << "{\"controls\":2,\"repeated_rule_applications\":2,\"distinct_"
               "rule_sites\":1,\"no_match_applications\":0}\n";
}
int run(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    selfTest();
    return 0;
  }
  require(argc == 3, "Expected input.hlo output_directory");
  std::filesystem::path out(argv[2]);
  std::filesystem::create_directories(out);
  auto module = take(ParseAndReturnUnverifiedModule(read(argv[1])));
  take(HloVerifier(false, false).Run(module.get()));
  module->mutable_config().set_replica_count(1);
  module->mutable_config().set_num_partitions(1);
  std::ofstream(out / "debug-options.pbtxt")
      << module->config().debug_options().DebugString();
  CpuClientOptions options;
  options.asynchronous = false;
  options.cpu_device_count = 1;
  options.max_transpose_threads = 1;
  HloRunner runner(take(GetXlaPjrtCpuClient(std::move(options))));
  SetAlgebraicRewriteCounter(count);
  auto executable =
      take(runner.CreateExecutable(module->Clone("workflow"), true));
  SetAlgebraicRewriteCounter(nullptr);
  require(pass_invocations > 0, "AlgebraicSimplifier was not invoked");
  auto* final = take(runner.HloModuleFromWrapped(executable.get()));
  std::ofstream(out / "final.hlo") << canonical(*final);
  llvm::json::Array rows;
  for (const auto& entry : counts)
    rows.push_back(llvm::json::Object{{"function", entry.first.first},
                                      {"original_line", entry.first.second},
                                      {"applications", entry.second}});
  llvm::json::Object result{
      {"input", std::string(argv[1])},
      {"scope",
       "All AlgebraicSimplifier invocations in one normal XLA CPU compilation"},
      {"rule_identity",
       "Outermost successful transformation branch, identified by function and "
       "original source line; equivalent branches remain separate variants"},
      {"model_compilations", 1},
      {"pass_invocations", pass_invocations},
      {"distinct_rule_sites", int64_t(counts.size())},
      {"total_applications", total()},
      {"rules", std::move(rows)}};
  std::cout
      << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result))).str();
  return 0;
}
}  // namespace
}  // namespace xla
int main(int argc, char** argv) {
  try {
    return xla::run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
