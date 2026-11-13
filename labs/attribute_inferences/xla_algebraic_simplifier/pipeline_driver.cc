#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "research/tepl_algebraic/branch_builder.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_comparison.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape_util.h"

namespace xla {
void SetTeplAlgebraicHook(absl::StatusOr<HloInstruction*> (*)(HloInstruction*,
                                                              int));
void SetTeplBranchHook(absl::StatusOr<HloInstruction*> (*)(
    HloInstruction*, int, absl::Span<HloInstruction* const>,
    absl::Span<const int64_t>, absl::Span<const int64_t>));
void SetAlgebraicRewriteCounter(void (*)(int, const char*));
namespace {
template <typename T>
T Take(absl::StatusOr<T> value) {
  if (!value.ok()) throw std::runtime_error(value.status().ToString());
  return std::move(value).value();
}
void Require(bool value, const std::string& reason) {
  if (!value) throw std::runtime_error(reason);
}
std::string Read(const char* path) {
  std::ifstream stream(path);
  Require(bool(stream), "missing HLO input");
  return {std::istreambuf_iterator<char>(stream), {}};
}
std::string Canonical(const HloModule& module) {
  return module.ToString(HloPrintOptions::Canonical()
                             .set_print_large_constants(true)
                             .set_print_only_essential_constants(false)
                             .set_print_ids(false)
                             .set_print_metadata(false)
                             .set_print_backend_config(true)
                             .set_print_control_dependencies(true)
                             .set_canonicalize_computations(true));
}
std::recursive_mutex mutex;
TeplBranchAdapter* adapter = nullptr;
std::map<std::pair<int, std::string>, int64_t>* counts = nullptr;
int64_t invocations = 0;
void Counter(int site, const char* function) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (!site)
    ++invocations;
  else if (counts)
    ++(*counts)[{site, function}];
  if (site && adapter) adapter->Observe(site);
}
absl::StatusOr<HloInstruction*> Branch(
    HloInstruction* root, int site, absl::Span<HloInstruction* const> captures,
    absl::Span<const int64_t> first, absl::Span<const int64_t> second) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  return adapter->Propose(root, site, captures, first, second);
}
absl::StatusOr<HloInstruction*> Transpose(HloInstruction* root, int site) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  return adapter->Transpose(root, site);
}
void ResetHooks() {
  SetTeplAlgebraicHook(nullptr);
  SetTeplBranchHook(nullptr);
  SetAlgebraicRewriteCounter(nullptr);
  adapter = nullptr;
  counts = nullptr;
}
std::vector<Literal> Arguments(const HloModule& module, int seed) {
  std::vector<Literal> arguments;
  for (const HloInstruction* parameter :
       module.entry_computation()->parameter_instructions()) {
    const auto& shape = parameter->shape();
    Require(shape.IsArray(), "tuple parameters need an input adapter");
    Literal literal = Literal::CreateFromShape(shape);
    auto number = [&](absl::Span<const int64_t> at) {
      int64_t value = seed + parameter->parameter_number() * 7;
      for (int64_t index : at) value = (value * 17 + index) % 997;
      return float(1 + value % 97) / 10000.f;
    };
    absl::Status status;
    switch (shape.element_type()) {
      case F32:
        status = literal.Populate<float>(number);
        break;
      case BF16:
        status = literal.Populate<bfloat16>(
            [&](auto at) { return bfloat16(number(at)); });
        break;
      case F64:
        status = literal.Populate<double>(number);
        break;
      case S32:
        status = literal.Populate<int32_t>([](auto) { return int32_t(1); });
        break;
      case S64:
        status = literal.Populate<int64_t>([](auto) { return int64_t(1); });
        break;
      case PRED:
        status = literal.Populate<bool>([](auto) { return false; });
        break;
      default:
        throw std::runtime_error("unsupported workload input dtype");
    }
    Require(status.ok(), status.ToString());
    arguments.push_back(std::move(literal));
  }
  return arguments;
}
bool Finite(const Literal& literal) {
  bool finite = true;
  ShapeUtil::ForEachSubshape(
      literal.shape(), [&](const Shape& shape, const ShapeIndex& at) {
        if (!shape.IsArray()) return;
        if (shape.element_type() == F32)
          for (float value : literal.data<float>(at))
            finite &= std::isfinite(value);
        if (shape.element_type() == BF16)
          for (bfloat16 value : literal.data<bfloat16>(at))
            finite &= std::isfinite(float(value));
        if (shape.element_type() == F64)
          for (double value : literal.data<double>(at))
            finite &= std::isfinite(value);
      });
  return finite;
}
llvm::json::Object InitializeShortforms(HloModule* module,
                                        const std::string& text, bool finite) {
  // The retained dumps omit large literal payloads. Keep explicit constants
  // untouched and record this input policy independently of the rewrite hooks.
  const std::regex pattern(
      R"((?:^|\n)[ \t]*(?:ROOT[ \t]+)?%?([^ \t=]+)[ \t]*=[^\n]*constant\(\{\.\.\.\}\))");
  std::set<std::string> names;
  int64_t placeholders = 0;
  for (auto match = std::sregex_iterator(text.begin(), text.end(), pattern);
       match != std::sregex_iterator(); ++match) {
    names.insert((*match)[1].str());
    ++placeholders;
  }
  int64_t found = 0, initialized = 0, indices = 0;
  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->instructions()) {
      if (!instruction->IsConstant() ||
          !names.count(std::string(instruction->name())))
        continue;
      ++found;
      if (!finite) continue;
      uint32_t seed = 0;
      for (unsigned char c : instruction->name()) seed = seed * 31 + c;
      auto number = [&](absl::Span<const int64_t> at) {
        uint64_t value = seed % 997;
        for (int64_t index : at) value = (value * 17 + index) % 997;
        return float(1 + value % 97) / 10000.f;
      };
      auto* literal =
          Cast<HloConstantInstruction>(instruction)->mutable_literal();
      absl::Status status;
      if (literal->shape().element_type() == S32 ||
          literal->shape().element_type() == S64) {
        auto index = [](absl::Span<const int64_t> at) {
          return at.empty() ? int64_t(0) : at.back();
        };
        status = literal->shape().element_type() == S32
                     ? literal->Populate<int32_t>(index)
                     : literal->Populate<int64_t>(index);
        Require(status.ok(), status.ToString());
        ++indices;
        continue;
      }
      switch (literal->shape().element_type()) {
        case F32:
          status = literal->Populate<float>(number);
          break;
        case BF16:
          status = literal->Populate<bfloat16>(
              [&](auto at) { return bfloat16(number(at)); });
          break;
        case F64:
          status = literal->Populate<double>(number);
          break;
        default:
          continue;
      }
      Require(status.ok(), status.ToString());
      ++initialized;
    }
  }
  Require(found == placeholders, "shortform literal identities are ambiguous");
  return llvm::json::Object{
      {"policy", finite ? "finite_inputs_v2" : "xla_parser_default"},
      {"placeholders", placeholders},
      {"floating_literals_initialized", initialized},
      {"integer_literals_initialized", indices}};
}
int Run(int argc, char** argv) {
  Require(argc == 8,
          "expected library.so automatic.json branches.json input.hlo "
          "output_dir strict|hybrid original|finite");
  const bool strict = std::string(argv[6]) == "strict";
  Require(strict || std::string(argv[6]) == "hybrid",
          "choose strict or hybrid explicitly");
  const std::filesystem::path out(argv[5]);
  std::filesystem::create_directories(out);
  const std::string input = Read(argv[4]);
  auto source = Take(ParseAndReturnUnverifiedModule(input));
  Require(
      std::string(argv[7]) == "original" || std::string(argv[7]) == "finite",
      "choose original or finite shortform initialization");
  auto shortforms = InitializeShortforms(source.get(), input,
                                         std::string(argv[7]) == "finite");
  std::ofstream(out / "input-policy.json")
      << llvm::formatv("{0:2}\n",
                       llvm::json::Value(llvm::json::Object(shortforms)))
             .str();
  Take(HloVerifier(false, false).Run(source.get()));
  source->mutable_config().set_replica_count(1);
  source->mutable_config().set_num_partitions(1);
  std::ofstream(out / "debug-options.pbtxt")
      << source->config().debug_options().DebugString();
  CpuClientOptions options;
  options.asynchronous = false;
  options.cpu_device_count = 1;
  options.max_transpose_threads = 1;
  HloRunner runner(Take(GetXlaPjrtCpuClient(std::move(options))));
  TeplBranchAdapter implementation(argv[1], argv[3], argv[2], true);
  std::unique_ptr<OpaqueExecutable> executables[2];
  std::map<std::pair<int, std::string>, int64_t> measured[2];
  int64_t calls[2];
  std::string final[2];
  for (int variant = 0; variant < 2; ++variant) {
    invocations = 0;
    counts = &measured[variant];
    SetAlgebraicRewriteCounter(Counter);
    if (variant) {
      adapter = &implementation;
      SetTeplAlgebraicHook(Transpose);
      SetTeplBranchHook(Branch);
    }
    auto executable =
        runner.CreateExecutable(source->Clone("comparison"), true);
    calls[variant] = invocations;
    ResetHooks();
    executables[variant] = Take(std::move(executable));
    final[variant] = Canonical(
        *Take(runner.HloModuleFromWrapped(executables[variant].get())));
    std::ofstream(out / (variant ? "tepl.final.hlo" : "baseline.final.hlo"))
        << final[variant];
    Require(calls[variant] > 0, "AlgebraicSimplifier was not invoked");
  }
  Require(final[0] == final[1], "final CPU graphs differ");
  Require(measured[0] == measured[1],
          "native/integrated rewrite counts differ");
  Require(calls[0] == calls[1],
          "native/integrated pass invocation counts differ");
  llvm::json::Array rows, missing;
  int64_t total = 0, delegated = 0;
  for (const auto& [identity, applications] : measured[1]) {
    const auto& [site, function] = identity;
    // Match the retained counter's bookkeeping normalization.
    if (site == 9009) continue;
    const int64_t tepl = implementation.committed[site];
    Require(tepl <= applications, "TEPL commits exceed actual applications");
    Require(!TeplBranchAdapter::IsPorted(site) || tepl == applications,
            "ported branch fell back to native");
    rows.push_back(llvm::json::Object{{"original_line", site},
                                      {"function", function},
                                      {"applications", applications},
                                      {"tepl_applications", tepl}});
    total += applications;
    delegated += tepl;
    if (tepl != applications) missing.push_back(site);
  }
  llvm::json::Object result{
      {"scope",
       "all AlgebraicSimplifier invocations in paired normal CPU compilations"},
      {"mode", strict ? "strict" : "hybrid"},
      {"shortform_constants", std::move(shortforms)},
      {"final_graphs_equal", true},
      {"rewrite_counts_equal", true},
      {"outputs", llvm::json::Array{}},
      {"rules", std::move(rows)},
      {"distinct_variants",
       int64_t(measured[1].size() -
               (measured[1].count({9009, "HandleReduce"}) ? 1 : 0))},
      {"total_applications", total},
      {"tepl_applications", delegated},
      {"baseline_pass_invocations", calls[0]},
      {"integrated_pass_invocations", calls[1]},
      {"all_observed_variants_delegated", missing.empty()},
      {"unported_sites", std::move(missing)}};
  std::ofstream(out / "compilation.json")
      << llvm::formatv("{0:2}\n", llvm::json::Value(llvm::json::Object(result)))
             .str();
  Require(
      !strict || result.getBoolean("all_observed_variants_delegated") == true,
      "strict integration incomplete; unported source sites are recorded in "
      "compilation.json");
  llvm::json::Array outputs;
  for (int seed = 1; seed <= 3; ++seed) {
    auto arguments = Arguments(*source, seed);
    std::vector<const Literal*> pointers;
    for (const auto& value : arguments) pointers.push_back(&value);
    auto baseline =
        Take(runner.ExecuteWithExecutable(executables[0].get(), pointers));
    auto integrated =
        Take(runner.ExecuteWithExecutable(executables[1].get(), pointers));
    Require(Finite(baseline) && Finite(integrated),
            "non-finite workload output");
    const auto comparison = literal_comparison::Near(
        baseline, integrated, ErrorSpec(1.e-5, 1.e-5), true, nullptr);
    Require(comparison.ok(), comparison.ToString());
    outputs.push_back(llvm::json::Object{{"seed", seed}, {"equal", true}});
  }
  result["outputs"] = std::move(outputs);
  const std::string rendered =
      llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result))).str();
  std::ofstream(out / "results.json") << rendered;
  std::cout << rendered;
  return 0;
}
}  // namespace
}  // namespace xla
int main(int argc, char** argv) {
  try {
    return xla::Run(argc, argv);
  } catch (const std::exception& error) {
    xla::ResetHooks();
    std::cerr << error.what() << '\n';
    return 1;
  }
}
