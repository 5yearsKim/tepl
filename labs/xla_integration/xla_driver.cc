// Actual XLA HLO pass integration; parsing/cloning/loading/reference execution
// are outside the pass timers. The upstream pass is the handwritten baseline.
#include <dlfcn.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/status/status_macros.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "research/tepl_integration/bridge.h"
#include "research/tepl_integration/transpose_folding.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/shape_inference.h"
#include "xla/service/transpose_folding.h"
#include "xla/shape_util.h"
namespace xla {
namespace {
template <class T>
T take(absl::StatusOr<T> v) {
  if (!v.ok()) throw std::runtime_error(v.status().ToString());
  return std::move(v).value();
}
void require(bool x, const std::string& why) {
  if (!x) throw std::runtime_error(why);
}
std::unique_ptr<HloModule> CreateNewVerifiedModule(const std::string& name) {
  return std::make_unique<HloModule>(name, HloModuleConfig());
}
PrecisionConfig DefaultPrecisionConfig(int n) {
  PrecisionConfig p;
  while (n--) p.add_operand_precision(PrecisionConfig::DEFAULT);
  return p;
}
#include "research/tepl_integration/upstream_builders.inc"
using LoadFn = void* (*)(const char*, char*, size_t);
using FoldFn = int (*)(void*, int, TeplList, TeplList, TeplList,
                       const TeplList*, int64_t*, size_t, size_t*, char*,
                       size_t);
using DestroyFn = void (*)(void*);
using CacheFn = int (*)(void*, int, uint64_t*, uint64_t*);
struct State {
  void* context = nullptr;
  FoldFn fold = nullptr;
  bool custom = false;
  size_t tepl_operands = 0, manual_dots = 0, manual_convs = 0;
};
thread_local State* current = nullptr;
TeplList view(const tsl::protobuf::RepeatedField<int64_t>& p) {
  return {p.data(), size_t(p.size())};
}
TeplList view(absl::Span<const int64_t> p) { return {p.data(), p.size()}; }
TeplList shape(const HloInstruction* n) {
  return {n->shape().dimensions().data(),
          size_t(n->shape().dimensions().size())};
}
std::string read(const std::filesystem::path& p) {
  std::ifstream f(p);
  require(bool(f), "missing " + p.string());
  return {std::istreambuf_iterator<char>(f), {}};
}
std::vector<Literal> arguments(const HloModule& module) {
  std::vector<Literal> args;
  for (auto* p : module.entry_computation()->parameter_instructions()) {
    auto x = Literal::CreateFromShape(p->shape());
    int64_t seed = p->parameter_number() + 1;
    if (p->shape().element_type() == F32) {
      auto s = x.Populate<float>([&](absl::Span<const int64_t> at) {
        int64_t v = seed;
        for (auto i : at) v = v * 3 + i;
        return float((v % 5) - 2);
      });
      require(s.ok(), s.ToString());
    } else if (p->shape().element_type() == S32) {
      auto s = x.Populate<int32_t>(
          [](absl::Span<const int64_t>) { return int32_t(0); });
      require(s.ok(), s.ToString());
    } else
      throw std::runtime_error("reference input dtype unsupported");
    args.push_back(std::move(x));
  }
  return args;
}
bool run(HloModule* m, bool integrated, bool custom) {
  if (integrated) {
    if (custom)
      return take(TeplTransposeFolding([](const HloInstruction&,
                                          int64_t) -> absl::StatusOr<bool> {
                    return true;
                  }).Run(m));
    return take(TeplTransposeFolding().Run(m));
  }
  if (custom)
    return take(TransposeFolding([](const HloInstruction&,
                                    int64_t) -> absl::StatusOr<bool> {
                  return true;
                }).Run(m));
  return take(TransposeFolding().Run(m));
}
void verify(HloModule* m) {
  auto s = HloVerifier(false, false).Run(m);
  require(s.ok(), s.status().ToString());
}
}  // namespace
bool UseHandwrittenDot() { return current->custom; }
void RecordHandwrittenDot() { ++current->manual_dots; }
void RecordHandwrittenConvolution() { ++current->manual_convs; }
absl::Status FoldDotViaTepl(HloInstruction* dot,
                            const std::vector<int64_t>& operands) {
  DotDimensionNumbers dims = dot->dot_dimension_numbers();
  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  for (auto side : operands) {
    HloInstruction* transposed = side ? rhs : lhs;
    TeplList fields[] = {view(dims.lhs_contracting_dimensions()),
                         view(dims.rhs_contracting_dimensions()),
                         view(dims.lhs_batch_dimensions()),
                         view(dims.rhs_batch_dimensions())};
    HloInstruction* left = side ? lhs : transposed->mutable_operand(0);
    HloInstruction* right = side ? transposed->mutable_operand(0) : rhs;
    if (left->shape().element_type() != F32 ||
        right->shape().element_type() != F32)
      return absl::UnimplementedError("integration currently uses f32");
    std::vector<int64_t> output(4 * (left->shape().dimensions().size() +
                                     right->shape().dimensions().size()) +
                                16);
    size_t lengths[4];
    char error[2048] = {};
    int code =
        current->fold(current->context, int(side), shape(left), shape(right),
                      view(transposed->dimensions()), fields, output.data(),
                      output.size(), lengths, error, sizeof(error));
    if (code) return absl::InternalError(error);
    size_t at = 0;
    auto set = [&](auto* field, size_t n) {
      field->Assign(output.begin() + at, output.begin() + at + n);
      at += n;
    };
    set(dims.mutable_lhs_contracting_dimensions(), lengths[0]);
    set(dims.mutable_rhs_contracting_dimensions(), lengths[1]);
    set(dims.mutable_lhs_batch_dimensions(), lengths[2]);
    set(dims.mutable_rhs_batch_dimensions(), lengths[3]);
    if (side)
      rhs = right;
    else
      lhs = left;
    ++current->tepl_operands;
  }
  auto* replacement = dot->parent()->AddInstruction(HloInstruction::CreateDot(
      dot->shape(), lhs, rhs, dims, dot->precision_config()));
  dot->SetupDerivedInstruction(replacement);
  return dot->parent()->ReplaceInstruction(dot, replacement);
}
int experiment(int argc, char** argv) {
  require(argc == 6,
          "expected: shared-library program.json manifest.json input-dir "
          "output-dir");
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) throw std::runtime_error(dlerror());
  auto load = reinterpret_cast<LoadFn>(dlsym(library, "tepl_xla_load"));
  auto fold = reinterpret_cast<FoldFn>(dlsym(library, "tepl_xla_fold"));
  auto destroy =
      reinterpret_cast<DestroyFn>(dlsym(library, "tepl_xla_destroy"));
  auto cache = reinterpret_cast<CacheFn>(dlsym(library, "tepl_xla_cache"));
  require(load && fold && destroy && cache, "bridge symbols");
  char error[2048] = {};
  auto start = std::chrono::steady_clock::now();
  void* context = load(argv[2], error, sizeof(error));
  require(context, error);
  double load_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  auto parsed = llvm::json::parse(read(argv[3]));
  require(bool(parsed), "manifest JSON");
  auto* manifest = parsed->getAsArray();
  require(manifest, "manifest array");
  std::filesystem::path input = argv[4], out = argv[5];
  std::filesystem::create_directories(out);
  llvm::json::Array results, samples;
  unsigned checked = 0;
  for (const auto& item : *manifest) {
    auto* row = item.getAsObject();
    require(row, "manifest row");
    std::string name = row->getString("name")->str(),
                kind = row->getString("input")->str();
    bool custom = *row->getString("config") == "custom";
    auto module = kind == "literal" ? take(ParseAndReturnUnverifiedModule(
                                          read(input / (name + ".hlo"))))
                                    : take(makeBuilder(name));
    verify(module.get());
    std::ofstream(out / (name + ".before.hlo")) << module->ToString();
    auto manual = module->Clone("comparison"),
         automatic = module->Clone("comparison");
    bool a = run(manual.get(), false, custom);
    State stats{context, fold, custom};
    current = &stats;
    bool b = run(automatic.get(), true, custom);
    verify(manual.get());
    verify(automatic.get());
    require(a == b, "changed differs " + name);
    require(manual->ToString() == automatic->ToString(), "HLO differs " + name);
    ++checked;
    std::ofstream(out / (name + ".manual.hlo")) << manual->ToString();
    std::ofstream(out / (name + ".tepl.hlo")) << automatic->ToString();
    bool evaluable = true;
    for (const auto* computation : module->computations())
      for (const auto* instruction : computation->instructions())
        if (instruction->opcode() == HloOpcode::kConvolution &&
            instruction->operand_count() != 2)
          evaluable = false;
    auto args = arguments(*module);
    HloEvaluator evaluator;
    absl::StatusOr<Literal> original =
        evaluable ? evaluator.Evaluate(*module, absl::MakeConstSpan(args))
                  : absl::StatusOr<Literal>(absl::UnimplementedError(
                        "reference harness excludes sparse/additional "
                        "convolution operands"));
    std::string numeric = "unsupported";
    if (original.ok()) {
      HloEvaluator manual_evaluator, automatic_evaluator;
      auto reference =
          manual_evaluator.Evaluate(*manual, absl::MakeConstSpan(args));
      auto actual =
          automatic_evaluator.Evaluate(*automatic, absl::MakeConstSpan(args));
      require(reference.ok() && actual.ok(),
              "replacement evaluation failed " + name);
      require(
          original->Equal(*reference, false) && original->Equal(*actual, false),
          "numerical mismatch " + name);
      numeric = "equal";
    } else
      numeric = original.status().ToString();
    results.push_back(llvm::json::Object{
        {"case", name},
        {"config", custom ? "custom" : "default"},
        {"family", row->getString("family")->str()},
        {"changed", a},
        {"hlo_equal", true},
        {"tepl_operands", int64_t(stats.tepl_operands)},
        {"manual_dot_fallbacks", int64_t(stats.manual_dots)},
        {"manual_conv_fallbacks", int64_t(stats.manual_convs)},
        {"numerical", numeric}});
    for (int round = -1; round < 10; ++round)
      for (int order = 0; order < 3; ++order) {
        int variant = (round + order + 3) % 3;
        bool integrated = variant != 0;
        std::vector<std::unique_ptr<HloModule>> batch;
        for (int i = 0; i < 100; ++i) batch.push_back(module->Clone("timing"));
        State timed{context, fold, custom};
        current = &timed;
        require(cache(context, 1, nullptr, nullptr) == 0, "cache reset");
        if (variant == 2) {
          auto warm = module->Clone("warmup");
          run(warm.get(), true, custom);
        }
        uint64_t hits_before = 0, misses_before = 0, hits_after = 0,
                 misses_after = 0;
        require(cache(context, 0, &hits_before, &misses_before) == 0,
                "cache counters");
        auto t = std::chrono::steady_clock::now();
        unsigned changes = 0;
        for (auto& m : batch) {
          if (variant == 1)
            require(cache(context, 1, nullptr, nullptr) == 0,
                    "cold cache reset");
          changes += run(m.get(), integrated, custom);
        }
        double ns = std::chrono::duration<double, std::nano>(
                        std::chrono::steady_clock::now() - t)
                        .count() /
                    100;
        require(cache(context, 0, &hits_after, &misses_after) == 0,
                "cache counters");
        require(changes == (a ? 100 : 0), "timing behavior differs");
        require(variant != 1 || hits_after == hits_before,
                "cold cache unexpectedly reused metadata");
        require(variant != 2 || misses_after == misses_before,
                "warm cache missed metadata");
        samples.push_back(llvm::json::Object{
            {"case", name},
            {"round", round},
            {"variant", variant == 0   ? "upstream"
                        : variant == 1 ? "tepl_cold"
                                       : "tepl_warm"},
            {"batch", 100},
            {"ns_per_pass", ns},
            {"cache_hits", int64_t(hits_after - hits_before)},
            {"cache_misses", int64_t(misses_after - misses_before)}});
      }
  }
  llvm::json::Object output{{"program_load_ms", load_ms},
                            {"cases_checked", int64_t(checked)},
                            {"results", std::move(results)},
                            {"samples", std::move(samples)}};
  std::cout
      << llvm::formatv("{0:2}", llvm::json::Value(std::move(output))).str()
      << "\n";
  destroy(context);
  dlclose(library);
  return 0;
}
}  // namespace xla
int main(int argc, char** argv) {
  try {
    return xla::experiment(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
