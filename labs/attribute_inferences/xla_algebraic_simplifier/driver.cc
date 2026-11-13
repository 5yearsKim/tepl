#include <dlfcn.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "research/tepl_algebraic/branch_builder.h"
#include "research/tepl_algebraic/bridge.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/literal_comparison.h"
#include "xla/literal_util.h"
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
template <class T>
T Take(absl::StatusOr<T> value) {
  if (!value.ok()) throw std::runtime_error(value.status().ToString());
  return std::move(value).value();
}
void Require(bool good, const std::string& message) {
  if (!good) throw std::runtime_error(message);
}
std::string Read(const char* path) {
  std::ifstream input(path);
  Require(bool(input), std::string("missing ") + path);
  return {std::istreambuf_iterator<char>(input), {}};
}
std::string Canonical(const HloModule& module) {
  return module.ToString(HloPrintOptions::Canonical()
                             .set_print_large_constants(true)
                             .set_print_only_essential_constants(false)
                             .set_print_ids(false)
                             .set_print_backend_config(true)
                             .set_print_control_dependencies(true)
                             .set_canonicalize_computations(true));
}
using Load = void* (*)(const char*, char*, size_t);
using Match = int (*)(void*, const AlgebraicNode*, size_t, size_t, uint64_t,
                      AlgebraicResult*, char*, size_t);
using Destroy = void (*)(void*);
struct Adapter {
  void* library;
  void* program;
  Match match;
  Destroy destroy;
  std::map<int, int64_t> applications;
  Adapter(const char* library_path, const char* program_path) {
    library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    Require(library != nullptr, "cannot load adapter library");
    auto load = reinterpret_cast<Load>(dlsym(library, "tepl_algebraic_load"));
    match = reinterpret_cast<Match>(dlsym(library, "tepl_algebraic_match"));
    destroy =
        reinterpret_cast<Destroy>(dlsym(library, "tepl_algebraic_destroy"));
    Require(load && match && destroy, "incompatible adapter ABI");
    char error[4096] = {};
    program = load(program_path, error, sizeof(error));
    Require(program != nullptr, error);
  }
  ~Adapter() {
    destroy(program);
    dlclose(library);
  }
};
Adapter* current = nullptr;
TeplBranchAdapter* branch_adapter = nullptr;
std::map<int, int64_t>* observed = nullptr;
void Observe(int site, const char*) {
  if (site && observed) ++(*observed)[site];
  if (site && branch_adapter) branch_adapter->Observe(site);
}
absl::StatusOr<HloInstruction*> BranchHook(
    HloInstruction* root, int site, absl::Span<HloInstruction* const> captures,
    absl::Span<const int64_t> first, absl::Span<const int64_t> second) {
  return branch_adapter->Propose(root, site, captures, first, second);
}
absl::StatusOr<HloInstruction*> TransposeHook(HloInstruction* root, int site) {
  return branch_adapter->Transpose(root, site);
}

// The first integration slice admits only f32 transposes with static shapes.
// Opaque boundaries preserve HLO identity, sharing, and arbitrary producers.
absl::StatusOr<HloInstruction*> Propose(HloInstruction* root, int site) {
  if (!current || !root->shape().IsArray() ||
      root->shape().element_type() != F32 || root->shape().is_dynamic())
    return nullptr;
  std::vector<AlgebraicNode> nodes;
  std::vector<HloInstruction*> originals;
  HloInstruction* operand = root->mutable_operand(0);
  const bool nested = site == 10118;
  HloInstruction* input = nested ? operand->mutable_operand(0) : operand;
  if (input->shape().is_dynamic()) return nullptr;
  auto add = [&](HloInstruction* instruction, bool transpose) {
    AlgebraicNode node{};
    node.original_id = instruction->unique_id();
    node.f32 = 1;
    node.shape = {instruction->shape().dimensions().data(),
                  size_t(instruction->shape().dimensions().size())};
    if (transpose) {
      node.op = 1;
      node.arity = 1;
      node.operands[0] = nodes.size() - 1;
      node.permutation = {instruction->dimensions().data(),
                          size_t(instruction->dimensions().size())};
    }
    nodes.push_back(node);
    originals.push_back(instruction);
  };
  add(input, false);
  if (nested) add(operand, true);
  add(root, true);
  AlgebraicResult result{};
  char error[4096] = {};
  int code = current->match(current->program, nodes.data(), nodes.size(),
                            nodes.size() - 1, 0, &result, error, sizeof(error));
  if (code < 0) return absl::InternalError(error);
  if (code == 0) return nullptr;
  const AlgebraicNode& replacement = result.nodes[result.root];
  HloInstruction* output = nullptr;
  if (replacement.original_id >= 0) {
    for (HloInstruction* original : originals)
      if (original->unique_id() == replacement.original_id) output = original;
  } else {
    if (replacement.op != 1 || replacement.arity != 1)
      return absl::InternalError(
          "adapter supports only transpose construction");
    const AlgebraicNode& captured = result.nodes[replacement.operands[0]];
    HloInstruction* source = nullptr;
    for (HloInstruction* original : originals)
      if (original->unique_id() == captured.original_id) source = original;
    if (source == nullptr) return absl::InternalError("missing captured HLO");
    if (!absl::c_equal(root->shape().dimensions(),
                       absl::Span<const int64_t>(replacement.shape.data,
                                                 replacement.shape.size)))
      return absl::InternalError("replacement shape differs");
    output = root->parent()->AddInstruction(HloInstruction::CreateTranspose(
        root->shape(), source,
        absl::Span<const int64_t>(replacement.permutation.data,
                                  replacement.permutation.size)));
    root->SetupDerivedInstruction(output);
  }
  if (output == nullptr || output == root)
    return absl::InternalError("invalid replacement identity");
  ++current->applications[site];
  return output;
}

llvm::json::Object Compare(const std::string& hlo, Adapter& adapter,
                           const std::filesystem::path& out,
                           TeplBranchAdapter* branches = nullptr) {
  auto original = Take(ParseAndReturnUnverifiedModule(hlo));
  Take(HloVerifier(false, false).Run(original.get()));
  auto baseline = original->Clone("comparison");
  auto integrated = original->Clone("comparison");
  AlgebraicSimplifierOptions pass_options;
  if (branches) {
    pass_options.set_executing_on_cpu(true);
    pass_options.set_supports_non_canonical_dots(false);
    pass_options.set_enable_dot_strength_reduction(false);
  }
  std::map<int, int64_t> baseline_counts, integrated_counts;
  observed = &baseline_counts;
  SetAlgebraicRewriteCounter(Observe);
  Take(AlgebraicSimplifier(pass_options).Run(baseline.get()));
  observed = &integrated_counts;
  current = &adapter;
  adapter.applications.clear();
  branch_adapter = branches;
  SetTeplAlgebraicHook(branches ? TransposeHook : Propose);
  SetTeplBranchHook(branches ? BranchHook : nullptr);
  auto status = AlgebraicSimplifier(pass_options).Run(integrated.get());
  SetTeplAlgebraicHook(nullptr);
  SetTeplBranchHook(nullptr);
  SetAlgebraicRewriteCounter(nullptr);
  observed = nullptr;
  branch_adapter = nullptr;
  current = nullptr;
  Take(std::move(status));
  Take(HloDCE().Run(baseline.get()));
  Take(HloDCE().Run(integrated.get()));
  Take(HloVerifier(false, false).Run(integrated.get()));
  Require(Canonical(*baseline) == Canonical(*integrated),
          "final pass graph differs");
  if (!out.empty()) {
    std::filesystem::create_directories(out);
    std::ofstream(out / "baseline.hlo") << Canonical(*baseline);
    std::ofstream(out / "tepl.hlo") << Canonical(*integrated);
  }
  // Exact checks are appropriate for these two pure reindexing rules.
  for (int seed = 0; seed < 3; ++seed) {
    std::vector<Literal> arguments;
    std::vector<const Literal*> pointers;
    for (HloInstruction* p :
         original->entry_computation()->parameter_instructions()) {
      Require(p->shape().IsArray(), "pass evaluator requires array parameters");
      auto literal = Literal::CreateFromShape(p->shape());
      auto number = [&](absl::Span<const int64_t> at) {
        int64_t number = seed + p->parameter_number();
        for (int64_t index : at) number = (number * 17 + index) % 997;
        return float(number - 400) / 16;
      };
      absl::Status populated;
      switch (p->shape().element_type()) {
        case F32:
          populated = literal.Populate<float>(number);
          break;
        case F64:
          populated = literal.Populate<double>(number);
          break;
        case BF16:
          populated = literal.Populate<bfloat16>(
              [&](auto at) { return bfloat16(number(at)); });
          break;
        case S32:
          populated = literal.Populate<int32_t>(
              [&](auto at) { return int32_t(number(at)); });
          break;
        case S64:
          populated = literal.Populate<int64_t>(
              [&](auto at) { return int64_t(number(at)); });
          break;
        case PRED:
          populated =
              literal.Populate<bool>([&](auto at) { return number(at) > -20; });
          break;
        default:
          throw std::runtime_error("unsupported evaluator parameter dtype");
      }
      Require(populated.ok(), populated.ToString());
      arguments.push_back(std::move(literal));
    }
    for (const Literal& arg : arguments) pointers.push_back(&arg);
    HloEvaluator evaluator;
    auto reference = Take(evaluator.Evaluate(*original, pointers));
    evaluator.ResetVisitStates();
    auto native = Take(evaluator.Evaluate(*baseline, pointers));
    evaluator.ResetVisitStates();
    auto actual = Take(evaluator.Evaluate(*integrated, pointers));
    const ErrorSpec source_tolerance =
        branches ? ErrorSpec(1.e-5, 1.e-5) : ErrorSpec(0, 0);
    Require(literal_comparison::Near(reference, native, source_tolerance, true,
                                     nullptr)
                .ok(),
            "native output differs from source");
    Require(
        literal_comparison::Near(native, actual, ErrorSpec(0, 0), true, nullptr)
            .ok(),
        "TEPL output differs from source");
  }
  llvm::json::Array applications;
  const auto& delegated = branches ? branches->committed : adapter.applications;
  for (const auto& [site, count] : delegated)
    applications.push_back(
        llvm::json::Object{{"source_site", site}, {"applications", count}});
  llvm::json::Array native;
  for (const auto& [site, count] : integrated_counts) {
    const auto found = delegated.find(site);
    int64_t tepl = found == delegated.end() ? 0 : found->second;
    if (tepl > count)
      throw std::runtime_error("delegation exceeds native commit count");
    if (branches && TeplBranchAdapter::IsPorted(site) && tepl != count)
      throw std::runtime_error("ported branch fell back to native: AS-" +
                               std::to_string(site));
    native.push_back(llvm::json::Object{
        {"source_site", site}, {"applications", count}, {"tepl", tepl}});
  }
  return llvm::json::Object{
      {"graph_equal", true},
      {"numerical_input_sets", 3},
      {"tepl_applications", std::move(applications)},
      {"rewrite_applications", std::move(native)},
      {"scope", "one AlgebraicSimplifier invocation followed by DCE"}};
}
int Run(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--branches") {
    Require(argc == 6 || argc == 7,
            "expected --branches library.so automatic.json branches.json "
            "input.hlo [output_dir]");
    Adapter automatic(argv[2], argv[3]);
    TeplBranchAdapter branches(argv[2], argv[4], argv[3], true);
    auto result =
        Compare(Read(argv[5]), automatic, argc == 7 ? argv[6] : "", &branches);
    std::cout
        << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result))).str();
    return 0;
  }
  Require(argc == 4 || argc == 5,
          "expected library.so program.json input.hlo [output_dir]");
  Adapter adapter(argv[1], argv[2]);
  auto result = Compare(Read(argv[3]), adapter, argc == 5 ? argv[4] : "");
  std::cout
      << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result))).str();
  return 0;
}
}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  try {
    return xla::Run(argc, argv);
  } catch (const std::exception& error) {
    xla::SetTeplAlgebraicHook(nullptr);
    xla::SetTeplBranchHook(nullptr);
    xla::SetAlgebraicRewriteCounter(nullptr);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
