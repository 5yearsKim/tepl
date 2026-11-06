#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "research/tepl_workflow/bridge.h"
#include "research/tepl_workflow/hook.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/literal_comparison.h"
#include "xla/literal_util.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape_util.h"
namespace xla {
namespace {
template <class T>
T take(absl::StatusOr<T> value) {
  if (!value.ok()) throw std::runtime_error(value.status().ToString());
  return std::move(value).value();
}
void require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}
std::string read(const char* p) {
  std::ifstream f(p);
  require(bool(f), std::string("missing ") + p);
  return {std::istreambuf_iterator<char>(f), {}};
}
void write(const std::filesystem::path& p, const std::string& s) {
  std::ofstream(p) << s;
}
using Load = void* (*)(const char*, char*, size_t);
using Match = int (*)(void*, const WorkflowNode*, size_t, size_t, uint64_t,
                      WorkflowResult*, char*, size_t);
using Destroy = void (*)(void*);
using Roots = uint64_t (*)(void*);
using Depth = size_t (*)(void*);
struct State {
  bool audit = false;
  int mode = 0;
  void* bridge = nullptr;
  Match match = nullptr;
  uint64_t roots = 0;
  size_t depth = 0;
  size_t tepl = 0, retained_dot = 0, retained_conv = 0, scanned = 0,
         proposals = 0, backend_rejected = 0, pass_calls = 0;
  std::vector<std::string> before, after;
};
State* current = nullptr;
WorkflowList list(const tsl::protobuf::RepeatedField<int64_t>& xs) {
  return {xs.data(), size_t(xs.size())};
}
WorkflowList list(absl::Span<const int64_t> xs) {
  return {xs.data(), xs.size()};
}
int opcode(const HloInstruction* p) {
  return p->opcode() == HloOpcode::kTranspose ? 1
         : p->opcode() == HloOpcode::kDot     ? 2
                                              : 0;
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
struct View {
  std::vector<WorkflowNode> nodes;
  std::map<HloInstruction*, size_t> ids, depths;
  int supported(HloInstruction* instruction) {
    if (!instruction->shape().IsArray() ||
        instruction->shape().element_type() != F32)
      return 0;
    int op = opcode(instruction);
    if (op == 2 && instruction->operand_count() != 2) return 0;
    return op;
  }
  void collect(HloInstruction* instruction, size_t depth) {
    auto old = depths.find(instruction);
    if (old != depths.end() && old->second >= depth) return;
    depths[instruction] = depth;
    if (depth && supported(instruction))
      for (auto* operand : instruction->mutable_operands())
        collect(operand, depth - 1);
  }
  size_t add(HloInstruction* instruction, size_t depth) {
    collect(instruction, depth);
    return emit(instruction);
  }
  size_t emit(HloInstruction* instruction) {
    size_t depth = depths.at(instruction);
    auto old = ids.find(instruction);
    if (old != ids.end()) return old->second;
    WorkflowNode v{};
    v.original_id = instruction->unique_id();
    v.f32 = instruction->shape().IsArray() &&
            instruction->shape().element_type() == F32;
    if (instruction->shape().IsArray())
      v.shape = list(instruction->shape().dimensions());
    v.op = depth ? supported(instruction) : 0;
    if (v.op) {
      v.arity = instruction->operand_count();
      require(v.arity <= 2, "adapter arity");
      for (size_t i = 0; i < v.arity; ++i)
        v.operands[i] = emit(instruction->mutable_operand(i));
      if (v.op == 1)
        v.permutation = list(instruction->dimensions());
      else {
        const auto& d = instruction->dot_dimension_numbers();
        v.fields[0] = list(d.lhs_contracting_dimensions());
        v.fields[1] = list(d.rhs_contracting_dimensions());
        v.fields[2] = list(d.lhs_batch_dimensions());
        v.fields[3] = list(d.rhs_batch_dimensions());
      }
    }
    size_t id = nodes.size();
    ids[instruction] = id;
    nodes.push_back(v);
    return id;
  }
};
void observer(int kind, size_t n) {
  if (current) {
    if (kind == 0)
      current->retained_dot += n;
    else
      current->retained_conv += n;
  }
}
absl::StatusOr<bool> hook(
    HloModule* module, const absl::flat_hash_set<absl::string_view>& threads,
    const TransposeFolding::CanFoldTransposeOperand& policy,
    const TransposeFolding::TransposableConvOperandsFn& conv);
absl::StatusOr<bool> teplRun(
    HloModule* module, const absl::flat_hash_set<absl::string_view>& threads,
    const TransposeFolding::CanFoldTransposeOperand& policy) {
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(threads)) {
    auto order = comp->MakeInstructionPostOrder();
    for (auto* original : order) {
      HloInstruction* root = original;
      uint64_t excluded = 0;
      bool policy_checked = false, eligible[2] = {false, false};
      for (int step = 0; step < 64; ++step) {
        int op = opcode(root);
        if (!(current->roots & (uint64_t(1) << op))) break;
        ++current->scanned;
        View view;
        size_t id = view.add(root, current->depth);
        WorkflowResult result{};
        char error[4096] = {};
        int code = current->match(current->bridge, view.nodes.data(),
                                  view.nodes.size(), id, excluded, &result,
                                  error, sizeof(error));
        if (code < 0) return absl::InternalError(error);
        if (!code) break;
        ++current->proposals;
        const auto& replacement = result.nodes[result.root];
        bool allowed = true;
        size_t folds = 0;
        if (root->opcode() == HloOpcode::kDot) {
          // The upstream pass asks both callbacks before replacing a dot.
          // Preserve that policy context when TEPL folds its sides in turn.
          if (!policy_checked) {
            for (size_t side = 0; side < 2; ++side)
              if (root->operand(side)->opcode() == HloOpcode::kTranspose) {
                ABSL_ASSIGN_OR_RETURN(eligible[side], policy(*root, side));
              }
            policy_checked = true;
          }
          for (size_t side = 0; side < replacement.arity; ++side) {
            auto source = result.nodes[replacement.operands[side]].original_id;
            if (source != root->operand(side)->unique_id()) {
              allowed &= eligible[side];
              ++folds;
            }
          }
        }
        if (!allowed) {
          ++current->backend_rejected;
          excluded |= uint64_t(1) << result.rule;
          continue;
        }
        std::map<int64_t, HloInstruction*> originals;
        for (const auto& entry : view.ids)
          originals[entry.first->unique_id()] = entry.first;
        std::map<size_t, HloInstruction*> built;
        std::function<HloInstruction*(size_t)> build =
            [&](size_t at) -> HloInstruction* {
          const auto& n = result.nodes[at];
          if (n.original_id >= 0) return originals.at(n.original_id);
          if (built.count(at)) return built.at(at);
          std::vector<HloInstruction*> operands;
          for (size_t k = 0; k < n.arity; ++k)
            operands.push_back(build(n.operands[k]));
          std::vector<int64_t> dims(n.shape.data, n.shape.data + n.shape.size);
          Shape shape = at == result.root ? root->shape()
                                          : ShapeUtil::MakeShape(F32, dims);
          require(absl::c_equal(shape.dimensions(), dims),
                  "replacement shape differs");
          std::unique_ptr<HloInstruction> node;
          if (n.op == 1)
            node = HloInstruction::CreateTranspose(
                shape, operands.at(0),
                absl::Span<const int64_t>(n.permutation.data,
                                          n.permutation.size));
          else if (n.op == 2) {
            DotDimensionNumbers d;
            auto set = [](auto* xs, WorkflowList l) {
              xs->Assign(l.data, l.data + l.size);
            };
            set(d.mutable_lhs_contracting_dimensions(), n.fields[0]);
            set(d.mutable_rhs_contracting_dimensions(), n.fields[1]);
            set(d.mutable_lhs_batch_dimensions(), n.fields[2]);
            set(d.mutable_rhs_batch_dimensions(), n.fields[3]);
            node =
                HloInstruction::CreateDot(shape, operands.at(0), operands.at(1),
                                          d, root->precision_config());
          } else
            throw std::runtime_error("unsupported replacement operation");
          auto* instruction = comp->AddInstruction(std::move(node));
          root->SetupDerivedInstruction(instruction);
          built[at] = instruction;
          return instruction;
        };
        auto* next = build(result.root);
        require(next != root, "non-progressing rewrite");
        ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(root, next));
        root = next;
        current->tepl += folds;
        changed = true;
        // Apply each selected rule at most once to an original root,
        // matching the upstream pass's single traversal.
        excluded |= uint64_t(1) << result.rule;
      }
    }
  }
  return changed;
}
absl::StatusOr<bool> hook(
    HloModule* module, const absl::flat_hash_set<absl::string_view>& threads,
    const TransposeFolding::CanFoldTransposeOperand& policy,
    const TransposeFolding::TransposableConvOperandsFn& conv) {
  ++current->pass_calls;
  if (current->audit) current->before.push_back(canonical(*module));
  bool changed = false;
  if (current->mode) {
    ABSL_ASSIGN_OR_RETURN(changed, teplRun(module, threads, policy));
  }
  SetTransposeFoldingWorkflowHook(nullptr);
  auto native = TransposeFolding(policy, conv).Run(module, threads);
  SetTransposeFoldingWorkflowHook(hook);
  if (!native.ok()) return native.status();
  changed |= *native;
  if (current->audit) {
    HloVerifier verifier(false, false);
    auto status = verifier.Run(module);
    if (!status.ok()) return status.status();
    current->after.push_back(canonical(*module));
  }
  return changed;
}
std::vector<Literal> arguments(const HloModule& m, int seed) {
  std::vector<Literal> args;
  for (auto* p : m.entry_computation()->parameter_instructions()) {
    const auto& shape = p->shape();
    require(shape.IsArray(), "tuple parameter unsupported");
    auto literal = Literal::CreateFromShape(shape);
    auto number = [&](absl::Span<const int64_t> index) {
      int64_t v = seed + p->parameter_number() * 7;
      for (auto n : index) v = (v * 17 + n) % 997;
      return float(1 + (v % 97)) / 10000.f;
    };
    absl::Status s;
    switch (shape.element_type()) {
      case F32:
        s = literal.Populate<float>(number);
        break;
      case S32:
        s = literal.Populate<int32_t>([](auto) { return int32_t(1); });
        break;
      case S64:
        s = literal.Populate<int64_t>([](auto) { return int64_t(1); });
        break;
      case PRED:
        s = literal.Populate<bool>([](auto) { return false; });
        break;
      default:
        throw std::runtime_error("input dtype unsupported");
    }
    if (!s.ok()) throw std::runtime_error(s.ToString());
    args.push_back(std::move(literal));
  }
  return args;
}
}  // namespace
bool finiteOutputs(const Literal& value) {
  bool finite = true;
  ShapeUtil::ForEachSubshape(
      value.shape(), [&](const Shape& shape, const ShapeIndex& at) {
        if (shape.IsArray() && shape.element_type() == F32)
          for (float x : value.data<float>(at)) finite &= std::isfinite(x);
      });
  return finite;
}
int selfTest(const char* library_path, const char* program_path) {
  void* library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  if (!library) throw std::runtime_error(dlerror());
  auto load = reinterpret_cast<Load>(dlsym(library, "tepl_workflow_load"));
  auto match = reinterpret_cast<Match>(dlsym(library, "tepl_workflow_match"));
  auto destroy =
      reinterpret_cast<Destroy>(dlsym(library, "tepl_workflow_destroy"));
  auto roots = reinterpret_cast<Roots>(dlsym(library, "tepl_workflow_roots"));
  auto depth = reinterpret_cast<Depth>(dlsym(library, "tepl_workflow_depth"));
  const std::vector<std::string> controls = {R"(HloModule controls
ENTRY main {
 a = f32[3,2] parameter(0)
 b = f32[4,3] parameter(1)
 ta = f32[2,3] transpose(a), dimensions={1,0}
 tb = f32[3,4] transpose(b), dimensions={1,0}
 ROOT d = f32[2,4] dot(ta,tb), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})",
                                             R"(HloModule shared_control
ENTRY main {
 a = f32[3,3] parameter(0)
 t = f32[3,3] transpose(a), dimensions={1,0}
 ROOT d = f32[3,3] dot(t,t), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"};
  for (const auto& hlo : controls) {
    auto original = take(ParseAndReturnUnverifiedModule(hlo));
    take(HloVerifier(false, false).Run(original.get()));
    auto baseline = original->Clone("control"),
         integrated = original->Clone("control");
    State states[2];
    char error[4096] = {};
    void* bridge = load(program_path, error, sizeof(error));
    require(bridge, error);
    SetTransposeFoldingWorkflowObserver(observer);
    for (int mode = 0; mode < 2; ++mode) {
      current = &states[mode];
      current->mode = mode;
      current->audit = true;
      current->bridge = bridge;
      current->match = match;
      current->roots = roots(bridge);
      current->depth = depth(bridge);
      take(hook(mode ? integrated.get() : baseline.get(), {},
                TransposeFolding::IsRowColumnTransposeDotOperand,
                TransposeFolding::NeverFoldTranspose));
      take(HloDCE().Run(mode ? integrated.get() : baseline.get()));
    }
    current = nullptr;
    SetTransposeFoldingWorkflowHook(nullptr);
    SetTransposeFoldingWorkflowObserver(nullptr);
    require(states[0].retained_dot == 2 && states[1].tepl == 2 &&
                states[1].retained_dot == 0,
            "adapter did not delegate both operands");
    require(canonical(*baseline) == canonical(*integrated),
            "adapter control HLO mismatch");
    for (int seed = 1; seed <= 3; ++seed) {
      auto args = arguments(*original, seed);
      std::vector<const Literal*> ptrs;
      for (auto& a : args) ptrs.push_back(&a);
      HloEvaluator evaluator;
      auto reference = take(evaluator.Evaluate(*original, ptrs));
      auto native = take(evaluator.Evaluate(*baseline, ptrs));
      auto automatic = take(evaluator.Evaluate(*integrated, ptrs));
      require(literal_comparison::Near(reference, native, ErrorSpec(0, 0), true,
                                       nullptr)
                  .ok(),
              "native reference mismatch");
      require(literal_comparison::Near(reference, automatic, ErrorSpec(0, 0),
                                       true, nullptr)
                  .ok(),
              "TEPL reference mismatch");
    }
    destroy(bridge);
  }
  std::cout << "{\"hlo_controls\":2,\"tepl_applications\":4,\"numerical_"
               "checks\":6}\n";
  return 0;
}
int workflow(int argc, char** argv) {
  require(argc == 7,
          "expected input.hlo library.so program.json output_dir rounds seeds");
  const auto out = std::filesystem::path(argv[4]);
  std::filesystem::create_directories(out);
  int rounds = std::stoi(argv[5]), seeds = std::stoi(argv[6]);
  void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!library) throw std::runtime_error(dlerror());
  auto load = reinterpret_cast<Load>(dlsym(library, "tepl_workflow_load"));
  auto match = reinterpret_cast<Match>(dlsym(library, "tepl_workflow_match"));
  auto destroy =
      reinterpret_cast<Destroy>(dlsym(library, "tepl_workflow_destroy"));
  auto roots = reinterpret_cast<Roots>(dlsym(library, "tepl_workflow_roots"));
  auto depth = reinterpret_cast<Depth>(dlsym(library, "tepl_workflow_depth"));
  require(load && match && destroy && roots && depth, "missing symbols");
  auto source = take(ParseAndReturnUnverifiedModule(read(argv[1])));
  take(HloVerifier(false, false).Run(source.get()));
  source->mutable_config().set_replica_count(1);
  source->mutable_config().set_num_partitions(1);
  write(out / "debug-options.pbtxt",
        source->config().debug_options().DebugString());
  CpuClientOptions options;
  options.asynchronous = false;
  options.cpu_device_count = 1;
  options.max_transpose_threads = 1;
  HloRunner runner(take(GetXlaPjrtCpuClient(std::move(options))));
  SetTransposeFoldingWorkflowObserver(observer);
  SetTransposeFoldingWorkflowHook(hook);
  llvm::json::Array samples;
  std::unique_ptr<OpaqueExecutable> keep[2];
  State states[2];
  bool graphs = true, pass_graphs = true;
  for (int round = -1; round < rounds; ++round) {
    for (int at = 0; at < 2; ++at) {
      int mode = (at + (round >= 0 ? round : 0)) % 2;
      State state;
      state.audit = round == -1;
      state.mode = mode;
      state.match = match;
      current = &state;
      char error[4096] = {};
      auto input = source->Clone("workflow");
      auto begin = std::chrono::steady_clock::now();
      double load_ms = 0;
      if (mode) {
        state.bridge = load(argv[3], error, sizeof(error));
        require(state.bridge, error);
        state.roots = roots(state.bridge);
        state.depth = depth(state.bridge);
        load_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
      }
      auto executable = take(runner.CreateExecutable(std::move(input), true));
      double compile_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - begin)
                              .count();
      if (mode) destroy(state.bridge);
      state.bridge = nullptr;
      current = nullptr;
      require(state.pass_calls > 0,
              "target pass was not called by full compiler pipeline");
      auto* final = take(runner.HloModuleFromWrapped(executable.get()));
      if (round == -1) {
        write(out / (mode ? "tepl.final.hlo" : "baseline.final.hlo"),
              canonical(*final));
        for (size_t i = 0; i < state.before.size(); ++i) {
          write(out / ((mode ? "tepl" : "baseline") + std::string(".before.") +
                       std::to_string(i) + ".hlo"),
                state.before[i]);
          write(out / ((mode ? "tepl" : "baseline") + std::string(".after.") +
                       std::to_string(i) + ".hlo"),
                state.after[i]);
        }
      }
      if (round >= 0)
        samples.push_back(llvm::json::Object{
            {"round", round},
            {"mode", mode ? "tepl" : "baseline"},
            {"compile_ms", compile_ms},
            {"program_load_ms", load_ms},
            {"pass_calls", int64_t(state.pass_calls)},
            {"tepl", int64_t(state.tepl)},
            {"native_dot", int64_t(state.retained_dot)},
            {"native_conv", int64_t(state.retained_conv)},
            {"root_attempts", int64_t(state.scanned)},
            {"proposals", int64_t(state.proposals)},
            {"backend_rejected", int64_t(state.backend_rejected)}});
      states[mode] = std::move(state);
      keep[mode] = std::move(executable);
    }
    if (round >= -1) {
      graphs &= canonical(*take(runner.HloModuleFromWrapped(keep[0].get()))) ==
                canonical(*take(runner.HloModuleFromWrapped(keep[1].get())));
      pass_graphs &= states[0].after == states[1].after;
      require(
          states[0].retained_dot + states[0].retained_conv ==
              states[1].tepl + states[1].retained_dot + states[1].retained_conv,
          "rewrite accounting differs");
    }
  }
  SetTransposeFoldingWorkflowHook(nullptr);
  SetTransposeFoldingWorkflowObserver(nullptr);
  llvm::json::Array checks;
  for (int seed = 0; seed < seeds; ++seed) {
    auto args = arguments(*source, seed + 1);
    std::vector<const Literal*> pointers;
    for (auto& a : args) pointers.push_back(&a);
    auto baseline = take(runner.ExecuteWithExecutable(keep[0].get(), pointers));
    auto integrated =
        take(runner.ExecuteWithExecutable(keep[1].get(), pointers));
    require(finiteOutputs(baseline) && finiteOutputs(integrated),
            "non-finite model output");
    auto status = literal_comparison::Near(
        baseline, integrated, ErrorSpec(1.e-5, 1.e-5), true, nullptr);
    require(status.ok(), status.ToString());
    checks.push_back(llvm::json::Object{{"seed", seed + 1}, {"equal", true}});
  }
  llvm::json::Object result{
      {"samples", std::move(samples)},
      {"outputs", std::move(checks)},
      {"final_graphs_equal", graphs},
      {"pass_graphs_equal", pass_graphs},
      {"baseline_rewrites",
       int64_t(states[0].retained_dot + states[0].retained_conv)},
      {"tepl_rewrites", int64_t(states[1].tepl)},
      {"retained_rewrites",
       int64_t(states[1].retained_dot + states[1].retained_conv)}};
  std::cout
      << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result))).str();
  return 0;
}
}  // namespace xla
int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--self-test")
      return xla::selfTest(argv[2], argv[3]);
    return xla::workflow(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
