#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>

#include "nlohmann/json.hpp"
#include "src/core/lowering/lower.h"
#include "src/core/pipeline.h"
#include "src/core/resolution/loading.h"
#include "src/core/semantics/verify.h"
#include "src/rewrite/matcher.h"

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace tepl;
using namespace tepl::inference;
constexpr Limits kLimits{1000000, 1000000};
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
template <class F>
auto timed(double& elapsed, F run) {
  auto start = Clock::now();
  auto result = run();
  elapsed += ms(start, Clock::now());
  return result;
}
template <class T>
void requireResult(const T& result) {
  if (result.ok()) return;
  std::string reason;
  for (const auto& d : result.diagnostics) reason += d.message + "\n";
  throw std::runtime_error(reason);
}
struct Compilation {
  std::optional<executable::ExecutableProgram> program;
  Json sample;
};
Compilation compile(const std::string& source, const std::string& path) {
  Compilation out;
  double preparation = 0, construction = 0, lowering = 0;
  std::string reason;
  auto start = Clock::now();
  try {
    // Keep intermediate representations alive until the end of the total
    // interval. Their teardown belongs to residual "other", not proof time.
    out.program = [&] {
      auto parsed = timed(preparation, [&] { return parse(source, path); });
      requireResult(parsed);
      auto imported =
          timed(preparation, [&] { return loadImports(*parsed.program); });
      requireResult(imported);
      auto resolved = timed(preparation, [&] {
        return core::resolution::resolve(*imported.program);
      });
      requireResult(resolved);
      auto typed = timed(preparation, [&] {
        return core::checking::check(*resolved.program);
      });
      requireResult(typed);
      auto verified = timed(construction, [&] {
        return core::semantics::verify(
            *typed.program, {core::SemanticPolicy::kAlgebraic, true});
      });
      requireResult(verified);
      auto executable = timed(
          lowering, [&] { return core::lowering::lower(*verified.program); });
      requireResult(executable);
      return std::move(*executable.program);
    }();
  } catch (const std::exception& e) {
    reason = e.what();
  }
  auto total = ms(start, Clock::now());
  out.sample = {{"status", out.program ? "success" : "unresolved"},
                {"reason", reason},
                {"total", total},
                {"preparation", preparation},
                {"checking_construction", construction},
                {"lowering_validation", lowering},
                {"other", total - preparation - construction - lowering}};
  return out;
}
std::string read(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path);
  return {std::istreambuf_iterator<char>(stream), {}};
}
Metadata metadata(const Program&, std::size_t);
bool validateExecution(const Emission&, const Metadata&);
void validatePolymorphicCompilation(const executable::ExecutableProgram&,
                                    const Json&);
void validateCompilation(const executable::ExecutableProgram& p,
                         const Json& row) {
  const auto& rule = p.ir().rules.at(0);
  if (p.ir().rules.size() != 1 || !rule.core || !rule.derivations.empty())
    throw std::runtime_error("expected one complete automatic DAG rule");
  const auto& schedule = rule.core->program.replacement;
  auto descriptors =
      std::count_if(schedule.begin(), schedule.end(),
                    [](const auto& n) { return n.descriptor.has_value(); });
  if (descriptors != row.at("expected_constructors").get<int>())
    throw std::runtime_error("incomplete automatic constructors");
  validatePolymorphicCompilation(p, row);
}
// Isolate each compilation so even parsing/admission can be stopped at 10 s.
// Fork, IPC, serialization (runtime setup only) and loading are not phase
// times.
Compilation boundedCompile(const std::string& source, const std::string& path,
                           const Json* row = nullptr, bool retain = false) {
  char output[] = "/tmp/tepl-scaling-compile-XXXXXX";
  int fd = mkstemp(output);
  if (fd < 0) throw std::runtime_error("cannot create compiler result file");
  close(fd);
  auto start = Clock::now();
  pid_t child = fork();
  if (child < 0) {
    std::filesystem::remove(output);
    throw std::runtime_error("cannot fork compiler worker");
  }
  if (child == 0) {
    try {
      auto result = compile(source, path);
      if (result.program && row) {
        validateCompilation(*result.program, *row);
        result.sample["validated_ranks"] = row->at("validation_ranks");
      }
      Json payload{{"sample", result.sample}};
      if (retain && result.program)
        payload["program"] = executable::serialize(*result.program);
      {
        std::ofstream stream(output);
        stream << payload.dump();
      }
      _exit(0);
    } catch (const std::exception& e) {
      {
        std::ofstream stream(output);
        stream << Json{{"error", e.what()}}.dump();
      }
      _exit(1);
    }
  }
  int status = 0;
  bool timeout = false;
  while (waitpid(child, &status, WNOHANG) == 0) {
    if (Clock::now() - start >= std::chrono::seconds(10)) {
      timeout = true;
      kill(child, SIGKILL);
      waitpid(child, &status, 0);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  timeout |= Clock::now() - start >= std::chrono::seconds(10);
  Compilation result;
  if (timeout || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    result.sample = {{"status", timeout ? "time_limit" : "worker_failed"},
                     {"reason", timeout ? "compilation exceeded 10 seconds"
                                        : "compiler worker failed"},
                     {"timing_complete", false},
                     {"total", ms(start, Clock::now())}};
    if (!timeout) {
      auto payload = Json::parse(read(output), nullptr, false);
      if (!payload.is_discarded() && payload.contains("error"))
        result.sample["reason"] = payload.at("error");
    }
  } else {
    Json payload;
    std::ifstream(output) >> payload;
    result.sample = payload.at("sample");
    if (retain && payload.contains("program"))
      result.program =
          executable::load(payload.at("program").get<std::string>());
  }
  std::filesystem::remove(output);
  return result;
}
Metadata metadata(const Program& p, std::size_t rank) {
  Metadata m;
  for (auto capture : p.source.back().captures)
    m.shapes[capture] = std::vector<std::uint64_t>(rank, 2);
  List batch(rank - 2);
  std::iota(batch.begin(), batch.end(), 0);
  for (const auto& node : p.source) {
    if (!node.descriptor) continue;
    if (node.kind != Kind::Contraction)
      throw std::runtime_error("unexpected runtime workload");
    auto& fields = m.descriptors[*node.descriptor];
    fields.resize(4);
    fields[node.fields[0]] = fields[node.fields[1]] = batch;
    fields[node.fields[2]] = {static_cast<std::int64_t>(rank - 1)};
    fields[node.fields[3]] = {static_cast<std::int64_t>(rank - 2)};
  }
  return m;
}
bool validateExecution(const Emission& e, const Metadata& m) {
  ExecutionTiming timing;
  auto plain = executeSound(e, m, kLimits);
  auto profiled = executeSound(e, m, kLimits, &timing);
  if (plain.status != profiled.status ||
      plain.attributes != profiled.attributes ||
      plain.layouts != profiled.layouts || plain.work != profiled.work ||
      timing.feasibility_us < 0 || timing.construction_us < 0)
    throw std::runtime_error(
        "profile/plain execution at rank " +
        std::to_string(m.shapes.begin()->second.size()) + ": " +
        std::to_string(static_cast<int>(plain.status)) + "/" +
        std::to_string(static_cast<int>(profiled.status)) + " " + plain.reason +
        " / " + profiled.reason);
  if (plain.status == Status::ResourceLimit) return false;
  if (plain.status != Status::Solved)
    throw std::runtime_error("runtime correctness preflight failed: " +
                             plain.reason);
  auto completed = m;
  completed.descriptors.insert(plain.attributes.begin(),
                               plain.attributes.end());
  auto replay = replayReplacement(e.original, completed, kLimits);
  if (replay.status != Status::Analyzed)
    throw std::runtime_error("independent RHS legality replay failed: " +
                             replay.reason);
  for (std::size_t i = 0; i < e.blocks.size(); ++i) {
    const auto& built = e.blocks.size() == 1 ? plain : plain.components.at(i);
    const auto& target =
        e.original.components.empty() ? replay : replay.components.at(i);
    if (target.source.classes != built.source.classes ||
        target.source.output != built.source.output)
      throw std::runtime_error("independent RHS correspondence replay failed");
  }
  return true;
}
void validatePolymorphicCompilation(const executable::ExecutableProgram& p,
                                    const Json& row) {
  const auto& rule = p.ir().rules.at(0);
  if (row.at("rank") != "polymorphic" ||
      rule.constraints.size() != rule.captures.size())
    throw std::runtime_error("missing polymorphic shape constraints");
  for (const auto& constraint : rule.constraints)
    if (constraint.shape.size() != 3 ||
        constraint.shape.front().kind != core::ShapeElement::Kind::kSequence)
      throw std::runtime_error("compile workload specializes tensor rank");
  const auto& e = p.core(0);
  for (std::size_t rank : row.at("validation_ranks")) {
    auto m = metadata(e.original, rank);
    for (auto& [id, shape] : m.shapes) {
      const auto& name = rule.captures.at(id).name;
      auto at = name.find_last_of('A');
      if (at == std::string::npos)
        throw std::runtime_error("unexpected generated capture");
      auto index = std::stoul(name.substr(at + 1));
      shape.at(rank - 2) = 2 + (index - 1) % 2;
      shape.at(rank - 1) = 2 + index % 2;
    }
    if (!validateExecution(e, m))
      throw std::runtime_error(
          "polymorphic validation exceeded metadata work limit");
    // Native matching binds the Batch sequence and all scalar dimensions;
    // evaluateWhere checks the actual written C before executing q and f.
    core::ConcreteGraph graph{};
    std::vector<std::size_t> nodes;
    for (const auto& n : e.original.source) {
      core::ConcreteNode node{};
      if (n.kind == Kind::Input) {
        node.kind = core::ConcreteNode::Kind::kInput;
        node.text = rule.captures.at(*n.capture).name;
        node.dtype = core::DType::kF32;
        node.shape = m.shapes.at(*n.capture);
      } else {
        node.kind = core::ConcreteNode::Kind::kOperation;
        node.operation = core::OpId{*n.operation};
        for (auto input : n.inputs) node.operands.push_back(nodes.at(input));
        if (n.descriptor)
          for (const auto& field : m.descriptors.at(*n.descriptor)) {
            core::AttributeValue list{
                core::AttributeValue::Kind::kList, {}, {}};
            for (auto value : field)
              list.elements.push_back({core::AttributeValue::Kind::kInteger,
                                       std::to_string(value),
                                       {}});
            node.attributes.push_back(std::move(list));
          }
      }
      nodes.push_back(graph.nodes.size());
      graph.nodes.push_back(std::move(node));
    }
    graph.root = nodes.back();
    rewrite::Budget budget;
    for (std::size_t i = 0; i < graph.nodes.size(); ++i)
      rewrite::infer(p.ir(), graph, i, budget);
    rewrite::Bindings bindings;
    if (!rewrite::match(p.ir(), rule, graph, graph.root, bindings, budget))
      throw std::runtime_error("polymorphic shape matching failed");
    rewrite::Evaluator evaluator(p, graph, 0, bindings, budget);
    if (!evaluator.evaluateWhere())
      throw std::runtime_error(
          "polymorphic C && q rejected legal batch-prefix metadata");
  }
}
Json runtimeBatch(const Emission& e, const Metadata& m, std::size_t iterations,
                  bool profile) {
  double feasibility = 0, construction = 0;
  std::size_t checksum = 0;
  bool success = true;
  auto start = Clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    ExecutionTiming timing;
    auto result = executeSound(e, m, kLimits, profile ? &timing : nullptr);
    success &= result.status == Status::Solved;
    checksum += result.attributes.size() + result.work;
    feasibility += timing.feasibility_us;
    construction += timing.construction_us;
    // Output destruction is inside the total interval and outside q/f timers.
  }
  double total = ms(start, Clock::now()) * 1000 / iterations;
  feasibility /= iterations;
  construction /= iterations;
  return {{"status", success ? "success" : "execution_failed"},
          {"total", total},
          {"feasibility", feasibility},
          {"attribute_construction", construction},
          {"other", total - feasibility - construction},
          {"checksum", checksum},
          {"iterations", iterations}};
}
}  // namespace
int main(int argc, char** argv) {
  try {
#ifndef __OPTIMIZE__
    throw std::runtime_error("benchmark requires an optimized build (-c opt)");
#endif
    if (argc != 5)
      throw std::runtime_error(
          "usage: inference_benchmark MANIFEST OUTPUT WARMUPS REPETITIONS");
    Json manifest;
    std::ifstream(argv[1]) >> manifest;
    int warmups = std::stoi(argv[3]), repetitions = std::stoi(argv[4]);
    if (warmups < 0 || repetitions < 2)
      throw std::runtime_error("invalid repetitions");
    std::vector<std::string> sources;
    for (const auto& row : manifest.at("cases"))
      sources.push_back(read(row.at("file")));
    // Exactly one compilation for all runtime ranks. No loading, emission,
    // synthesis, or symbolic proof is performed by a timed metadata execution.
    auto runtime = boundedCompile(read(manifest.at("runtime_file")),
                                  manifest.at("runtime_file"), nullptr, true);
    if (!runtime.program) throw std::runtime_error(runtime.sample.dump());
    const auto& emission = runtime.program->core(0);
    if (!runtime.program->ir().rules[0].conditions.empty())
      throw std::runtime_error(
          "runtime rule must be rank independent with C=true");
    std::vector<Metadata> inputs;
    std::vector<std::size_t> batches;
    std::vector<bool> active(sources.size() + manifest.at("ranks").size(),
                             true);
    Json outcomes = Json::array(), preflight = Json::array();
    // Ascending preflight in each series. Never attempt a larger size after
    // that series has failed. The same policy applies to later repetitions.
    std::string stopped;
    std::string panel;
    for (std::size_t i = 0; i < sources.size(); ++i) {
      const auto& row = manifest.at("cases").at(i);
      if (panel != row.at("panel").get<std::string>()) {
        panel = row.at("panel");
        stopped.clear();
      }
      Json outcome{{"case_id", row.at("case_id")},
                   {"panel", panel},
                   {"x", row.at("x")},
                   {"status", "success"},
                   {"reason", ""}};
      if (stopped.empty()) {
        auto attempt = boundedCompile(sources[i], row.at("file"), &row);
        auto sample = attempt.sample;
        sample["case_id"] = row.at("case_id");
        preflight.push_back(sample);
        if (sample.at("status") != "success") {
          stopped = row.at("case_id").get<std::string>() + ": " +
                    sample.at("reason").get<std::string>();
          outcome["status"] = sample.at("status");
          outcome["reason"] = stopped;
        }
      } else {
        outcome["status"] = "not_run";
        outcome["reason"] = "series stopped after " + stopped;
      }
      if (outcome.at("status") != "success") active[i] = false;
      outcomes.push_back(outcome);
    }
    stopped.clear();
    for (auto rank : manifest.at("ranks")) {
      auto index = inputs.size() + sources.size();
      Json outcome{{"case_id", "rank_" + rank.dump()},
                   {"panel", "c"},
                   {"x", rank},
                   {"status", "success"},
                   {"reason", ""}};
      auto m = metadata(emission.original, rank);
      std::size_t count = 1;
      if (stopped.empty()) {
        bool success = validateExecution(emission, m);
        auto pilot = runtimeBatch(emission, m, 5, false);
        pilot["case_id"] = outcome.at("case_id");
        preflight.push_back(pilot);
        if (!success) {
          stopped = outcome.at("case_id").get<std::string>() +
                    ": metadata work limit";
          outcome["status"] = "resource_limit";
          outcome["reason"] = stopped;
        } else {
          count = std::clamp<std::size_t>(
              static_cast<std::size_t>(
                  10000.0 / std::max(0.001, pilot.at("total").get<double>())),
              1, 2000);
        }
      } else {
        outcome["status"] = "not_run";
        outcome["reason"] = "series stopped after " + stopped;
      }
      if (outcome.at("status") != "success") active[index] = false;
      outcomes.push_back(outcome);
      inputs.push_back(std::move(m));
      batches.push_back(count);
    }
    Json records = Json::array();
    std::vector<std::size_t> order(sources.size() + inputs.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(manifest.at("seed").get<unsigned>());
    for (int round = -warmups; round < repetitions; ++round) {
      std::shuffle(order.begin(), order.end(), random);
      for (auto index : order) {
        if (!active[index]) continue;
        Json sample;
        if (index < sources.size()) {
          const auto& row = manifest.at("cases").at(index);
          auto compiled = boundedCompile(sources[index], row.at("file"), &row);
          sample = compiled.sample;

          sample["case_id"] = row.at("case_id");
          sample["panel"] = row.at("panel");
          sample["x"] = row.at("x");
        } else {
          auto i = index - sources.size();
          Json unprofiled;
          if (round % 2 == 0) {
            sample = runtimeBatch(emission, inputs[i], batches[i], true);
            unprofiled = runtimeBatch(emission, inputs[i], batches[i], false);
          } else {
            unprofiled = runtimeBatch(emission, inputs[i], batches[i], false);
            sample = runtimeBatch(emission, inputs[i], batches[i], true);
          }
          if (unprofiled.at("status") != "success")
            sample["status"] = "execution_failed";
          sample["unprofiled_total"] = unprofiled.at("total");
          sample["case_id"] = "rank_" + manifest.at("ranks").at(i).dump();
          sample["panel"] = "c";
          sample["x"] = manifest.at("ranks").at(i);
        }
        if (sample.at("status") != "success") {
          for (std::size_t j = 0; j < outcomes.size(); ++j)
            if (outcomes[j].at("panel") == sample.at("panel") &&
                outcomes[j].at("x") >= sample.at("x")) {
              active[j] = false;
              outcomes[j]["status"] =
                  j == index ? sample.at("status") : Json("stopped");
              outcomes[j]["reason"] = "series stopped after " +
                                      sample.at("case_id").get<std::string>();
            }
        }
        sample["round"] = round;
        sample["stage"] = round < 0 ? "warmup" : "measured";
        records.push_back(std::move(sample));
      }
      std::cerr << "round " << round << " completed\n";
    }
    std::ofstream(argv[2]) << Json{{"schema_version", 4},
                                   {"records", records},
                                   {"runtime_compilations", 1},
                                   {"compile_timeout_seconds", 10},
                                   {"case_outcomes", outcomes},
                                   {"preflight", preflight},
                                   {"runtime_compile", runtime.sample},
                                   {"runtime_limits",
                                    {{"work", kLimits.work},
                                     {"axes", kLimits.axes}}}}
                                  .dump(2)
                           << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
