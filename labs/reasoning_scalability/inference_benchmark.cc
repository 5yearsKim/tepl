#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "src/core/pipeline.h"
#include "src/core/semantics/prove_rule.h"

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace tepl::core::semantics;
Plan prepare(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path);
  std::string source{std::istreambuf_iterator<char>(stream), {}};
  auto parsed = tepl::parse(source, path);
  if (!parsed.ok()) throw std::runtime_error("cannot parse " + path);
  auto inspected = tepl::core::inspect(
      *parsed.program, {tepl::core::SemanticPolicy::kAlgebraic, true});
  if (!inspected.ok()) {
    for (const auto& d : inspected.diagnostics) std::cerr << d.message << '\n';
    throw std::runtime_error("cannot inspect " + path);
  }
  if (inspected.program->rules.size() != 1 ||
      !inspected.program->rules[0].semantic_plan)
    throw std::runtime_error("expected one semantic rule in " + path);
  auto plan = *inspected.program->rules[0].semantic_plan;
  plan.proof = {};  // Repeated calls cannot reuse the preparation proof.
  return plan;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 6)
      throw std::runtime_error(
          "usage: inference_benchmark MANIFEST OUTPUT WARMUPS REPETITIONS "
          "TIMEOUT_MS");
    Json manifest;
    std::ifstream(argv[1]) >> manifest;
    const int warmups = std::stoi(argv[3]), repetitions = std::stoi(argv[4]);
    const auto timeout = std::chrono::milliseconds(std::stoll(argv[5]));
    if (warmups < 0 || repetitions < 1 || timeout.count() <= 0)
      throw std::runtime_error("invalid measurement settings");
    std::vector<Plan> plans;
    for (const auto& row : manifest.at("cases"))
      plans.push_back(prepare(row.at("file")));
    Json records = Json::array();
    std::vector<std::size_t> order(plans.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(manifest.at("seed").get<unsigned>());
    // Parsing, import loading, type checking, and initial inspection are above
    // the timer. Each proveRule invocation creates fresh reasoning contexts.
    for (int round = -warmups; round < repetitions; ++round) {
      std::shuffle(order.begin(), order.end(), random);
      for (auto index : order) {
        const auto& row = manifest.at("cases").at(index);
        ProofTiming timing;
        const auto start = Clock::now();
        auto proof =
            proveRule(plans[index], std::numeric_limits<std::size_t>::max(),
                      &timing, start + timeout);
        const auto total =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
        std::string status =
            proof.status == ProofStatus::Proved ? "proved"
            : proof.reason.find("time limit exceeded:") != std::string::npos
                ? "time_limit"
            : proof.reason.find("budget exhausted:") != std::string::npos
                ? "budget_exhausted"
                : "unresolved";
        if (status == "proved" &&
            (!proof.guard.empty() ||
             proof.constructors.size() !=
                 row.at("expected_constructors").get<std::size_t>()))
          throw std::runtime_error(
              "incomplete or conditional constructors for " +
              row.at("case_id").get<std::string>());
        records.push_back(
            {{"case_id", row.at("case_id")},
             {"stage", round < 0 ? "warmup" : "measured"},
             {"round", round},
             {"status", status},
             {"reason", proof.reason},
             {"total_ms", total},
             {"construction_ms", timing.construction_ms},
             {"verification_ms", timing.verification_ms},
             {"other_ms", std::max(0.0, total - timing.construction_ms -
                                            timing.verification_ms)},
             {"construction_attempts", timing.construction_attempts},
             {"verification_attempts", timing.verification_attempts},
             {"constructors", proof.constructors.size()}});
      }
      std::cerr << "round " << round << " completed\n";
    }
    std::ofstream(argv[2]) << Json{{"records", records}}.dump(2) << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
