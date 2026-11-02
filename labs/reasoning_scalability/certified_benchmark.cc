// Certified compilation, split at the same API boundaries as core::compile.
// Parsing/import loading, validation controls, and JSON reporting are untimed.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "src/core/pipeline.h"
#include "src/core/resolution/loading.h"
#include "src/core/lowering/lower.h"

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace tepl;
constexpr core::SemanticOptions kCertified{core::SemanticPolicy::kAlgebraic,
                                           false, true};
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
template <class T> void requireOk(const T& result) {
  if (result.ok()) return;
  std::string error;
  for (const auto& d : result.diagnostics) error += d.message + "\n";
  throw std::runtime_error(error);
}
struct Compilation {
  executable::ExecutableProgram program;
  double preparation_ms, certification_ms, lowering_ms;
};
Compilation compilePhases(const core::resolution::SourceProject& project) {
  const auto start = Clock::now();
  auto resolved = core::resolution::resolve(project);
  requireOk(resolved);
  auto typed = core::checking::check(*resolved.program);
  requireOk(typed);
  const auto prepared = Clock::now();
  auto verified = core::semantics::verify(*typed.program, kCertified);
  requireOk(verified);
  const auto certified = Clock::now();
  auto lowered = core::lowering::lower(*verified.program);
  requireOk(lowered);
  const auto finished = Clock::now();
  return {std::move(*lowered.program), ms(start, prepared),
          ms(prepared, certified), ms(certified, finished)};
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6 || std::string(argv[5]) != "--certified")
      throw std::runtime_error(
          "usage: certified_benchmark MANIFEST OUTPUT WARMUPS REPETITIONS --certified");
    Json manifest;
    std::ifstream(argv[1]) >> manifest;
    const int warmups = std::stoi(argv[3]), repetitions = std::stoi(argv[4]);
    if (warmups < 0 || repetitions < 1) throw std::runtime_error("invalid rounds");
    std::vector<core::resolution::SourceProject> projects;
    Json controls = Json::array();
    for (const auto& row : manifest.at("cases")) {
      const auto path = row.at("file").get<std::string>();
      std::ifstream input(path);
      if (!input) throw std::runtime_error("cannot open " + path);
      const std::string source{std::istreambuf_iterator<char>(input), {}};
      auto parsed = parse(source, path);
      requireOk(parsed);
      auto loaded = loadImports(*parsed.program);
      requireOk(loaded);
      projects.push_back(std::move(*loaded.program));
      // Independent public-pipeline invocation catches stage omissions, while
      // serialize/load also checks portable acceptance. None is timed.
      auto direct = core::compile(projects.back(), kCertified);
      requireOk(direct);
      auto split = compilePhases(projects.back());
      const auto bytes = executable::serialize(*direct.program);
      if (bytes != executable::serialize(split.program))
        throw std::runtime_error("phase/public pipeline disagreement: " + path);
      auto portable = executable::load(bytes);
      if (executable::serialize(portable) != bytes ||
          portable.ir().rules.size() != 1 || !portable.ir().rules[0].core)
        throw std::runtime_error("missing certified portable program: " + path);
      // Selecting both mode flags is excluded by the CLI; verify that this
      // driver's fixed options are exactly the --certified selection.
      if (kCertified.core_backend || !kCertified.certified_backend)
        throw std::runtime_error("wrong backend");
      const auto program_path = row.at("program_file").get<std::string>();
      std::ofstream(program_path) << bytes;
      controls.push_back({{"case_id", row.at("case_id")},
                          {"public_pipeline_equal", true},
                          {"portable_roundtrip", true}});
    }
    Json records = Json::array();
    std::vector<std::size_t> order(projects.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(manifest.at("seed").get<unsigned>());
    for (int round = -warmups; round < repetitions; ++round) {
      std::shuffle(order.begin(), order.end(), random);
      for (auto index : order) {
        const auto start = Clock::now();
        auto result = compilePhases(projects[index]);
        const auto total = ms(start, Clock::now());
        const auto phases = result.preparation_ms + result.certification_ms +
                            result.lowering_ms;
        if (!result.program.ir().rules.at(0).core || phases > total)
          throw std::runtime_error("invalid sample");
        records.push_back({{"case_id", manifest.at("cases")[index].at("case_id")},
                           {"stage", round < 0 ? "warmup" : "measured"},
                           {"round", round}, {"status", "accepted"},
                           {"total_ms", total},
                           {"preparation_ms", result.preparation_ms},
                           {"certification_ms", result.certification_ms},
                           {"lowering_ms", result.lowering_ms},
                           {"other_ms", total - phases}});
      }
      std::cerr << "certified round " << round << " completed\n";
    }
    std::ofstream(argv[2]) << Json({{"backend", "certified"},
                                   {"controls", controls},
                                   {"records", records}}).dump(2) << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
