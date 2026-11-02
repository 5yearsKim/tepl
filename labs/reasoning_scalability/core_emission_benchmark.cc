#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>

#include "src/core/attr_synth/ordered/compile.h"
#include "src/core/pipeline.h"
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("expected manifest and output");
    Json manifest;
    std::ifstream(argv[1]) >> manifest;
    Json records = Json::array();
    for (const auto& row : manifest.at("cases")) {
      if (row.at("mode") != "automatic") continue;
      std::string path = row.at("file");
      std::ifstream f(path);
      std::string source{std::istreambuf_iterator<char>(f), {}};
      auto parsed = tepl::parse(source, path);
      if (!parsed.ok()) throw std::runtime_error("parse failed");
      tepl::core::InspectionOptions options{
          tepl::core::SemanticPolicy::kAlgebraic, true};
      options.defer_semantic_proofs = true;
      auto inspected = tepl::core::inspect(*parsed.program, options);
      if (!inspected.ok()) throw std::runtime_error("inspection failed");
      const auto& plan = *inspected.program->rules[0].semantic_plan;
      for (int round = -1; round < 10; ++round) {
        auto t = Clock::now();
        auto compiled = tepl::core::attr_synth::ordered::compile(plan);
        double emit_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t).count();
        if (!compiled.emission) throw std::runtime_error(compiled.reason);
        tepl::inference::Metadata metadata;
        unsigned rank = row.at("rank");
        for (auto cap : compiled.program->source.back().captures)
          metadata.shapes[cap] = std::vector<std::uint64_t>(rank, 2);
        for (const auto& ins : compiled.program->source)
          if (ins.descriptor) {
            auto& fields = metadata.descriptors[*ins.descriptor];
            fields.resize(ins.fields.size());
            if (ins.kind == tepl::inference::Kind::Contraction) {
              fields[ins.fields[0]] = {0};
              fields[ins.fields[1]] = {0};
              fields[ins.fields[2]] = {2};
              fields[ins.fields[3]] = {1};
            } else if (ins.kind == tepl::inference::Kind::Permutation)
              for (unsigned k = rank; k-- > 0;)
                fields[ins.fields[0]].push_back(k);
          }
        t = Clock::now();
        auto actual = tepl::inference::execute(*compiled.emission, metadata);
        double eval_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t).count();
        if (actual.status != tepl::inference::Status::Solved)
          throw std::runtime_error("emitted execution failed");
        // Separate constructor and independent positional replay, outside
        // timing.
        auto reference = tepl::inference::evaluate(*compiled.program, metadata);
        auto completed = metadata;
        completed.descriptors.insert(actual.attributes.begin(),
                                     actual.attributes.end());
        auto replay =
            tepl::inference::replayReplacement(*compiled.program, completed);
        if (reference.status != actual.status ||
            replay.status != tepl::inference::Status::Analyzed ||
            replay.source.classes != actual.source.classes ||
            replay.source.output != actual.source.output)
          throw std::runtime_error("independent replay disagreement");
        std::size_t nodes = 0;
        for (const auto& b : compiled.emission->blocks)
          nodes += b.expressions.size();
        records.push_back({{"case", row.at("case_id")},
                           {"round", round},
                           {"emit_ms", emit_ms},
                           {"execute_ms", eval_ms},
                           {"expressions", nodes},
                           {"status", "solved"}});
      }
    }
    std::ofstream(argv[2])
        << Json{{"records", records}, {"warmups", 1}, {"repetitions", 10}}.dump(
               2)
        << "\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
