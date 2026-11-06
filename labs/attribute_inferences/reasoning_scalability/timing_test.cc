#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/core/pipeline.h"

using namespace tepl;
using namespace tepl::inference;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected dialect path");
    std::ifstream stream(argv[1]);
    std::string dialect{std::istreambuf_iterator<char>(stream), {}};
    auto compiled = [&](std::string rule) {
      auto parsed = parse(dialect + rule, "timing_test.tepl");
      require(parsed.ok(), "timing fixture parses");
      auto output = core::compile(*parsed.program,
                                  {core::SemanticPolicy::kAlgebraic, true});
      require(output.ok(), "timing fixture compiles");
      return std::move(*output.program);
    };
    auto p = compiled("rule r {(transpose A)=>A}");
    const auto& e = p.core(0);
    auto descriptor = *e.original.source.back().descriptor;
    Metadata metadata{{{0, {2, 3}}}, {{descriptor, {{0, 1}}}}};
    auto check = [&](const Emission& code, const Metadata& m, Status expected,
                     Limits limits = {}) {
      auto plain = executeSound(code, m, limits);
      ExecutionTiming timing{123, 123};
      auto start = std::chrono::steady_clock::now();
      auto profiled = executeSound(code, m, limits, &timing);
      double total = std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - start)
                         .count();
      require(plain.status == expected && profiled.status == expected,
              "profiling preserves status");
      require(plain.attributes == profiled.attributes &&
                  plain.layouts == profiled.layouts &&
                  plain.source == profiled.source &&
                  plain.work == profiled.work,
              "profiling preserves output and resource accounting");
      require(timing.feasibility_us >= 0 && timing.construction_us >= 0 &&
                  timing.feasibility_us + timing.construction_us <= total,
              "timers form disjoint subintervals of execution");
      if (expected != Status::Solved)
        require(timing.construction_us == 0,
                "unsuccessful feasibility never executes builders");
    };
    check(e, metadata, Status::Solved);
    metadata.descriptors[descriptor] = {{1, 0}};
    check(e, metadata, Status::Inapplicable);
    metadata.descriptors[descriptor] = {{0, 0}};
    check(e, metadata, Status::InvalidSource);
    check(e, metadata, Status::ResourceLimit, {0, 100});
    auto ext = compiled("rule r {(reverse(reverse A))=>(reverse A)}");
    Metadata view{{{0, {2, 3}}}, {}};
    for (const auto& node : ext.core(0).original.source)
      if (node.descriptor) view.descriptors[*node.descriptor] = {{0}};
    check(ext.core(0), view, Status::Solved);
    std::cout << "profiled/plain success, infeasibility, invalid source, "
                 "limits and extension agree\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
