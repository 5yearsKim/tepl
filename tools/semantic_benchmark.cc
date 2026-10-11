#include <z3.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/analyze.h"
#include "src/core/semantics/solver.h"
#include "src/parse.h"

namespace {
#ifdef __OPTIMIZE__
const std::string build_mode = "optimized";
#else
const std::string build_mode = "unoptimized";
#endif
using namespace tepl::core;
using semantics::List;
using semantics::SolveOptions;
using semantics::Status;
struct Case {
  std::string name, rule;
  std::vector<List> shapes, known, pins;
};
std::string quote(const std::string& s) {
  std::string out = "\"";
  for (auto c : s) out += c == '"' ? "\"\"" : std::string(1, c);
  return out + "\"";
}
std::string lists(const std::vector<List>& xs) {
  std::string s;
  for (auto& x : xs) {
    s += '[';
    for (auto v : x) s += std::to_string(v) + ' ';
    s += ']';
  }
  return s;
}
const std::string extra = R"(
dialect Benchmark {
  op unused(x:tensor)->tensor {
    attrs { axes:index[]; }
    dtype(t){yield t;}
    shape(s){ assert is_valid_axis_list(attrs.axes,len(s)); yield s; }
    semantics(i){ yield sum(k in index_space(gather(shape(x),attrs.axes))) { read(x,i) }; }
  }
  op twice(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){ yield sum(k in index_space([2])) { read(x,i) } +
                       sum(j in index_space([2])) { read(x,i) }; }
  }
  op four(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){yield 2 * sum(q in index_space([2])) { read(x,i) };}
  }
  op unit(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){yield sum(k in index_space([1])) { read(x,i) };}
  }
}
use Benchmark::{unused,twice,four,unit};
)";
std::vector<Case> cases() {
  std::vector<Case> out;
  for (std::size_t rank : {2, 3, 5}) {
    List batch(rank - 2);
    std::iota(batch.begin(), batch.end(), 0);
    List a(rank, 2), b(rank, 2), c(rank, 2);
    a[rank - 1] = 3;
    b[rank - 2] = 3;
    b[rank - 1] = 4;
    c[rank - 2] = 4;
    c[rank - 1] = 5;
    std::vector<List> inner{{rank - 1}, {rank - 2}, batch, batch};
    auto both = inner;
    both.insert(both.end(), inner.begin(), inner.end());
    auto suffix = "_r" + std::to_string(rank);
    out.push_back(
        {"transpose" + suffix,
         "rule r { (dot[@d] A B) => (dot[auto] (transpose[auto] A) B) }",
         {a, b},
         inner});
    out.push_back({"distribution" + suffix,
                   "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) "
                   "(dot[auto] B C)) }",
                   {a, a, b},
                   inner});
    out.push_back({"association" + suffix,
                   "rule r { (dot[@o] (dot[@i] A B) C) => (dot[auto] A "
                   "(dot[auto] B C)) }",
                   {a, b, c},
                   both});
    List low_a = b, low_b = c;
    low_a[rank - 1] = 2;
    low_b[rank - 2] = 2;
    low_b[rank - 1] = 4;
    out.push_back({"lora" + suffix,
                   "rule r { (dot[@o] X (add W (dot[@i] A B))) => (add "
                   "(dot[auto] X W) (dot[auto] (dot[auto] X A) B)) }",
                   {a, b, low_a, low_b},
                   both});
  }
  for (std::uint64_t extent : {0, 1, 2}) {
    auto suffix = "_extent" + std::to_string(extent);
    out.push_back({"distribution" + suffix,
                   "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) "
                   "(dot[auto] B C)) }",
                   {{2, extent}, {2, extent}, {extent, 3}},
                   {{1}, {0}, {}, {}}});
    out.push_back({"unused" + suffix,
                   "rule r { (unused[@u] X) => (transpose[auto] X) }",
                   {{extent, 3}},
                   {{0}}});
  }
  out.push_back({"multi_axes",
                 "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) "
                 "(dot[auto] B C)) }",
                 {{2, 3, 2, 3}, {2, 3, 2, 3}, {2, 2, 3, 4}},
                 {{2, 3}, {1, 2}, {0}, {0}}});
  out.push_back({"equal_extents",
                 "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) "
                 "(dot[auto] B C)) }",
                 {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}},
                 {{2}, {1}, {0}, {0}}});
  out.push_back({"pinned_cycle",
                 "rule r { (dot[@d] A B) => (dot[auto] "
                 "(transpose[auto,permutation=[1,2,0]] A) B) }",
                 {{3, 2, 4}, {3, 5}},
                 {{0}, {0}, {}, {}},
                 {{1, 2, 0}}});
  out.push_back({"ordering_impossible",
                 "rule r { (dot[@d] A B) => (dot[auto] "
                 "(transpose[auto,permutation=[1,2,0]] A) B) }",
                 {{2, 4, 3}, {3, 5}},
                 {{2}, {0}, {}, {}},
                 {{1, 2, 0}}});
  out.push_back({"drop_impossible",
                 "rule r { (dot[@d] A B) => (transpose[auto] A) }",
                 {{2, 3}, {3, 3}},
                 {{1}, {0}, {}, {}}});
  out.push_back({"repeated_symmetry",
                 "rule r { (dot[@d] A A) => (transpose[auto,permutation=[1,0]] "
                 "(dot[@d] A A)) }",
                 {{3}},
                 {{}, {}, {}, {}, {}, {}, {}, {}},
                 {{1, 0}}});
  out.push_back({"repeated_asymmetric",
                 "rule r { (dot[@d] A A) => (transpose[auto,permutation=[1,0]] "
                 "(dot[@d] A A)) }",
                 {{2, 2}},
                 {{1}, {0}, {}, {}, {1}, {0}, {}, {}},
                 {{1, 0}}});
  out.push_back(
      {"unit", "rule r { (unit X) => (transpose[auto] X) }", {{2, 3}}});
  out.push_back({"dynamic_rhs_unit",
                 "rule r { X => (unused[auto,axes=[0]] (transpose[auto] X)) }",
                 {{1, 3}},
                 {},
                 {{0}}});
  out.push_back({"alpha_terms",
                 "rule r { (twice X) => (transpose[auto] (four X)) }",
                 {{2, 3}}});
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    unsigned repeats = 3, warmup = 1, timeout = 500, seed = 0;
    std::size_t budget = 1000000, max_candidates = 64;
    std::string filter, mode = "all";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cerr
            << "semantic_benchmark [--repeats N] [--warmup N] [--timeout-ms N] "
               "[--budget N] [--max-candidates N] [--seed N] [--filter TEXT] "
               "[--mode all|metadata|compiled|production]\nCSV is written to "
               "stdout.\n";
        return 0;
      }
      if (++i == argc) throw std::runtime_error("missing option value");
      auto value = std::string(argv[i]);
      if (arg == "--repeats")
        repeats = std::stoul(value);
      else if (arg == "--warmup")
        warmup = std::stoul(value);
      else if (arg == "--timeout-ms")
        timeout = std::stoul(value);
      else if (arg == "--budget")
        budget = std::stoull(value);
      else if (arg == "--max-candidates")
        max_candidates = std::stoull(value);
      else if (arg == "--seed")
        seed = std::stoul(value);
      else if (arg == "--filter")
        filter = value;
      else if (arg == "--mode")
        mode = value;
      else
        throw std::runtime_error("unknown option: " + arg);
    }
    if (!repeats || !timeout || !budget ||
        (mode != "all" && mode != "metadata" && mode != "compiled" &&
         mode != "production"))
      throw std::runtime_error("invalid benchmark settings");
    std::string error;
    auto runfiles = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    if (!runfiles) throw std::runtime_error(error);
    std::ifstream in(
        runfiles->Rlocation("_main/examples/semantic/dialects/tensor.tepl"));
    std::string dialect{std::istreambuf_iterator<char>(in), {}};
    if (dialect.empty()) throw std::runtime_error("cannot read TensorLang");
    struct Compiled {
      inference::SemanticPlan plan;
      double compile_ms;
    };
    std::map<std::string, Compiled> cache;
    std::cout << "case,mode,repeat,shapes,known,pins,compile_ms,total_ms,"
                 "metadata_ms,constraints_ms,z3_ms,verification_ms,ordered_ms,"
                 "solver_calls,candidates,rejected_candidates,constraint_"
                 "attempted,fallback,status,method,reason,fallback_reason,"
                 "timeout_ms,budget,max_candidates,seed,z3_version,compiler,"
                 "build_mode,preferred_attempted,plan_available\n";
    std::cout << std::fixed << std::setprecision(6);
    std::size_t selected = 0;
    for (auto& c : cases()) {
      if (c.name.find(filter) == std::string::npos) continue;
      ++selected;
      if (!cache.contains(c.rule)) {
        auto start = std::chrono::steady_clock::now();
        auto parsed = tepl::parse(dialect + extra + c.rule, "benchmark.tepl");
        if (!parsed.ok())
          throw std::runtime_error("benchmark parsing failed: " + c.name);
        auto analyzed = tepl::core::analyze(
            *parsed.program, {SemanticPolicy::kAlgebraic, timeout});
        if (!analyzed.ok()) {
          for (auto& d : analyzed.diagnostics) std::cerr << d.message << '\n';
          throw std::runtime_error("benchmark analysis failed: " + c.name);
        }
        auto p = *analyzed.program->rules.at(0).inference->semantic;
        auto elapsed = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();
        cache.emplace(c.rule, Compiled{std::move(p), elapsed});
      }
      const auto& compiled = cache.at(c.rule);
      for (std::string selected_mode : {"metadata", "compiled", "production"}) {
        if (mode != "all" && mode != selected_mode) continue;
        SolveOptions options;
        options.semantic_constraints = selected_mode != "metadata";
        options.preferred_candidate = false;
        options.random_seed = seed;
        options.max_candidates = max_candidates;
        for (unsigned repeat = 0; repeat < warmup + repeats; ++repeat) {
          auto s =
              selected_mode == "production"
                  ? semantics::solve(compiled.plan, c.shapes, c.known, c.pins,
                                     budget)
                  : semantics::solveGeneric(compiled.plan, c.shapes, c.known,
                                            c.pins, budget, options);
          if (repeat < warmup) continue;
          auto& t = s.statistics;
          std::cout << quote(c.name) << ',' << selected_mode << ','
                    << repeat - warmup << ',' << quote(lists(c.shapes)) << ','
                    << quote(lists(c.known)) << ',' << quote(lists(c.pins))
                    << ',' << compiled.compile_ms << ',' << t.total_ms << ','
                    << t.metadata_ms << ',' << t.constraints_ms << ','
                    << t.z3_ms << ',' << t.verification_ms << ','
                    << t.ordered_ms << ',' << t.solver_calls << ','
                    << t.candidates << ',' << t.rejected_candidates << ','
                    << t.constraint_attempted << ',' << t.constraint_fallback
                    << ','
                    << (s.status == Status::Found     ? "found"
                        : s.status == Status::Invalid ? "invalid"
                                                      : "unknown")
                    << ',' << s.method << ',' << quote(s.reason) << ','
                    << quote(t.fallback_reason) << ',' << timeout << ','
                    << budget << ','
                    << (selected_mode == "production" ? 0 : max_candidates)
                    << ',' << (selected_mode == "production" ? 0 : seed) << ','
                    << quote(Z3_get_full_version()) << ',' << quote(__VERSION__)
                    << ',' << build_mode << ',' << t.preferred_attempted << ','
                    << compiled.plan.constraints.available << '\n';
        }
      }
    }
    if (!selected) throw std::runtime_error("no cases match filter");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
