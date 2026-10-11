#include "src/core/semantics/axis_domains.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/analyze.h"
#include "src/core/semantics/solver.h"
#include "src/parse.h"

namespace {
using namespace tepl::core;
using semantics::List;
using semantics::Status;
std::string dialect;
void check(bool okay, const std::string& message) {
  if (!okay) throw std::runtime_error(message);
}
AnalysisResult analyze(const std::string& rule,
                       const std::string& definitions = "") {
  auto parsed = tepl::parse(dialect + definitions + rule, "axis_domains.tepl");
  check(parsed.ok(), "parse propagation fixture");
  auto out =
      tepl::core::analyze(*parsed.program, {SemanticPolicy::kAlgebraic, 10000});
  for (auto& d : out.diagnostics) std::cerr << d.message << '\n';
  check(out.ok(), "analyze propagation fixture");
  return out;
}
const inference::SemanticPlan& plan(const AnalysisResult& a) {
  return *a.program->rules.at(0).inference->semantic;
}
semantics::Solution solve(const inference::SemanticPlan& p,
                          const std::vector<List>& shapes,
                          const std::vector<List>& known = {},
                          const std::vector<List>& pins = {},
                          bool propagate = true) {
  semantics::SolveOptions options;
  options.axis_propagation = propagate;
  options.preferred_candidate = false;
  return semantics::solveGeneric(p, shapes, known, pins, 10000000, options);
}
void found(const semantics::Solution& s, const std::string& message) {
  if (s.status != Status::Found)
    std::cerr << message << ": " << s.reason << ", "
              << s.statistics.propagation_reason << '\n';
  check(s.status == Status::Found, message);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    check(argc == 2, "dialect argument");
    std::string error;
    auto files = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    check(bool(files), error);
    std::ifstream input(files->Rlocation(argv[1]));
    dialect = {std::istreambuf_iterator<char>(input), {}};
    auto transpose =
        analyze("rule r { (transpose[@p] X) => (transpose[auto] X) }");
    for (std::size_t rank : {0, 1, 10, 32, 100, 200}) {
      List shape(rank, 2), permutation(rank);
      std::iota(permutation.rbegin(), permutation.rend(), 0);
      auto s = solve(plan(transpose), {shape}, {permutation});
      found(s, "large permutation through generic propagation");
      check(s.fields.back() == permutation,
            "equal extents do not merge ordered axes");
      check(s.statistics.propagation_direct &&
                s.statistics.symbolic_axis_variables == 0 &&
                s.statistics.solver_calls == 0,
            "forced permutation requires no SMT search");
    }
    // Check every inferred domain against every valid small concrete witness.
    // A different supported DSL operation exercises gather rather than scatter.
    auto gather = analyze("rule r { (transpose[@p] X) => (route[auto] X) }", R"(
      dialect Custom {
        op route(x:tensor)->tensor {
          attrs { order:index[]; }
          dtype(t){yield t;}
          shape(s){assert is_valid_axis_list(attrs.order,+len(s)); assert len(attrs.order)==len(s); yield gather(s,attrs.order);}
          semantics(i){yield read(x,gather(i,attrs.order));}
        }
      }
      use Custom::{route};
    )");
    for (List shape : {List{2, 2, 2}, List{0, 1, 2}, List{1, 1, 1}}) {
      List permutation{0, 1, 2};
      do {
        auto a = solve(plan(gather), {shape}, {permutation});
        auto b = solve(plan(gather), {shape}, {permutation}, {}, false);
        found(a, "propagated gather");
        found(b, "unreduced gather");
        check(a.status == b.status, "small exact differential verification");
        if (shape == List{2, 2, 2}) {
          List inverse(3);
          for (std::size_t i = 0; i < 3; ++i) inverse[permutation[i]] = i;
          check(a.fields.back() == inverse,
                "gather derives inverse permutation from semantics");
          auto domains = semantics::propagateAxes(plan(gather), {shape},
                                                  {permutation}, {}, [] {});
          check(domains.available, "standalone domain relation interpretation");
          auto field = domains.fields.begin()->second;
          check(field.lengths == List{3}, "inferred permutation length");
          for (std::size_t i = 0; i < 3; ++i)
            check(std::binary_search(field.entries[i].begin(),
                                     field.entries[i].end(), inverse[i]),
                  "propagation preserves every valid small witness");
        }
      } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
    auto distribution = analyze(
        "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B "
        "C)) }");
    for (std::size_t rank : {10, 32, 100, 200}) {
      List shape(rank, 2);
      auto s = solve(plan(distribution), {shape, shape, {2, 3}},
                     {{rank - 1}, {0}, {}, {}});
      std::cerr << "distribution rank=" << rank
                << " propagation_ms=" << s.statistics.propagation_ms
                << " forced=" << s.statistics.forced_axis_entries
                << " residual=" << s.statistics.residual_axis_entries
                << " construction_ms=" << s.statistics.constraints_ms
                << " metadata_ms=" << s.statistics.metadata_ms << '\n';
      found(s, "large distribution without ordered inference r" +
                   std::to_string(rank));
      check(s.statistics.propagation_applied,
            "distribution uses semantic propagation");
      check(s.statistics.residual_axis_entries < rank,
            "most axes eliminated before SMT");
    }
    auto associate = analyze(
        "rule r { (dot[@o] (dot[@i] A B) C) => (dot[auto] A (dot[auto] B C)) "
        "}");
    auto lora = analyze(
        "rule r { (dot[@o] X (add W (dot[@i] A B))) => (add (dot[auto] X W) "
        "(dot[auto] (dot[auto] X A) B)) }");
    List large(100, 2);
    auto native_budget = semantics::solveGeneric(
        plan(distribution), {large, large, {2, 3}}, {{99}, {0}, {}, {}});
    found(native_budget,
          "rank-100 distribution within the native default work budget");
    check(native_budget.statistics.propagation_direct,
          "default budget preserves deterministic inference");
    auto association_result = solve(plan(associate), {large, {2, 3}, {3, 4}},
                                    {{99}, {0}, {}, {}, {99}, {0}, {}, {}});
    found(association_result, "rank-100 generic association");
    check(association_result.statistics.propagation_direct,
          "association attributes forced from primitive IR");
    auto lora_result = solve(plan(lora), {large, {2, 3}, {2, 2}, {2, 3}},
                             {{1}, {0}, {}, {}, {99}, {0}, {}, {}});
    found(lora_result, "rank-100 generic LoRA");
    check(lora_result.statistics.propagation_direct,
          "LoRA attributes propagated jointly across terms");
    auto coupled = analyze(
        "rule r { (transpose[@p] X) => (transpose[auto] (transpose[auto] X)) "
        "}");
    auto reduction = analyze("rule r { (fold[@a] X) => (fold[auto] X) }", R"(
      dialect Fold {
        op fold(x:tensor)->tensor {
          attrs { axes:index[]; }
          dtype(t){yield t;}
          shape(s){assert is_valid_axis_list(attrs.axes,len(s)); yield gather(s,exclude(range(len(s)),attrs.axes));}
          semantics(i){
            let free=exclude(range(rank(x)),attrs.axes);
            yield sum(k in index_space(gather(shape(x),attrs.axes))) {
              read(x,scatter(rank(x),concat(free,attrs.axes),concat(i,k)))
            };
          }
        }
      }
      use Fold::{fold};
    )");
    const auto& reduction_plan = plan(reduction);
    auto domain = semantics::propagateAxes(reduction_plan, {{2, 3, 4}},
                                           {{0, 2}}, {}, [] {});
    check(domain.available, "reduction axis-domain preprocessing");
    auto pinned_plan = reduction_plan;
    auto& field = pinned_plan.nodes.back().fields[0];
    const auto reduced_field = domain.fields.at(field.output);
    field.known = -2;
    std::size_t witnesses = 0;
    List order{0, 1, 2};
    std::set<List> candidates;
    do {
      for (std::size_t n = 0; n <= order.size(); ++n)
        candidates.insert(List(order.begin(), order.begin() + n));
    } while (std::next_permutation(order.begin(), order.end()));
    for (const auto& axes : candidates) {
      auto candidate = solve(pinned_plan, {{2, 3, 4}}, {{0, 2}}, {axes}, false);
      if (candidate.status != Status::Found ||
          candidate.statistics.constraint_fallback)
        continue;
      ++witnesses;
      check(std::binary_search(reduced_field.lengths.begin(),
                               reduced_field.lengths.end(), axes.size()),
            "every valid small matcher witness retains its length");
      for (std::size_t i = 0; i < axes.size(); ++i)
        check(std::binary_search(reduced_field.entries[i].begin(),
                                 reduced_field.entries[i].end(), axes[i]),
              "every valid small matcher witness retains its axis assignment");
    }
    check(witnesses == 2,
          "both reduction orderings preserved without symmetry guessing");
    found(solve(reduction_plan, {{2, 3, 4}}, {{0, 2}}),
          "residual reduction permutation synthesized");
    auto shared = plan(coupled);
    shared.nodes.back().fields[0].output =
        shared.nodes[shared.source_end].fields[0].output;
    auto s = solve(shared, {{2, 2, 2, 2}}, {{1, 0, 3, 2}});
    found(s, "residual coupled domains solved with SMT");
    check(s.statistics.propagation_applied &&
              !s.statistics.propagation_direct &&
              s.statistics.symbolic_axis_variables > 0,
          "coupled ambiguity retained");
    auto q = s.fields.back();
    for (std::size_t i = 0; i < q.size(); ++i)
      check(q[q[i]] == List{1, 0, 3, 2}[i], "shared solution reconstruction");
    auto impossible = solve(shared, {{2, 2}}, {{1, 0}});
    check(impossible.status == Status::Invalid,
          "reduced failure uses complete fallback");

    // The matcher cannot prove this identity by reduction bijections. Its
    // contradiction must not restrict the exact fallback's identity
    // permutation.
    auto fallback =
        analyze("rule r { (two X) => (transpose[auto] (double_value X)) }", R"(
      dialect U {
        op two(x:tensor)->tensor {dtype(t){yield t;} shape(s){yield s;}
          semantics(i){yield sum(k in index_space([2])) {read(x,i)};}}
        op double_value(x:tensor)->tensor {dtype(t){yield t;} shape(s){yield s;}
          semantics(i){yield 2*read(x,i);}}
      }
      use U::{two,double_value};
    )");
    found(solve(plan(fallback), {{2, 3}}),
          "sufficient contradiction preserves valid fallback");
    auto repeated =
        analyze("rule r { (dot[@d] A A) => (transpose[auto] (dot[@d] A A)) }");
    auto repeated_result = solve(plan(repeated), {{3}}, std::vector<List>(8));
    found(repeated_result,
          "ambiguous factor correspondence retained for matcher");
    check(!repeated_result.statistics.propagation_applied,
          "no arbitrary repeated-factor choice");

    semantics::SolveOptions bounded;
    bounded.max_symbolic_axis_variables = 2;
    bounded.preferred_candidate = false;
    bounded.axis_propagation = false;
    auto limited = semantics::solveGeneric(
        shared, {{2, 2, 2, 2}}, {{1, 0, 3, 2}}, {}, 1000000, bounded);
    check(limited.status == Status::Unknown,
          "symbolic resource limit is not impossibility");
    std::cout << "Axis domain propagation passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
