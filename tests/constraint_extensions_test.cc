#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/analyze.h"
#include "src/core/semantics/solver.h"
#include "src/parse.h"

namespace {
using namespace tepl::core;
using semantics::List;
using semantics::Status;
void check(bool b, const std::string& message) {
  if (!b) throw std::runtime_error(message);
}
const std::string custom = R"(
dialect Extra {
  attrs Axes { axes:index[]; }
  op unused(x:tensor)->tensor {
    attrs:Axes; dtype(t){yield t;}
    shape(s){assert is_valid_axis_list(attrs.axes,len(s));yield s;}
    semantics(i){yield sum(k in index_space(gather(shape(x),attrs.axes))) {read(x,i)};}
  }
  op fixed_zero(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){yield sum(k in index_space([0])) {read(x,i)};}
  }
  op fixed_one(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){yield sum(k in index_space([1])) {read(x,i)};}
  }
  op fixed_two(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){yield sum(k in index_space([2])) {read(x,i)};}
  }
  op twice(x:tensor)->tensor {
    attrs:Axes; dtype(t){yield t;}
    shape(s){assert is_valid_axis_list(attrs.axes,len(s));yield gather(s,exclude(range(len(s)),attrs.axes));}
    semantics(i){
      let f=exclude(range(rank(x)),attrs.axes);
      yield sum(k in index_space(gather(shape(x),attrs.axes))) {
        read(x,scatter(rank(x),concat(f,attrs.axes),concat(i,k)))
      } + sum(j in index_space(gather(shape(x),attrs.axes))) {
        read(x,scatter(rank(x),concat(f,attrs.axes),concat(i,j)))
      };
    }
  }
  op twice_one(x:tensor)->tensor {
    attrs:Axes; dtype(t){yield t;}
    shape(s){assert is_valid_axis_list(attrs.axes,len(s));yield gather(s,exclude(range(len(s)),attrs.axes));}
    semantics(i){
      let f=exclude(range(rank(x)),attrs.axes);
      yield 2 * sum(q in index_space(gather(shape(x),attrs.axes))) {
        read(x,scatter(rank(x),concat(f,attrs.axes),concat(i,q)))
      };
    }
  }
  op cancelled(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){yield sum(k in index_space([2])) {read(x,i)} -
                       sum(j in index_space([2])) {read(x,i)} + read(x,i);}
  }
  op fresh(x:tensor)->tensor {
    attrs:Axes; dtype(t){yield t;}
    shape(s){assert is_valid_axis_list(attrs.axes,len(s));yield gather(s,exclude(range(len(s)),attrs.axes));}
    semantics(i){
      let f=exclude(range(rank(x)),attrs.axes);
      let s=sum(k in index_space(gather(shape(x),attrs.axes))) {
        read(x,scatter(rank(x),concat(f,attrs.axes),concat(i,k)))
      };
      yield s*s;
    }
  }
  op double_value(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){yield 2*read(x,i);}
  }
  op square_value(x:tensor)->tensor {
    dtype(t){yield t;} shape(s){yield s;}
    semantics(i){yield read(x,i)*read(x,i);}
  }
  op diagonal_square(x:tensor)->tensor {
    attrs:Axes; dtype(t){yield t;}
    shape(s){assert is_valid_axis_list(attrs.axes,len(s));yield gather(s,exclude(range(len(s)),attrs.axes));}
    semantics(i){
      let f=exclude(range(rank(x)),attrs.axes);
      yield sum(k in index_space(gather(shape(x),attrs.axes))) {
        let v=read(x,scatter(rank(x),concat(f,attrs.axes),concat(i,k)));
        v*v
      };
    }
  }
}
use Extra::{unused,fixed_zero,fixed_one,fixed_two,twice,twice_one,cancelled,fresh,double_value,square_value,diagonal_square};
)";
std::string dialect;
AnalysisResult analyze(std::string rule) {
  auto parsed = tepl::parse(dialect + custom + rule, "extensions.tepl");
  check(parsed.ok(), "parse extension fixture");
  auto result =
      tepl::core::analyze(*parsed.program, {SemanticPolicy::kAlgebraic, 10000});
  for (auto& d : result.diagnostics) std::cerr << d.message << '\n';
  check(result.ok(), "analyze extension fixture");
  return result;
}
const inference::SemanticPlan& plan(const AnalysisResult& a) {
  return *a.program->rules.at(0).inference->semantic;
}
auto solve(const inference::SemanticPlan& p, const std::vector<List>& shapes,
           const std::vector<List>& known = {},
           const std::vector<List>& pins = {}, bool compiled = true) {
  semantics::SolveOptions options;
  options.semantic_constraints = compiled;
  options.preferred_candidate = false;
  return semantics::solveGeneric(p, shapes, known, pins, 1000000, options);
}
void compiled(const semantics::Solution& s, const std::string& message) {
  if (s.status != Status::Found) std::cerr << s.reason << '\n';
  check(s.status == Status::Found && s.method == "constraints-z3" &&
            s.statistics.constraint_attempted &&
            !s.statistics.constraint_fallback,
        message);
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
    check(!dialect.empty(), "load dialect");
    auto symmetry = analyze(
        "rule r { (dot[@d] A A) => (transpose[auto,permutation=[1,0]] (dot[@d] "
        "A A)) }");
    check(plan(symmetry).constraints.available, "repeated captures admitted");
    compiled(solve(plan(symmetry), {{3}}, std::vector<List>(8), {{1, 0}}),
             "factor occurrences can swap");
    auto asymmetric = solve(plan(symmetry), {{2, 2}},
                            {{1}, {0}, {}, {}, {1}, {0}, {}, {}}, {{1, 0}});
    check(asymmetric.status == Status::Invalid &&
              asymmetric.statistics.constraint_fallback,
          "repeated captures do not imply symmetry");
    auto multiplicity_read =
        analyze("rule r { (square_value X) => (transpose[auto] X) }");
    check(solve(plan(multiplicity_read), {{2, 3}}).status == Status::Invalid,
          "occurrence matching preserves repeated factor multiplicity");

    auto one = analyze("rule r { (fixed_one X) => (transpose[auto] X) }");
    check(plan(one).constraints.lhs[0].reductions.empty(),
          "constant singleton removed at compile time");
    compiled(solve(plan(one), {{2, 3}}),
             "fixed singleton uses compiled constraints");
    auto zero = analyze("rule r { (fixed_zero X) => (transpose[auto] X) }");
    check(!plan(zero).constraints.lhs[0].reductions.empty(),
          "empty domain not eliminated");
    check(solve(plan(zero), {{2, 3}}).status == Status::Invalid,
          "zero reduction differs from identity");
    auto two = analyze("rule r { (fixed_two X) => (transpose[auto] X) }");
    check(solve(plan(two), {{2, 3}}).status == Status::Invalid,
          "unused larger reduction retains multiplicity");
    auto multiplicity = analyze(
        "rule r { (fixed_two X) => (transpose[auto] (double_value X)) }");
    auto fallback = solve(plan(multiplicity), {{2, 3}});
    check(fallback.status == Status::Found &&
              fallback.statistics.constraint_fallback &&
              fallback.method == "z3",
          "insufficient reduction bijection still permits exact fallback");
    auto dynamic = analyze("rule r { (unused[@u] X) => (transpose[auto] X) }");
    for (List shape : {List{1, 3}, List{1, 1, 3}, List{3}}) {
      List axes(shape.size() - 1);
      std::iota(axes.begin(), axes.end(), 0);
      compiled(solve(plan(dynamic), {shape}, {axes}),
               "matched singleton domain eliminated");
    }
    auto dynamic_rhs =
        analyze("rule r { X => (unused[auto,axes=[0]] (transpose[auto] X)) }");
    compiled(solve(plan(dynamic_rhs), {{1, 3}}, {}, {{0}}),
             "symbolic RHS singleton domain guarded");
    check(
        solve(plan(dynamic_rhs), {{2, 3}}, {}, {{0}}).status == Status::Invalid,
        "symbolic domain cannot be dropped without singleton guard");
    compiled(solve(plan(dynamic), {{2, 3}}, {{}}),
             "zero-dimensional index space has one point");

    auto alpha = analyze(
        "rule r { (twice[@a] X) => (transpose[auto] (twice_one[@a] X)) }");
    check(plan(alpha).constraints.lhs.size() == 1 &&
              plan(alpha).constraints.lhs[0].coefficient == 2,
          "alpha-equivalent reductions combine coefficients");
    compiled(solve(plan(alpha), {{2, 3, 4}}, {{2}, {2}}),
             "combined terms synthesize attributes");
    auto cancel = analyze("rule r { (cancelled X) => (transpose[auto] X) }");
    check(plan(cancel).constraints.lhs.size() == 1 &&
              plan(cancel).constraints.lhs[0].reductions.empty(),
          "alpha-equivalent cancellation removes zero terms");
    compiled(solve(plan(cancel), {{2, 3}}),
             "cancelled reductions infer identity");
    auto fresh =
        analyze("rule r { (fresh[@a] X) => (transpose[auto] (fresh[@a] X)) }");
    auto& term = plan(fresh).constraints.lhs[0];
    check(
        term.reductions.size() == 2 && term.reductions[0] != term.reductions[1],
        "each use of a reduction gets independent binders");
    compiled(solve(plan(fresh), {{2, 3}}, {{1}, {1}}),
             "repeated reduction products retain independence");
    auto diagonal = analyze(
        "rule r { (fresh[@a] X) => (transpose[auto] (diagonal_square[@a] X)) "
        "}");
    check(solve(plan(diagonal), {{2}}, {{0}, {0}}).status == Status::Invalid,
          "independent reductions cannot become one shared reduction");
    check(solve(plan(diagonal), {{1}}, {{0}, {0}}).status == Status::Found,
          "singleton used coordinates retain exact verifier fallback");

    // Differential checks use the old metadata-only path, without preferences.
    for (std::uint64_t m : {0, 1, 2})
      for (std::uint64_t k : {0, 1, 2}) {
        auto a = solve(plan(fresh), {{m, k}}, {{1}, {1}});
        auto b = solve(plan(fresh), {{m, k}}, {{1}, {1}}, {}, false);
        check(a.status == b.status && a.status == Status::Found,
              "small repeated-input differential cases");
      }
    auto measured = solve(plan(alpha), {{2, 3}}, {{1}, {1}});
    check(
        measured.statistics.solver_calls >= 2 &&
            measured.statistics.candidates >= 1 &&
            measured.statistics.total_ms >= measured.statistics.verification_ms,
        "solver metrics cover compilation-guided inference");
    auto invalid = analyze("rule r { (dot[@d] A B) => (transpose[auto] A) }");
    semantics::SolveOptions limited;
    limited.semantic_constraints = false;
    limited.preferred_candidate = false;
    limited.max_candidates = 1;
    auto bounded =
        semantics::solveGeneric(plan(invalid), {{2, 2}, {2, 2}},
                                {{1}, {0}, {}, {}}, {}, 1000000, limited);
    check(bounded.status == Status::Unknown &&
              bounded.reason == "candidate limit" &&
              bounded.statistics.candidates == 1 &&
              bounded.statistics.rejected_candidates == 1,
          "candidate limit is measured and never proves invalid");
    std::cout << "Constraint extensions passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
