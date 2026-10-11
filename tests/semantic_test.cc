#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/codegen/generate.h"
#include "src/core/analyze.h"
#include "src/core/print.h"
#include "src/core/semantics/solver.h"
#include "src/parse.h"
namespace {
using tepl::core::semantics::List;
using tepl::core::semantics::Status;
void check(bool b, const std::string& message) {
  if (!b) throw std::runtime_error(message);
}
std::string dialect;
auto analyze(const std::string& rule, bool algebraic = true,
             const std::string& definitions = "") {
  auto parsed = tepl::parse(
      (definitions.empty() ? dialect : definitions) + rule, "semantic.tepl");
  if (!parsed.ok())
    for (auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  check(parsed.ok(), "semantic fixture must parse");
  return tepl::core::analyze(*parsed.program,
                             {algebraic ? tepl::core::SemanticPolicy::kAlgebraic
                                        : tepl::core::SemanticPolicy::kStrict,
                              10000});
}
void success(const tepl::core::AnalysisResult& r) {
  for (auto& d : r.diagnostics) std::cerr << d.message << '\n';
  check(r.ok(), "expected lowered rule");
  check(r.program->rules.at(0).inference->semantic.has_value(),
        "one generic constraint plan");
  check(r.program->host_functions.empty(), "no dialect host inference");
}
void error(const tepl::core::AnalysisResult& r, const std::string& text) {
  check(!r.ok(), "expected diagnostic " + text);
  for (auto& d : r.diagnostics)
    if (d.message.find(text) != std::string::npos) return;
  for (auto& d : r.diagnostics) std::cerr << d.message << '\n';
  throw std::runtime_error("missing diagnostic " + text);
}
auto solve(const tepl::core::AnalysisResult& r, const std::vector<List>& shapes,
           const std::vector<List>& known, const std::vector<List>& pinned = {},
           std::size_t budget = 1000000) {
  success(r);
  auto result =
      tepl::core::semantics::solve(*r.program->rules.at(0).inference->semantic,
                                   shapes, known, pinned, budget);
  if (result.status == Status::Unknown)
    std::cerr << "unknown: " << result.reason << '\n';
  return result;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    check(argc == 2, "dialect fixture argument");
    std::string err;
    auto files = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &err));
    check(bool(files), err);
    std::ifstream input(files->Rlocation(argv[1]));
    dialect = {std::istreambuf_iterator<char>(input), {}};
    check(!dialect.empty(), "read dialect");
    const std::vector<List> matrix{{1}, {0}, {}, {}};
    auto identity = analyze(
        "rule r { (dot[@d] A B) => (dot[auto] (transpose[auto] A) B) }");
    check(solve(identity, {{2, 3}, {3, 4}}, matrix).status == Status::Found,
          "infer transpose and contraction jointly");
    check(solve(identity, {{2, 3}, {3, 4}}, matrix, {}, 1).status ==
              Status::Unknown,
          "budget exhaustion unknown");
    for (std::size_t rank : {3, 5, 12}) {
      List shape(rank, 2);
      check(
          solve(identity, {shape, {2, 3}}, {{rank - 1}, {0}, {}, {}}).status ==
              Status::Found,
          "rank-generic identity");
    }
    auto cycle = analyze(
        "rule r { (dot[@d] A B) => (dot[auto] (transpose[auto, "
        "permutation=[1,2,0]] A) B) }");
    auto result =
        solve(cycle, {{3, 2, 4}, {3, 5}}, {{0}, {0}, {}, {}}, {{1, 2, 0}});
    check(result.status == Status::Found,
          "infer inverse axis remapping from reads");
    check(solve(cycle, {{2, 4, 3}, {3, 5}}, {{2}, {0}, {}, {}}, {{1, 2, 0}})
                  .status == Status::Invalid,
          "all legal attributes rejected for output order");
    auto distribute = analyze(
        "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B "
        "C)) }");
    check(solve(distribute, {{2, 3}, {2, 3}, {3, 4}}, matrix).status ==
              Status::Found,
          "distribution from polynomial semantics");
    auto associate = analyze(
        "rule r { (dot[@d] (dot[@inner] A B) C) => (dot[auto] A (dot[auto] B "
        "C)) }");
    auto double_matrix = matrix;
    double_matrix.insert(double_matrix.end(), matrix.begin(), matrix.end());
    check(solve(associate, {{2, 3}, {3, 4}, {4, 5}}, double_matrix).status ==
              Status::Found,
          "association from semantics");
    auto batched = std::vector<List>{{2}, {1}, {0}, {0}, {2}, {1}, {0}, {0}};
    check(solve(associate, {{2, 3, 4}, {2, 4, 5}, {2, 5, 6}}, batched).status ==
              Status::Found,
          "batched association");
    List large_a(10, 2), large_b(10, 2), large_c(10, 2), batch_axes(10);
    std::iota(batch_axes.begin(), batch_axes.end(), 0);
    large_a.insert(large_a.end(), {3, 4});
    large_b.insert(large_b.end(), {4, 5});
    large_c.insert(large_c.end(), {5, 6});
    auto large_plan = *associate.program->rules[0].inference->semantic;
    large_plan.timeout_ms = 5000;
    auto large =
        tepl::core::semantics::solve(large_plan, {large_a, large_b, large_c},
                                     {{11},
                                      {10},
                                      batch_axes,
                                      batch_axes,
                                      {11},
                                      {10},
                                      batch_axes,
                                      batch_axes},
                                     {}, 100000);
    if (large.status == Status::Unknown) std::cerr << large.reason << '\n';
    check(large.status == Status::Found,
          "large batched inference with bounded solver work");
    auto strict = analyze(
        "rule r { (dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B "
        "C)) }",
        false);
    check(
        solve(strict, {{2, 3}, {2, 3}, {3, 4}}, matrix).status != Status::Found,
        "strict does not silently reassociate");
    error(analyze("rule r { (add X Y) => X }"), "scalar counterexample");
    error(analyze("rule r { (dot[auto] A B) => A }"),
          "only allowed on the RHS");
    error(analyze("rule r { (dot[@d] A B) => (dot[auto,bad=[]] A B) }"),
          "unknown auto override field");
    auto nosem =
        "dialect D { op f(x:tensor)->tensor { attrs { p:index[]; } "
        "shape(s){yield s;} } }";
    check(analyze("rule r { (f[@p] A) => (f[@p] A) }", true, nosem).ok(),
          "semantics optional");
    error(analyze("rule r { (f[@p] A) => (f[auto] A) }", true, nosem),
          "missing semantics");
    auto renamed = dialect;
    for (auto [a, b] : std::vector<std::pair<std::string, std::string>>{
             {"dot_general", "contract"},
             {"alias: dot;", "alias: arbitrary;"},
             {"transpose", "shuffle"},
             {"permutation", "order"},
             {"lhs_contracting_dimensions", "left_axes"}})
      for (std::size_t pos = 0;
           (pos = renamed.find(a, pos)) != std::string::npos; pos += b.size())
        renamed.replace(pos, a.size(), b);
    auto generic = analyze(
        "rule r { (arbitrary[@d] A B) => (arbitrary[auto] (shuffle[auto] A) B) "
        "}",
        true, renamed);
    check(solve(generic, {{2, 3}, {3, 4}}, matrix).status == Status::Found,
          "names are irrelevant");
    // A custom unary reduction and negate exercise expressions absent from the
    // old operation templates, including ordered filtered selector domains.
    auto reduce = R"(dialect D {
      op fold(x:tensor)->tensor {
        attrs { axes:index[]; }
        dtype(t){yield t;}
        shape(s){ assert is_valid_axis_list(attrs.axes,len(s));
          yield gather(s,exclude(range(len(s)),attrs.axes)); }
        semantics(o){ let free=exclude(range(rank(x)),attrs.axes);
          yield sum(k in index_space(gather(shape(x),attrs.axes))) {
            -read(x,scatter(rank(x),concat(free,attrs.axes),concat(o,k)))
          }; }
      }
    })";
    auto reduction =
        analyze("rule r { (fold[@s] X) => (fold[auto] X) }", true, reduce);
    check(solve(reduction, {{2, 3, 4}}, {{1}}).status == Status::Found,
          "custom reduction and negation");
    check(solve(reduction, {{2, 0, 4}}, {{1}}).status == Status::Found,
          "empty reduction domain");
    auto invalid =
        analyze("rule r { (fold[@s] X) => (fold[auto] X) }", true, reduce);
    check(solve(invalid, {{2, 3, 4}}, {{1}}, {}, 1).status == Status::Unknown,
          "limits never prove invalid");
    auto filtered = R"(dialect D {
      op select(x:tensor)->tensor {
        attrs { order:index[]; selected:index[]; }
        dtype(t){yield t;}
        shape(s){
          assert is_valid_axis_list(attrs.selected,len(s));
          assert is_valid_axis_list(attrs.order,len(attrs.selected));
          assert len(attrs.order)==len(attrs.selected);
          yield gather(s,gather(attrs.selected,attrs.order));
        }
        semantics(o){
          let free=gather(attrs.selected,attrs.order);
          let reduced=exclude(range(rank(x)),attrs.selected);
          yield sum(k in index_space(gather(shape(x),reduced))) {
            read(x,scatter(rank(x),concat(free,reduced),concat(o,k)))
          };
        }
      }
    })";
    auto selection = analyze("rule r { (select[@s] X) => (select[auto] X) }",
                             true, filtered);
    check(
        solve(selection, {{2, 3, 4}}, {{1, 0}, {0, 2}}).status == Status::Found,
        "filtered selector bounds lower in dependency order");
    auto coefficient = R"(dialect D {
      op twice(x:tensor)->tensor {
        dtype(t){yield t;} shape(s){yield s;}
        semantics(i){yield 2 * read(x,i);}
      }
      op duplicate(x:tensor)->tensor {
        dtype(t){yield t;} shape(s){yield s;}
        semantics(i){yield read(x,i) + read(x,i);}
      }
    })";
    auto coefficients =
        analyze("rule r { (twice X) => (duplicate X) }", true, coefficient);
    check(solve(coefficients, {{2, 3}}, {}).status == Status::Found,
          "exact coefficient arithmetic");
    auto constants = R"(dialect D {
      op fill(x:tensor)->tensor {
        attrs { axes:index[]; } dtype(t){yield t;}
        shape(s){assert is_valid_axis_list(attrs.axes,len(s));yield s;}
        semantics(i){yield -2;}
      }
    })";
    auto fill =
        analyze("rule r { (fill[@d] X) => (fill[auto] X) }", true, constants);
    check(solve(fill, {{2, 3}}, {{1}}).status == Status::Found,
          "constant scalar elements");
    auto rotation = R"(dialect D {
      op rotate(x:tensor)->tensor {
        dtype(t){yield t;}
        shape(s){assert len(s)==1; assert s[0]>0; yield s;}
        semantics(i){yield read(x,[(i[0]+1)%shape(x)[0]]);}
      }
    })";
    auto rotations =
        analyze("rule r { (rotate (rotate X)) => X }", true, rotation);
    check(solve(rotations, {{2}}, {}).status == Status::Found,
          "finite fallback proves index arithmetic");
    check(solve(rotations, {{3}}, {}).status == Status::Invalid,
          "failed index normalization alone does not determine validity");
    auto bad_index = R"(dialect D {
      op identity(x:tensor)->tensor {
        dtype(t){yield t;} shape(s){yield s;}
        semantics(i){yield read(x,i);}
      }
      op invalid_read(x:tensor)->tensor {
        attrs { axes:index[]; } dtype(t){yield t;}
        shape(s){assert is_valid_axis_list(attrs.axes,len(s));yield s;}
        semantics(i){yield read(x,[1]);}
      }
    })";
    auto bounds = analyze("rule r { (identity X) => (invalid_read[auto] X) }",
                          true, bad_index);
    check(solve(bounds, {{1}}, {}).status != Status::Found,
          "singleton normalization never hides an out-of-bounds read");
    auto short_circuit = R"(dialect D {
      op identity(x:tensor)->tensor {
        dtype(t){yield t;} shape(s){yield s;}
        semantics(i){yield read(x,i);}
      }
      op checked(x:tensor)->tensor {
        attrs { axes:index[]; } dtype(t){yield t;}
        shape(s){
          assert is_valid_axis_list(attrs.axes,len(s));
          assert len(attrs.axes)==0 || attrs.axes[0]<len(s);
          yield s;
        }
        semantics(i){yield read(x,i);}
      }
    })";
    auto lazy = analyze("rule r { (identity X) => (checked[auto] X) }", true,
                        short_circuit);
    check(solve(lazy, {{}}, {}).status == Status::Found,
          "short-circuit assertions preserve the valid empty attribute list");
    for (auto target :
         {tepl::codegen::Target::kRust, tepl::codegen::Target::kCpp}) {
      tepl::codegen::Options opts;
      opts.target = target;
      auto generated = tepl::codegen::generate(*identity.program, opts);
      check(generated.ok(), "generate shared solver adapter");
      std::string output;
      for (auto& f : generated.files) output += f.contents;
      check(output.find("semantic::construct") != std::string::npos,
            "generic solver emitted");
      check(output.find("axis_construct") == std::string::npos,
            "axis machine retired");
    }
    auto& plan = *identity.program->rules[0].inference->semantic;
    auto wire = tepl::core::semantics::encodePlan(plan);
    check(!wire.empty(), "serialize lowered plan");
    check(tepl::core::semantics::solveRequest("garbage") == "2 0 ",
          "bad wire unknown");
    std::cout << "Generic element semantics tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
