#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/pipeline.h"
#include "src/core/syntax/parse.h"
#include "src/executable/program.h"
#include "src/rewrite/builtins/dispatch.h"
#include "src/rewrite/engine.h"
#include "src/rewrite/printer.h"
#include "src/rewrite/value.h"

namespace {
using namespace tepl;
const std::string dialect = R"(
dialect D {
 op copy(x:tensor)->tensor {dtype(t){yield t } shape(s){yield s }}
 op add(x:tensor,y:tensor)->tensor {dtype(a,b){assert a==b yield a } shape(a,b){assert a==b yield a }}
 op scalar(x:tensor)->tensor {dtype(t){yield t } shape(s){yield [] }}
 op transpose(x:tensor)->tensor {
  attrs {permutation:axis[] }
  dtype(t){yield t }
  shape(s){assert is_valid_axis_list(attrs.permutation,len(s)) assert len(attrs.permutation)==len(s) yield gather(s,attrs.permutation) }
 }
}
)";
executable::ExecutableProgram compile(
    const std::string& text, bool prefix = true,
    core::SemanticPolicy policy = core::SemanticPolicy::kStrict) {
  auto parsed = parse((prefix ? dialect : "") + text, "rewrite_test.tepl");
  for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  assert(parsed.ok());
  auto checked = core::inspect(*parsed.program, {policy});
  for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  assert(checked.ok());
  auto result = core::compile(*parsed.program, {policy});
  for (const auto& d : result.diagnostics) std::cerr << d.message << '\n';
  assert(result.ok());
  auto compiled = std::move(*result.program);
  return compiled;
}
void expectInvalid(nlohmann::json j, std::string_view message = {}) {
  bool failed = false;
  try {
    executable::load(j.dump());
  } catch (const std::invalid_argument& error) {
    assert(std::string_view(error.what()).find(message) !=
           std::string_view::npos);
    failed = true;
  }
  assert(failed);
}
void constructors() {
  auto p = compile(R"(
rule compose {(outer: transpose (inner: transpose X)) => (out: transpose X)
 derive {let permutation=gather(inner.attrs.permutation,outer.attrs.permutation) out.attrs={..outer.attrs,permutation:permutation} }}
rule identity {X:[Dims...] (p: transpose X)=>X
 where {let axes=range(len(Dims)) yield p.attrs.permutation==axes}}
graph g {input X:f32[2,3] let a=(transpose[permutation=[1,0]] X)
 let b=(transpose[permutation=[1,0]] a) yield (add b b) }
)");
  auto json = executable::serialize(p);
  auto loaded = executable::load(json);
  assert(executable::serialize(loaded) == json);
  auto before = executable::serialize(p);
  rewrite::Options opts;
  opts.trace = true;
  auto out = rewrite::run(loaded, loaded.ir().graphs[0], opts);
  assert(out.rewrite_count == 2 &&
         out.stop_reason == rewrite::StopReason::FixedPoint &&
         out.trace.size() == 2);
  assert(out.graph.nodes.size() == 2);
  auto& root = out.graph.nodes[out.graph.root];
  assert(root.operands == std::vector<std::size_t>({0, 0}));
  assert(executable::serialize(p) == before);
  auto printed = rewrite::printGraph(p.ir(), out.graph, false);
  auto again = compile(printed);
  assert(again.ir().graphs[0].nodes.size() == 2);
  // Strict schema and semantic validation catch independent error classes.
  auto j = nlohmann::json::parse(json);
  assert(j["version"] == 4);
  j["version"] = 2;
  expectInvalid(j);
  j = nlohmann::json::parse(json);
  j["program"]["types"][0]["kind"] = "index";
  expectInvalid(j);
  j = nlohmann::json::parse(json);
  j["version"] = 99;
  expectInvalid(j);
  j = nlohmann::json::parse(json);
  j["extra"] = true;
  expectInvalid(j);
  j = nlohmann::json::parse(json);
  j["program"]["graphs"][0]["root"] = 999;
  expectInvalid(j);
  j = nlohmann::json::parse(json);
  j["program"]["rules"][0]["rhs"]["value"]["data"]["operands"][0]["value"]
   ["data"]["capture"] = 999;
  expectInvalid(j);
  j = nlohmann::json::parse(json);
  j["program"]["graphs"][0]["nodes"][1]["operands"][0] = 1;
  expectInvalid(j);
  j = nlohmann::json::parse(json);
  j["program"]["operations"][0]["shape"]["result"]["value"]["type"] = 999;
  expectInvalid(j);
  bool duplicateRejected = false;
  try {
    executable::load("{\"version\":1,\"version\":1}");
  } catch (const std::invalid_argument& e) {
    duplicateRejected =
        std::string(e.what()).find("duplicate") != std::string::npos;
  }
  assert(duplicateRejected);
  // Decimal strings preserve extents beyond JSON's interoperable integer range.
  auto wide =
      compile("graph wide {input X:f32[18446744073709551615] yield X }");
  auto wideLoaded = executable::load(executable::serialize(wide));
  assert(wideLoaded.ir().graphs[0].nodes[0].shape->at(0) == UINT64_MAX);
}
void builtinCatalogs() {
  std::set<std::string> names;
  for (const auto& entry : core::builtins::catalog()) names.emplace(entry.name);
  auto spec = nlohmann::json::parse(executable::schema());
  const auto encoded =
      spec["$defs"]["builtin"]["enum"].get<std::vector<std::string>>();
  assert(names == std::set<std::string>(encoded.begin(), encoded.end()));

  auto metadata = compile(R"(
    dialect D { op copy(input:tensor)->tensor {
      shape(s) { assert all([true]) yield s }
    } }
  )",
                          false);
  auto j = nlohmann::json::parse(executable::serialize(metadata));
  j["program"]["operations"][0]["shape"]["statements"][0]["value"]["data"]
   ["condition"]["value"]["data"]["builtin"] = "is_float";
  expectInvalid(j, "unavailable in section");

  auto axes = compile(R"(
    dialect D { op copy(input:tensor)->tensor {
      axes { bind S { input:all_axes(input) } yield S }
      semantics { yield input }
    } }
  )",
                      false);
  j = nlohmann::json::parse(executable::serialize(axes));
  j["program"]["operations"][0]["axis_contract"]["groups"][0]["bindings"][0]
   ["positions"]["name"] = "distinct_axes";
  expectInvalid(j, "unsupported metadata instruction");
}
void feasibilityBeforeBuilder() {
  // Inject q into a structural-runtime fixture with an undefined builder.
  auto p = compile(
      "rule r {(copy X)=>(p:transpose X) derive "
      "{p.attrs={permutation:gather([], [0])} }} graph g {input X:f32[2] "
      "yield(copy X) }");
  auto j = nlohmann::json::parse(executable::serialize(p));
  auto& rule = j["program"]["rules"][0];
  rule["feasibility"] =
      nlohmann::json::array({{{"kind", "bool"},
                              {"sort", "bool"},
                              {"name", "false"},
                              {"owner", 0},
                              {"index", 0},
                              {"args", nlohmann::json::array()}}});
  auto blocked = executable::load(j.dump());
  assert(rewrite::run(blocked, blocked.ir().graphs[0]).rewrite_count == 0);
  rule["feasibility"][0]["name"] = "true";
  auto exposed = executable::load(j.dump());
  bool failed = false;
  try {
    rewrite::run(exposed, exposed.ir().graphs[0]);
  } catch (const rewrite::EvaluationError&) {
    failed = true;
  }
  assert(failed);
}
void guardsAndFailure() {
  const std::string graph = "graph g {input X:f32[2] yield (copy X) }";
  auto p = compile(
      "rule r {(copy X)=>X where true || "
      "len(range(18446744073709551615))==0 }" +
      graph);
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 1);
  p = compile(
      "rule r {(copy X)=>X where false && "
      "len(range(18446744073709551615))==0 }" +
      graph);
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 0);
  p = compile("rule r {(copy X)=>X where 1 / 0 == 0 }" + graph);
  {
    bool failed = false;
    try {
      rewrite::run(p, p.ir().graphs[0]);
    } catch (const rewrite::EvaluationError&) {
      failed = true;
    }
    assert(
        failed);  // Evaluation failures remain distinct from non-application.
  }
  p = compile(
      "dialect Bad {op bad_scalar(x:tensor)->tensor {"
      "dtype(t){yield t} shape(s){assert len(s)==0 yield []}}} "
      "rule r {(copy X)=>(Bad.bad_scalar X)}" +
      graph);
  {
    bool failed = false;
    try {
      rewrite::run(p, p.ir().graphs[0]);
    } catch (const rewrite::EvaluationError&) {
      failed = true;
    }
    assert(
        failed);  // Evaluation failures remain distinct from non-application.
  }
  p = compile(
      "rule r {(copy X)=>(p: transpose X) derive "
      "{p.attrs={permutation:[9]} }}" +
      graph);
  {
    bool failed = false;
    try {
      rewrite::run(p, p.ir().graphs[0]);
    } catch (const rewrite::EvaluationError&) {
      failed = true;
    }
    assert(
        failed);  // Evaluation failures remain distinct from non-application.
  }
  p = compile("rule r {(copy X)=>(copy X)}" + graph);
  assert(rewrite::run(p, p.ir().graphs[0]).stop_reason ==
         rewrite::StopReason::FixedPoint);
  auto limited = compile(
      "rule r {(copy X)=>X where len(range(18446744073709551615))==0 }" +
      graph);
  auto out = rewrite::run(limited, limited.ir().graphs[0]);
  assert(out.rewrite_count == 0 &&
         out.stop_reason == rewrite::StopReason::ResourceLimit);
}
void sharingAndLimits() {
  auto p = compile(
      "rule r {(copy X)=>X} graph g {input X:f32[2] let "
      "a=(transpose[permutation=[0]] X) let b=(copy a) yield (add a b) }");
  auto out = rewrite::run(p, p.ir().graphs[0]);
  assert(out.rewrite_count == 1);
  const auto& root = out.graph.nodes[out.graph.root];
  assert(root.operands[0] == root.operands[1]);
  p = compile(
      "rule r {(add X X)=>X} graph g {input X:f32[2] let a=(copy X) let "
      "b=(copy X) yield (add a b) }");
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 0);
  p = compile(
      "rule r {(add (let A = (copy X)) (copy X)) => (add A A)} "
      "graph g {input X:f32[2] yield (add (copy X) (copy X)) }");
  out = rewrite::run(p, p.ir().graphs[0]);
  assert(out.rewrite_count == 1);
  assert(out.graph.nodes[out.graph.root].operands[0] ==
         out.graph.nodes[out.graph.root].operands[1]);
  p = compile(
      "rule r {(add X Y)=>(add Y X)} graph g {input X:f32[2] input "
      "Y:f32[2] yield (add X Y) }");
  rewrite::Options opt;
  opt.max_steps = 3;
  out = rewrite::run(p, p.ir().graphs[0], opt);
  assert(out.rewrite_count == 3 &&
         out.stop_reason == rewrite::StopReason::StepLimit);
  opt.cancelled = []() { return true; };
  out = rewrite::run(p, p.ir().graphs[0], opt);
  assert(out.rewrite_count == 0 &&
         out.stop_reason == rewrite::StopReason::Cancelled);
}
void metadataAndDepth() {
  auto p = compile("rule r {(copy X)=>X} graph g {input X yield (copy X) }");
  auto input = p.ir().graphs[0];
  input.nodes[input.root].shape = std::vector<std::uint64_t>{99};
  input.nodes[input.root].dtype = core::DType::kF32;
  auto out = rewrite::run(p, input);
  assert(out.rewrite_count == 1 && !out.graph.nodes[out.graph.root].shape);
  // Flat DAGs need not fit the C++ call stack during traversal or replacement.
  auto node = input.nodes[input.root];
  for (std::size_t i = input.nodes.size(); i < 12000; ++i) {
    node.operands = {i - 1};
    input.nodes.push_back(node);
  }
  input.root = input.nodes.size() - 1;
  rewrite::Options options;
  options.max_steps = 1;
  out = rewrite::run(p, input, options);
  assert(out.rewrite_count == 1 &&
         out.stop_reason == rewrite::StopReason::StepLimit);
  assert(!rewrite::printGraph(p.ir(), out.graph, false).empty());
  rewrite::Budget budget;
  using rewrite::Integer;
  auto zero = rewrite::evaluateBuiltin(
      core::builtins::Builtin::kProduct,
      {rewrite::Value(
          rewrite::List{Integer(UINT64_MAX), Integer(UINT64_MAX), Integer(0)})},
      budget);
  assert(rewrite::integer(zero) == 0);
  bool rejected = false;
  try {
    rewrite::evaluateBuiltin(core::builtins::Builtin::kCeilDiv,
                             {Integer(4), Integer(-2)}, budget);
  } catch (const rewrite::EvaluationError&) {
    rejected = true;
  }
  assert(rejected);
}
void metadataLiteralPrograms() {
  // These operation contracts exercise the runtime metadata interpreter, both
  // on imported graph nodes and on nodes constructed by a replacement rule.
  for (const auto& expression :
       {std::string("[]"), std::string("[7,s[0]+1]"),
        std::string("[if i==0 then 5 else s[i]+1 for i in range(len(s))]")}) {
    auto p = compile(
        "dialect M {op custom(a:tensor)->tensor {dtype(t){yield t} "
        "shape(s){yield " +
        expression +
        "}}}\nrule r {(copy (M.custom X)) => (M.custom X)}\n"
        "graph g {input X:f32[2,3] yield (copy (M.custom X))}");
    auto loaded = executable::load(executable::serialize(p));
    auto result = rewrite::run(loaded, loaded.ir().graphs[0]);
    assert(result.rewrite_count == 1);
    const auto& root = result.graph.nodes[result.graph.root];
    auto expected = expression == "[]"           ? std::vector<uint64_t>{}
                    : expression == "[7,s[0]+1]" ? std::vector<uint64_t>{7, 3}
                                                 : std::vector<uint64_t>{5, 4};
    assert(root.shape && *root.shape == expected &&
           root.dtype == core::DType::kF32);
  }
}
void automatic(const std::string& path) {
  std::ifstream f(path);
  std::string definitions{std::istreambuf_iterator<char>(f), {}};
  auto implicit = compile(definitions + R"(
rule compose {(transpose (transpose X)) => (transpose X)}
graph g {input X:f32[2,3,4] let a=(transpose[permutation=[1,2,0]] X) yield (transpose[permutation=[2,0,1]] a) }
)",
                          false);
  assert(implicit.ir().rules[0].descriptors.size() == 3);
  auto implicit_loaded = executable::load(executable::serialize(implicit));
  auto implicit_out =
      rewrite::run(implicit_loaded, implicit_loaded.ir().graphs[0]);
  assert(implicit_out.rewrite_count == 1);
  assert(implicit_out.graph.nodes[implicit_out.graph.root].shape ==
         std::optional<std::vector<std::uint64_t>>({2, 3, 4}));
  const auto& implicit_attrs =
      implicit_out.graph.nodes[implicit_out.graph.root].attributes[0].elements;
  assert(implicit_attrs.size() == 3 && implicit_attrs[0].text == "0" &&
         implicit_attrs[1].text == "1" && implicit_attrs[2].text == "2");

  auto p = compile(definitions + R"(
rule compose {(a: transpose (b: transpose X)) => (transpose X)}
graph g {input X:f32[2,3,4] let a=(transpose[permutation=[1,2,0]] X) yield (transpose[permutation=[2,0,1]] a) }
)",
                   false);
  auto loaded = executable::load(executable::serialize(p));
  auto out = rewrite::run(loaded, loaded.ir().graphs[0]);
  assert(out.rewrite_count == 1);
  assert(out.graph.nodes[out.graph.root].attributes[0].elements[0].text == "0");
  assert(out.graph.nodes[out.graph.root].shape ==
         std::optional<std::vector<std::uint64_t>>({2, 3, 4}));
  p = compile(definitions + R"(
rule move_reduce {(r: reduce_sum (p: transpose A)) => (transpose (reduce_sum A))}
graph g {input A:f32[2,3,4] yield (reduce_sum[dimensions=[1]] (transpose[permutation=[2,0,1]] A)) }
)",
              false, core::SemanticPolicy::kAlgebraic);
  loaded = executable::load(executable::serialize(p));
  out = rewrite::run(loaded, loaded.ir().graphs[0]);
  assert(out.rewrite_count == 1);
  const auto& transposed = out.graph.nodes[out.graph.root];
  assert(transposed.shape == std::optional<std::vector<std::uint64_t>>({4, 3}));
  assert(transposed.attributes[0].elements[0].text == "1");
  assert(transposed.attributes[0].elements[1].text == "0");
  assert(
      out.graph.nodes[transposed.operands[0]].attributes[0].elements[0].text ==
      "0");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    constructors();
    builtinCatalogs();
    feasibilityBeforeBuilder();
    guardsAndFailure();
    sharingAndLimits();
    metadataAndDepth();
    metadataLiteralPrograms();
    std::unique_ptr<rules_cc::cc::runfiles::Runfiles> files(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0]));
    assert(argc == 2);
    automatic(files->Rlocation(argv[1]));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
