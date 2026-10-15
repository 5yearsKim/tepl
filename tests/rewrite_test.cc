#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/codegen/generate.h"
#include "src/core/analyze.h"
#include "src/parse.h"
#include "src/program/program.h"
#include "src/rewrite/engine.h"
#include "src/rewrite/printer.h"
#include "src/rewrite/value.h"

namespace {
using namespace tepl;
const std::string dialect = R"(
dialect D {
 op copy(x:tensor)->tensor {dtype(t){yield t;} shape(s){yield s;}}
 op add(x:tensor,y:tensor)->tensor {dtype(a,b){assert a==b;yield a;} shape(a,b){assert a==b;yield a;}}
 op scalar(x:tensor)->tensor {dtype(t){yield t;} shape(s){yield [];}}
 op transpose(x:tensor)->tensor {
  attrs {permutation:index[];}
  dtype(t){yield t;}
  shape(s){assert is_valid_axis_list(attrs.permutation,len(s));assert len(attrs.permutation)==len(s);yield gather(s,attrs.permutation);}
 }
}
)";
program::Program compile(
    const std::string& text, bool prefix = true,
    core::SemanticPolicy policy = core::SemanticPolicy::kStrict) {
  auto parsed = parse((prefix ? dialect : "") + text, "rewrite_test.tepl");
  for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  assert(parsed.ok());
  auto checked = core::analyze(*parsed.program, {policy});
  for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  assert(checked.ok());
  auto compiled = program::compile(*checked.program);
  for (std::size_t i = 0; i < checked.program->rules.size(); ++i) {
    const auto& r = checked.program->rules[i];
    assert(compiled.ir().rules[i].conditions.size() ==
           r.conditions.size() +
               (r.semantic_plan ? r.semantic_plan->proof.guard.size() : 0));
  }
  return compiled;
}
void expectInvalid(nlohmann::json j) {
  bool failed = false;
  try {
    program::load(j.dump());
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  assert(failed);
}
void constructors() {
  auto p = compile(R"(
rule compose {(transpose[@outer] (transpose[@inner] X)) => (transpose[@out] X)
 derive {let permutation=gather(@inner.permutation,@outer.permutation); @out={..@outer,permutation:permutation};}}
rule identity {X:[Dims...] (transpose[@p] X)=>X
 where {let axes=range(len(Dims)); @p.permutation==axes}}
graph g {input X:f32[2,3];let a=(transpose[permutation=[1,0]] X);
 let b=(transpose[permutation=[1,0]] a);yield (add b b);}
)");
  auto json = program::serialize(p);
  auto loaded = program::load(json);
  assert(program::serialize(loaded) == json);
  auto before = program::serialize(p);
  rewrite::Options opts;
  opts.trace = true;
  auto out = rewrite::run(loaded, loaded.ir().graphs[0], opts);
  assert(out.rewrite_count == 2 &&
         out.stop_reason == rewrite::StopReason::FixedPoint &&
         out.trace.size() == 2);
  assert(out.graph.nodes.size() == 2);
  auto& root = out.graph.nodes[out.graph.root];
  assert(root.operands == std::vector<std::size_t>({0, 0}));
  assert(program::serialize(p) == before);
  auto printed = rewrite::printGraph(p.ir(), out.graph, false);
  auto again = compile(printed);
  assert(again.ir().graphs[0].nodes.size() == 2);
  // Strict schema and semantic validation catch independent error classes.
  auto j = nlohmann::json::parse(json);
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
    program::load("{\"version\":1,\"version\":1}");
  } catch (const std::invalid_argument& e) {
    duplicateRejected =
        std::string(e.what()).find("duplicate") != std::string::npos;
  }
  assert(duplicateRejected);
  // Decimal strings preserve extents beyond JSON's interoperable integer range.
  auto wide =
      compile("graph wide {input X:f32[18446744073709551615];yield X;}");
  auto wideLoaded = program::load(program::serialize(wide));
  assert(wideLoaded.ir().graphs[0].nodes[0].shape->at(0) == UINT64_MAX);
}
void guardsAndFailure() {
  const std::string graph = "graph g {input X:f32[2];yield (copy X);}";
  auto p = compile(
      "rule r {(copy X)=>X where {true || "
      "len(range(18446744073709551615))==0}}" +
      graph);
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 1);
  p = compile(
      "rule r {(copy X)=>X where {false && "
      "len(range(18446744073709551615))==0}}" +
      graph);
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 0);
  p = compile("rule r {(copy X)=>X where {1 / 0 == 0}}" + graph);
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 0);
  p = compile("rule r {(copy X)=>(scalar X)}" + graph);
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 0);
  p = compile(
      "rule r {(copy X)=>(transpose[@p] X) derive {@p={permutation:[9]};}}" +
      graph);
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 0);
  p = compile("rule r {(copy X)=>(copy X)}" + graph);
  assert(rewrite::run(p, p.ir().graphs[0]).stop_reason ==
         rewrite::StopReason::FixedPoint);
  auto limited = compile(
      "rule r {(copy X)=>X where {len(range(18446744073709551615))==0}}" +
      graph);
  auto out = rewrite::run(limited, limited.ir().graphs[0]);
  assert(out.rewrite_count == 0 &&
         out.stop_reason == rewrite::StopReason::ResourceLimit);
}
void sharingAndLimits() {
  auto p = compile(
      "rule r {(copy X)=>X} graph g {input X:f32[2];let "
      "a=(transpose[permutation=[0]] X);let b=(copy a);yield (add a b);}");
  auto out = rewrite::run(p, p.ir().graphs[0]);
  assert(out.rewrite_count == 1);
  const auto& root = out.graph.nodes[out.graph.root];
  assert(root.operands[0] == root.operands[1]);
  p = compile(
      "rule r {(add X X)=>X} graph g {input X:f32[2];let a=(copy X);let "
      "b=(copy X);yield (add a b);}");
  assert(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 0);
  p = compile(
      "rule r {(add (let A = (copy X)) (copy X)) => (add A A)} "
      "graph g {input X:f32[2];yield (add (copy X) (copy X));}");
  out = rewrite::run(p, p.ir().graphs[0]);
  assert(out.rewrite_count == 1);
  assert(out.graph.nodes[out.graph.root].operands[0] ==
         out.graph.nodes[out.graph.root].operands[1]);
  p = compile(
      "rule r {(add X Y)=>(add Y X)} graph g {input X:f32[2];input "
      "Y:f32[2];yield (add X Y);}");
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
  auto p = compile("rule r {(copy X)=>X} graph g {input X;yield (copy X);}");
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
  auto zero =
      rewrite::call("product",
                    {rewrite::Value(rewrite::List{
                        Integer(UINT64_MAX), Integer(UINT64_MAX), Integer(0)})},
                    budget);
  assert(rewrite::integer(zero) == 0);
  bool rejected = false;
  try {
    rewrite::call("ceil_div", {Integer(4), Integer(-2)}, budget);
  } catch (const rewrite::EvaluationError&) {
    rejected = true;
  }
  assert(rejected);
}
void testCodegen() {
  auto p = compile("rule r {X:[N] (copy X)=>X where {N>0}}");
  auto loaded = program::load(program::serialize(p));
  for (auto target : {codegen::Target::kRust, codegen::Target::kCpp}) {
    codegen::Options opts;
    opts.target = target;
    auto a = codegen::generate(p, opts), b = codegen::generate(loaded, opts);
    assert(a.ok() && b.ok());
    assert(a.files.size() == b.files.size());
    for (std::size_t i = 0; i < a.files.size(); ++i)
      assert(a.files[i].path == b.files[i].path &&
             a.files[i].contents == b.files[i].contents);
  }
}
void automatic(const std::string& path) {
  std::ifstream f(path);
  std::string definitions{std::istreambuf_iterator<char>(f), {}};
  auto p = compile(definitions + R"(
rule compose {(transpose[@a] (transpose[@b] X)) => (transpose[auto] X)}
graph g {input X:f32[2,3,4];let a=(transpose[permutation=[1,2,0]] X);yield (transpose[permutation=[2,0,1]] a);}
)",
                   false);
  assert(p.ir().rules[0].attribute_holes.empty());
  auto loaded = program::load(program::serialize(p));
  auto out = rewrite::run(loaded, loaded.ir().graphs[0]);
  assert(out.rewrite_count == 1);
  assert(out.graph.nodes[out.graph.root].attributes[0].elements[0].text == "0");
  assert(out.graph.nodes[out.graph.root].shape ==
         std::optional<std::vector<std::uint64_t>>({2, 3, 4}));
  p = compile(definitions + R"(
rule move_reduce {(reduce_sum[@r] (transpose[@p] A)) => (transpose[auto] (reduce_sum[auto] A))}
graph g {input A:f32[2,3,4];yield (reduce_sum[dimensions=[1]] (transpose[permutation=[2,0,1]] A));}
)",
              false, core::SemanticPolicy::kAlgebraic);
  loaded = program::load(program::serialize(p));
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
    guardsAndFailure();
    sharingAndLimits();
    metadataAndDepth();
    testCodegen();
    std::unique_ptr<rules_cc::cc::runfiles::Runfiles> files(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0]));
    assert(argc == 2);
    automatic(files->Rlocation(argv[1]));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
