#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/codegen/generate.h"
#include "src/core/pipeline.h"
#include "src/inference/evaluate.h"
#include "src/rewrite/engine.h"
#include "src/shared/catalog/axis_conditions.h"

namespace {
using namespace tepl;
std::string dialect;
void expect(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}
executable::ExecutableProgram compile(const std::string& text) {
  auto parsed = parse(dialect + text, "inferred.tepl");
  expect(parsed.ok(), "inferred fixture parses");
  auto compiled =
      core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic, true});
  if (!compiled.ok()) {
    for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
  }
  expect(compiled.ok(), "inferred fixture compiles");
  return std::move(*compiled.program);
}
void invalid(const nlohmann::json& j, const std::string& reason) {
  try {
    executable::load(j.dump());
  } catch (const std::invalid_argument& e) {
    expect(std::string(e.what()).find(reason) != std::string::npos,
           "expected rejection: " + reason + ", got: " + e.what());
    return;
  }
  throw std::runtime_error("mutated inference program was accepted");
}
core::AttributeValue positions(std::size_t rank) {
  core::AttributeValue out{core::AttributeValue::Kind::kList, {}, {}};
  for (std::size_t i = 0; i < rank; ++i)
    out.elements.push_back(
        {core::AttributeValue::Kind::kInteger, std::to_string(i), {}});
  return out;
}
const std::string rule = R"(
rule remove_output_permutation {
 (source: transpose (dot A B)) => (result: dot A B)
}
)";
const std::string graph = R"(
graph g {
 input A:f32[2]; input B:f32[2];
 let x = (dot[lhs_contracting_dimensions=[],rhs_contracting_dimensions=[],lhs_batching_dimensions=[],rhs_batching_dimensions=[]] A B);
 yield (transpose[permutation=[0,1]] x);
}
)";

void exactMatchesAndSerialization() {
  auto p = compile(rule + graph);
  expect(p.ir().rules[0].inference && p.ir().rules[0].derivations.empty(),
         "rule contains a completed inference program instead of symbolic "
         "constructors");
  auto text = executable::serialize(p);
  expect(text.find("tepl.inferred-applicability.v1") != std::string::npos,
         "serialized program advertises required capability");
  auto loaded = executable::load(text);
  auto generated = codegen::generate(loaded);
  expect(!generated.ok() && generated.files.empty() &&
             generated.diagnostics[0].message.find("native C++") !=
                 std::string::npos,
         "deferred backends reject programs requiring inferred applicability");
  expect(executable::serialize(loaded) == text,
         "completed inference round-trips");
  auto g = loaded.ir().graphs[0];
  auto out = rewrite::run(loaded, g);
  expect(out.rewrite_count == 1, "feasible match constructs the RHS");
  auto& permutation = g.nodes[g.root].attributes[0].elements;
  std::swap(permutation[0], permutation[1]);
  out = rewrite::run(loaded, g);
  expect(out.rewrite_count == 0 &&
             out.stop_reason == rewrite::StopReason::FixedPoint,
         "equal-shaped but impossible match is skipped");

  // Exactly one compiled program handles scalars and increasing ranks.
  for (std::size_t rank : {0u, 1u, 5u, 32u, 128u}) {
    g = loaded.ir().graphs[0];
    g.nodes[0].shape = std::vector<std::uint64_t>(rank, 2);
    g.nodes[1].shape = std::vector<std::uint64_t>{};
    g.nodes[g.root].attributes[0] = positions(rank);
    out = rewrite::run(loaded, g);
    expect(out.rewrite_count == 1 &&
               out.graph.nodes[out.graph.root].shape == g.nodes[0].shape,
           "compiled schedules retain runtime rank lengths");
    if (rank > 1) {
      auto& order = g.nodes[g.root].attributes[0].elements;
      std::reverse(order.begin(), order.end());
      expect(rewrite::run(loaded, g).rewrite_count == 0,
             "where subsequence checks reject reversed input order at "
             "arbitrary rank");
    }
  }
  g = loaded.ir().graphs[0];
  g.nodes[0].shape = std::vector<std::uint64_t>{1};
  expect(rewrite::run(loaded, g).rewrite_count == 0,
         "outside-domain metadata does not run the constructor");

  auto j = nlohmann::json::parse(text);
  auto& program = j["program"]["rules"][0]["inference"];
  auto original = program;
  program["replacement"].back()["inputs"] = {999};
  invalid(j, "producer-first");
  program = original;
  program["replacement"].back()["captures"] = nlohmann::json::array();
  invalid(j, "capture set");
  program = original;
  auto fields = program["replacement"].back()["fields"];
  std::swap(fields[0], fields[1]);
  program["replacement"].back()["fields"] = fields;
  invalid(j, "field adapter");
  program = original;
  j["policy"] = "strict";
  invalid(j, "algebraic policy");
  j["policy"] = "algebraic";
  j["capabilities"].erase(j["capabilities"].end() - 1);
  invalid(j, "capability");

  auto parsed = parse(dialect + rule, "coverage.tepl");
  expect(
      !core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic}).ok(),
      "ordinary acceptance still requires coverage under declared conditions");
  expect(!core::compile(*parsed.program, {core::SemanticPolicy::kStrict, true})
              .ok(),
         "inferred mode requires the theorem's arithmetic policy");
}

void internalConstructionAndConditions() {
  auto p = compile(R"(
rule internal {
 (p: transpose (d: dot A B)) => (dot (transpose A) B)
 where { p.attrs.permutation == [1,0,2] }
}
graph g {
 input A:f32[2,3]; input B:f32[5];
 let x=(dot[lhs_contracting_dimensions=[],rhs_contracting_dimensions=[],lhs_batching_dimensions=[],rhs_batching_dimensions=[]] A B);
 yield (transpose[permutation=[1,0,2]] x);
})");
  auto loaded = executable::load(executable::serialize(p));
  auto out = rewrite::run(loaded, loaded.ir().graphs[0]);
  expect(out.rewrite_count == 1,
         "nontrivial constructor executes after feasibility");
  const auto& root = out.graph.nodes[out.graph.root];
  const auto& transpose = out.graph.nodes[root.operands[0]];
  expect(transpose.attributes[0].elements[0].text == "1" &&
             transpose.attributes[0].elements[1].text == "0",
         "internal permutation comes from required parent order");
  expect(root.shape == std::optional<std::vector<std::uint64_t>>({3, 2, 5}),
         "constructed RHS has the source's ordered shape");
  auto g = loaded.ir().graphs[0];
  g.nodes[g.root].attributes[0] = positions(3);
  expect(rewrite::run(loaded, g).rewrite_count == 0,
         "declared condition remains part of applicability");
}

void explicitValidation() {
  auto p = compile(R"(
rule explicit {
 (outer: transpose (inner: transpose A)) => (out: transpose A)
 derive { out.attrs = {permutation: outer.attrs.permutation}; }
}
graph g {
 input A:f32[2,2];
 let x=(transpose[permutation=[0,1]] A);
 yield (transpose[permutation=[1,0]] x);
})");
  auto loaded = executable::load(executable::serialize(p));
  expect(loaded.ir().rules[0].inference->mode == inference::Mode::Validate,
         "fully explicit RHS generates supplied-assignment validation");
  auto g = loaded.ir().graphs[0];
  expect(rewrite::run(loaded, g).rewrite_count == 1,
         "correct explicit assignment applies");
  auto inner = g.nodes[g.root].operands[0];
  std::swap(g.nodes[inner].attributes[0].elements[0],
            g.nodes[inner].attributes[0].elements[1]);
  expect(rewrite::run(loaded, g).rewrite_count == 0,
         "wrong explicit assignment is rejected despite equal output shapes");

  auto shared = compile(R"(
rule same {(p: transpose (transpose[p.attrs] A)) => (transpose A)}
graph g {input A:f32[2,2];let x=(transpose[permutation=[1,0]] A);
yield (transpose[permutation=[1,0]] x);}
)");
  auto out = rewrite::run(shared, shared.ir().graphs[0]);
  expect(
      out.rewrite_count == 1 &&
          out.graph.nodes[out.graph.root].attributes[0].elements[0].text == "0",
      "shared source descriptors retain their actual values");
}

void explicitWhereMigration() {
  auto generated = compile(rule + graph);
  const auto& r = generated.ir().rules[0];
  expect(r.conditions.size() > 1,
         "feasibility is stored as ordinary where conditions");
  auto source = parse(dialect + rule, "suggest.tepl");
  auto info = core::inspect(
      *source.program, {core::SemanticPolicy::kAlgebraic, false, {}, true});
  expect(info.ok(), "source suggestion generated");
  auto text = info.program->rules[0].semantic_plan->suggested_rule;
  auto parsed = parse(dialect + text + graph, "explicit-where.tepl");
  expect(parsed.ok(), "suggested where clause parses as TEPL");
  auto compiled =
      core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic});
  if (!compiled.ok())
    for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
  expect(compiled.ok(),
         "explicit constraints compile without inferred-applicability flag");
  auto loaded = executable::load(executable::serialize(*compiled.program));
  expect(rewrite::run(loaded, loaded.ir().graphs[0]).rewrite_count == 1,
         "explicit where admits feasible attributes");
  auto g = loaded.ir().graphs[0];
  std::swap(g.nodes[g.root].attributes[0].elements[0],
            g.nodes[g.root].attributes[0].elements[1]);
  expect(rewrite::run(loaded, g).rewrite_count == 0,
         "ordinary where rejects infeasible layout before construction");

  // The constructor cannot be reached by removing or weakening its conditions
  // in serialized input. Recognition is repeated by the executable loader.
  auto j = nlohmann::json::parse(executable::serialize(loaded));
  auto saved = j["program"]["rules"][0]["conditions"];
  j["program"]["rules"][0]["conditions"] = nlohmann::json::array();
  invalid(j, "missing axis feasibility condition");
  j["program"]["rules"][0]["conditions"] = saved;
  j["program"]["rules"][0]["conditions"].back()["value"] = {{"kind", "bool"},
                                                            {"data", true}};
  invalid(j, "missing axis feasibility condition");

  auto missing = parse(dialect + R"(
rule r {(p: transpose (d: dot A B)) => (dot A B)
 where { axis_domain() }}
)",
                       "missing.tepl");
  expect(missing.ok(), "incomplete where has valid syntax");
  expect(
      !core::compile(*missing.program, {core::SemanticPolicy::kAlgebraic}).ok(),
      "domain guard alone cannot justify the constructor");

  auto weakened = text;
  weakened.insert(weakened.find("axis_domain()"), "true || ");
  auto weak = parse(dialect + weakened, "weak-where.tepl");
  expect(weak.ok() &&
             !core::compile(*weak.program, {core::SemanticPolicy::kAlgebraic})
                  .ok(),
         "obligations hidden under a disjunction do not justify construction");

  // Local lets and grouping of conjunctions do not change the recognized law.
  auto local_text = text;
  for (std::size_t pos = 0;
       (pos = local_text.find("fromA", pos)) != std::string::npos; pos += 15)
    local_text.replace(pos, 5, "from_left_order");
  auto locals = parse(dialect + local_text + graph, "local-where.tepl");
  expect(locals.ok() &&
             core::compile(*locals.program, {core::SemanticPolicy::kAlgebraic})
                 .ok(),
         "generated constraints recognized through local bindings");

  auto wrong_type = parse(dialect + R"(
rule r {(p: transpose A) => (transpose A)
 where { axis_domain() && subsequence([0], A.axes.order) }}
)",
                          "typed-axes.tepl");
  expect(wrong_type.ok() && !core::compile(*wrong_type.program,
                                           {core::SemanticPolicy::kAlgebraic})
                                 .ok(),
         "logical axes cannot be confused with numeric positions");
}

void explicitWhereDependencies() {
  const std::string body = R"(
rule explicit {
 (outer: transpose (inner: transpose A)) => (out: transpose A)
 WHERE_CLAUSE
 derive { out.attrs = {permutation: outer.attrs.permutation}; }
}
graph g { input A:f32[2,2];
 let x=(transpose[permutation=[0,1]] A);
 yield (transpose[permutation=[1,0]] x); }
)";
  auto seed = body;
  seed.replace(seed.find("WHERE_CLAUSE"), 12, "");
  auto generated = compile(seed);
  auto illegal = seed;
  illegal.replace(illegal.find("outer.attrs.permutation"),
                  std::string("outer.attrs.permutation").size(), "[]");
  auto rejected_rhs = compile(illegal);
  expect(
      rewrite::run(rejected_rhs, rejected_rhs.ir().graphs[0]).rewrite_count ==
          0,
      "where RHS legality condition rejects an illegal derived permutation");
  auto text = body;
  text.replace(text.find("WHERE_CLAUSE"), 12,
               R"(where {
 let source_order = outer.axes.order;
 let replacement_order = out.axes.order;
 axis_domain() && out.legal &&
 outer.axes.partition == out.axes.partition &&
 source_order == replacement_order
})");
  auto parsed = parse(dialect + text, "explicit-dependencies.tepl");
  expect(parsed.ok(), "explicit validation where parses");
  auto compiled =
      core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic});
  expect(compiled.ok(), "explicit validation with local dependencies compiles");
  auto loaded = executable::load(executable::serialize(*compiled.program));
  auto g = loaded.ir().graphs[0];
  expect(rewrite::run(loaded, g).rewrite_count == 1,
         "RHS where facts evaluate after descriptor derivation");
  auto inner = g.nodes[g.root].operands[0];
  std::swap(g.nodes[inner].attributes[0].elements[0],
            g.nodes[inner].attributes[0].elements[1]);
  expect(rewrite::run(loaded, g).rewrite_count == 0,
         "dependent where rejects wrong supplied RHS attributes");
}

void unsupported() {
  for (const auto& text :
       {"rule r {(p: transpose A)=>(transpose (transpose[p.attrs] A))}",
        "rule r {(d: dot A A)=>(dot A A)}",
        "rule r {(p: transpose A)=>(out: transpose (transpose A)) derive "
        "{out.attrs=p.attrs;}}"}) {
    auto parsed = parse(dialect + text, "unsupported.tepl");
    expect(parsed.ok(), "unsupported feature fixture has valid syntax");
    expect(!core::compile(*parsed.program,
                          {core::SemanticPolicy::kAlgebraic, true})
                .ok(),
           "unsupported coupling/repeated captures do not enter complete mode");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string error;
    auto files = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    expect(files && argc == 2, "load runfiles");
    std::ifstream input(files->Rlocation(argv[1]));
    dialect.assign(std::istreambuf_iterator<char>(input), {});
    expect(!dialect.empty(), "load tensor semantics");
    exactMatchesAndSerialization();
    internalConstructionAndConditions();
    explicitValidation();
    explicitWhereMigration();
    explicitWhereDependencies();
    unsupported();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
