#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/pipeline.h"
#include "src/rewrite/engine.h"
#include "src/rewrite/evaluator.h"

namespace {
using namespace tepl;
std::string dialect;
void expect(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
executable::ExecutableProgram compile(const std::string& text,
                                      bool guards = false) {
  auto parsed = parse(dialect + text, "named.tepl");
  if (!parsed.ok())
    for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  expect(parsed.ok(), "new syntax parses");
  auto out = core::compile(*parsed.program,
                           {core::SemanticPolicy::kAlgebraic, guards});
  if (!out.ok())
    for (const auto& d : out.diagnostics) std::cerr << d.message << '\n';
  expect(out.ok(), "new syntax compiles");
  return std::move(*out.program);
}
void rejects(const std::string& text, const std::string& reason) {
  auto parsed = parse(dialect + text, "rejected.tepl");
  expect(parsed.ok(), "negative fixture parses");
  auto checked =
      core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic, true});
  expect(!checked.ok(), "negative fixture rejected: " + text);
  bool found = false;
  for (const auto& d : checked.diagnostics)
    found |= d.message.find(reason) != std::string::npos;
  if (!found)
    for (const auto& d : checked.diagnostics) std::cerr << d.message << '\n';
  expect(found, "expected diagnostic: " + reason);
}
const std::string graph = R"(
graph g { input A:f32[2,2];
 let x=(transpose[permutation=[1,0]] A);
 yield (transpose[permutation=[1,0]] x); }
)";
void inferenceAndDerivation() {
  auto inferred = compile(R"(
rule composition {
 (outer: transpose (inner: transpose A)) => (result: transpose A)
}
)" + graph);
  expect(inferred.ir().rules[0].captures.size() == 1,
         "operation labels do not introduce tensor captures");
  expect(inferred.ir().rules[0].derivations.size() == 1,
         "omitted attributes request synthesis");
  auto loaded = executable::load(executable::serialize(inferred));
  auto out = rewrite::run(loaded, loaded.ir().graphs[0]);
  expect(out.rewrite_count == 1, "inferred permutation applies");
  expect(out.graph.nodes[out.graph.root].attributes[0].elements[0].text == "0",
         "composition infers identity");

  auto explicit_attrs = compile(R"(
rule composition {
 (outer: transpose (inner: transpose A)) => (result: transpose A)
 derive {
   let p = gather(inner.attrs.permutation, outer.attrs.permutation);
   result.attrs = { ..outer.attrs, permutation: p };
 }
}
)" + graph);
  auto manual = rewrite::run(explicit_attrs, explicit_attrs.ir().graphs[0]);
  expect(manual.rewrite_count == 1 &&
             rewrite::attributeValue(
                 manual.graph.nodes[manual.graph.root].attributes[0]) ==
                 rewrite::attributeValue(
                     out.graph.nodes[out.graph.root].attributes[0]),
         "derive supplies the same complete descriptor as inference");
  auto anonymous = compile(
      "rule composition {(transpose (transpose A)) => (transpose A)}" + graph);
  expect(rewrite::run(anonymous, anonymous.ir().graphs[0]).rewrite_count == 1,
         "node names are optional");
  auto copy = compile(R"(
rule same {(source: transpose A) => (transpose[source.attrs] A)}
)" + graph);
  expect(copy.ir().rules[0].derivations.empty(),
         "inline descriptor reuse needs no inference");
}
void symbolicGroups() {
  compile(R"(
rule no_attributes {
 A:[Dims...]
 (source: add A B) => (add A B)
 where {source.axes.S.positions == range(len(Dims))}
}
)");

  auto program = compile(R"(
rule keep {
 (source: dot A B) => (result: dot A B)
 where {
   let a = A.axes.order;
   let b = B.axes.order;
   source.axes.B.positions == [0,1]
   && source.axes.B.order == gather(a, [1,0])
   && source.axes.B.order == gather(b, [0,1])
   && source.axes.I.order == gather(a, [2])
   && source.axes.K.order == gather(a, [3])
   && source.axes.B.members == difference(source.axes.B.members, source.axes.K.members)
 }
}
graph g {
 input A:f32[2,3,4,5]; input B:f32[3,2,5,6];
 yield (dot[lhs_batching_dimensions=[1,0],rhs_batching_dimensions=[0,1],lhs_contracting_dimensions=[3],rhs_contracting_dimensions=[2]] A B);
}
)",
                         true);
  auto loaded = executable::load(executable::serialize(program));
  // Evaluate predicates even though rebuilding an identical dot is a fixed
  // point.
  rewrite::Budget budget;
  rewrite::Bindings bindings;
  const auto& r = loaded.ir().rules[0];
  for (const auto& c : r.captures) bindings.captures[c.id.value] = c.id.value;
  const auto& node = loaded.ir().graphs[0].nodes.back();
  auto source = r.inference->source.back().descriptor.value();
  auto schema =
      loaded.ir().types[r.descriptors[source].type.value].schema.value();
  rewrite::DescriptorValue descriptor{schema.value, {}};
  for (const auto& a : node.attributes)
    descriptor.fields.push_back(rewrite::attributeValue(a));
  bindings.descriptors[source] = descriptor;
  rewrite::Evaluator eval(loaded, loaded.ir().graphs[0], 0, bindings, budget);
  expect(eval.evaluateWhere(),
         "members, group order, input positions, and output positions have "
         "distinct correct meanings");

  rejects(
      "rule r {(d: dot A B)=>(dot A B) where {d.axes.B.members == "
      "d.axes.B.order}}",
      "type mismatch");
  rejects(
      "rule r {(d: dot A B)=>(dot A B) where {d.axes.Missing.members == "
      "d.axes.B.members}}",
      "unknown semantic axis group");
  rejects("rule r {(d: dot A B)=>(dot A B) where {d.axes.K.positions == []}}",
          "absent");
}
void dependenciesAndErrors() {
  compile(R"(
rule shared {
 (outer: transpose (inner: transpose[outer.attrs] A)) => (transpose A)
 where { outer.axes.S.positions == inner.axes.S.positions }
}
)");

  auto p = compile(R"(
rule composition {
 (outer: transpose (inner: transpose A)) => (result: transpose A)
 where {
   let positions = result.axes.S.positions;
   result.attrs.permutation == [1,0]
   && positions == [1,0]
   && result.axes.S.order == A.axes.order
 }
 derive { result.attrs = outer.attrs; }
}
graph g {input A:f32[2,2];let x=(transpose[permutation=[0,1]] A);yield (transpose[permutation=[1,0]] x);}
)",
                   true);
  expect(rewrite::run(p, p.ir().graphs[0]).rewrite_count == 1,
         "where reads derived attributes after their definition");
  rejects("rule r {(same: transpose (same: transpose A)) => A}",
          "duplicate node label");
  rejects("rule r {(A: transpose A) => A}", "conflicts");
  rejects(
      "rule r {(source: transpose A)=>(target: transpose A) derive "
      "{source.attrs=target.attrs;}}",
      "read-only");
  rejects(
      "rule r {(source: transpose A)=>(target: transpose A) derive "
      "{missing.attrs=source.attrs;}}",
      "unknown RHS node");
  rejects(
      "rule r {(source: transpose A)=>(target: transpose[source.attrs] A) "
      "derive {target.attrs=source.attrs;}}",
      "competing attribute");
  rejects(
      "rule r {(source: transpose A)=>(target: transpose A) where "
      "{target.attrs.permutation == []}}",
      "unavailable before construction");
  rejects(
      "rule r {(source: transpose A)=>(target: transpose A) where "
      "{target.axes.S.order == A.axes.order}}",
      "unavailable before construction");
  rejects(
      "rule r {(source: transpose A)=>(target: transpose A) derive "
      "{target.attrs=target.attrs;}}",
      "forward reference or cycle");
}
void suggestions() {
  const std::string input = R"(
rule r {(transpose (dot wanted B)) => (dot wanted B)}
)";
  auto parsed = parse(dialect + input, "suggestion.tepl");
  expect(parsed.ok(), "suggestion fixture parses");
  auto inspected = core::inspect(
      *parsed.program, {core::SemanticPolicy::kAlgebraic, false, {}, true});
  expect(inspected.ok(), "generate symbolic source suggestion");
  auto suggestion = inspected.program->rules[0].semantic_plan->suggested_rule;
  expect(suggestion.find("axis_required(") == std::string::npos &&
             suggestion.find("[auto") == std::string::npos &&
             suggestion.find('@') == std::string::npos,
         "suggestions contain no numeric query API or old bindings");
  expect(suggestion.find("wanted_2") != std::string::npos,
         "generated names avoid capture collisions");
  auto text = suggestion + R"(
graph feasible {input wanted:f32[2];input B:f32[2];let x=(dot[lhs_batching_dimensions=[],rhs_batching_dimensions=[],lhs_contracting_dimensions=[],rhs_contracting_dimensions=[]] wanted B);yield (transpose[permutation=[0,1]] x);}
)";
  auto compiled = compile(text);
  expect(rewrite::run(compiled, compiled.ir().graphs[0]).rewrite_count == 1,
         "complete suggestion reparses and compiles without a flag");
  auto g = compiled.ir().graphs[0];
  std::swap(g.nodes[g.root].attributes[0].elements[0],
            g.nodes[g.root].attributes[0].elements[1]);
  expect(rewrite::run(compiled, g).rewrite_count == 0,
         "symbolic suggestion preserves infeasibility");

  auto qualified = parse(dialect + R"(
rule qualified {(TensorLang.transpose (TensorLang.dot A B)) => (TensorLang.dot A B)}
)",
                         "qualified.tepl");
  expect(qualified.ok(), "qualified source parses");
  auto qualified_info = core::inspect(
      *qualified.program, {core::SemanticPolicy::kAlgebraic, false, {}, true});
  expect(qualified_info.ok(), "qualified suggestion generated");
  auto qualified_text =
      qualified_info.program->rules[0].semantic_plan->suggested_rule;
  expect(qualified_text.find(": TensorLang.dot") != std::string::npos,
         "suggestion preserves qualified operation spellings");
  compile(qualified_text);

  auto nested = parse(dialect + R"(
rule associate {(d: dot (inner: dot A B) C) => (dot A (dot B C))}
)",
                      "nested.tepl");
  auto nested_info = core::inspect(
      *nested.program, {core::SemanticPolicy::kAlgebraic, false, {}, true});
  expect(nested_info.ok(), "nested suggestion generated");
  auto nested_rule =
      nested_info.program->rules[0].semantic_plan->suggested_rule;
  auto nested_program = compile(nested_rule + R"(
graph reassociate {
 input A:f32[2,3]; input B:f32[3,4]; input C:f32[4,5];
 let inner=(dot[lhs_contracting_dimensions=[1],rhs_contracting_dimensions=[0],lhs_batching_dimensions=[],rhs_batching_dimensions=[]] A B);
 yield (dot[lhs_contracting_dimensions=[1],rhs_contracting_dimensions=[0],lhs_batching_dimensions=[],rhs_batching_dimensions=[]] inner C);
}
)");
  auto loaded_nested = executable::load(executable::serialize(nested_program));
  auto nested_result =
      rewrite::run(loaded_nested, loaded_nested.ir().graphs[0]);
  expect(nested_result.rewrite_count == 1 &&
             nested_result.graph.nodes[nested_result.graph.root].shape ==
                 std::optional<std::vector<std::uint64_t>>({2, 5}),
         "suggested nested rule constructs legal reassociation attributes "
         "after JSON replay");
  // B/C batching moves that class before A in the source output. The fixed
  // RHS root can only output A's free axes before the inner B/C result.
  auto impossible_nested = compile(nested_rule + R"(
graph impossible {
 input A:f32[2]; input B:f32[3]; input C:f32[3];
 let inner=(dot[lhs_contracting_dimensions=[],rhs_contracting_dimensions=[],lhs_batching_dimensions=[],rhs_batching_dimensions=[]] A B);
 yield (dot[lhs_contracting_dimensions=[],rhs_contracting_dimensions=[],lhs_batching_dimensions=[1],rhs_batching_dimensions=[0]] inner C);
}
)");
  expect(rewrite::run(impossible_nested, impossible_nested.ir().graphs[0])
                 .rewrite_count == 0,
         "suggested nested condition rejects an impossible batch/free output "
         "order");

  // User conditions and descriptor derivations remain in a complete suggestion.
  auto source = parse(dialect + R"(
rule explicit {(outer: transpose (inner: transpose A)) => (result: transpose A)
 where { outer.attrs.permutation == [1,0] }
 derive {result.attrs = outer.attrs;}}
)",
                      "explicit-suggestion.tepl");
  auto info = core::inspect(
      *source.program, {core::SemanticPolicy::kAlgebraic, false, {}, true});
  expect(info.ok(), "explicit suggestion generated");
  auto s = info.program->rules[0].semantic_plan->suggested_rule;
  expect(s.find("result.attrs = outer.attrs") != std::string::npos &&
             s.find("outer.attrs.permutation") != std::string::npos,
         "suggestion retains original metadata definitions and restrictions");
  compile(s);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    std::string error;
    auto files = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));
    expect(files && argc == 2, "runfiles");
    std::ifstream input(files->Rlocation(argv[1]));
    dialect.assign(std::istreambuf_iterator<char>(input), {});
    expect(!dialect.empty(), "tensor dialect");
    inferenceAndDerivation();
    symbolicGroups();
    dependenciesAndErrors();
    suggestions();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
