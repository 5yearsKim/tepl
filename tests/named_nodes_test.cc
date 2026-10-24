#include <algorithm>
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
executable::ExecutableProgram compile(const std::string& text) {
  auto parsed = parse(dialect + text, "named.tepl");
  if (!parsed.ok())
    for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  expect(parsed.ok(), "new syntax parses");
  auto out = core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic});
  if (!out.ok())
    for (const auto& d : out.diagnostics) std::cerr << d.message << '\n';
  expect(out.ok(), "new syntax compiles");
  return std::move(*out.program);
}
void rejects(const std::string& text, const std::string& reason) {
  auto parsed = parse(dialect + text, "rejected.tepl");
  expect(parsed.ok(), "negative fixture parses");
  auto checked =
      core::compile(*parsed.program, {core::SemanticPolicy::kAlgebraic});
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
)");
  auto loaded = executable::load(executable::serialize(program));
  // Evaluate predicates even though rebuilding an identical dot is a fixed
  // point.
  rewrite::Budget budget;
  rewrite::Bindings bindings;
  const auto& r = loaded.ir().rules[0];
  for (const auto& c : r.captures) bindings.captures[c.id.value] = c.id.value;
  const auto& node = loaded.ir().graphs[0].nodes.back();
  auto source =
      std::find_if(r.descriptors.begin(), r.descriptors.end(),
                   [](const auto& d) {
                     return d.kind ==
                            core::completed::Descriptor::Kind::kCaptured;
                   })
          ->id.value;
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
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
