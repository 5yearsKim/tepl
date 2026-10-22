#include "src/core/pipeline.h"

#include <cassert>
#include <map>
#include <type_traits>

#include "src/core/lowering/lower.h"
#include "src/core/resolution/loading.h"
#include "src/rewrite/engine.h"

using namespace tepl;
using namespace tepl::core;
template <class T>
concept HasProofPlan = requires(T rule) { rule.semantic_plan; };
static_assert(!HasProofPlan<core::Rule>);
static_assert(!HasProofPlan<core::RuleData>);
static_assert(!std::is_default_constructible_v<checking::TypedProgram>);
static_assert(!std::is_default_constructible_v<semantics::VerifiedProgram>);
static_assert(!std::is_default_constructible_v<executable::ExecutableProgram>);
static_assert(!std::is_invocable_v<decltype(lowering::lower),
                                   const checking::TypedProgram&>);

template <class T>
concept MutableResolvedGraph = requires(const T& p) {
  p.data().rules[0].lhs->resolution_id = std::size_t{1};
};
template <class T>
concept HasInference = requires { T::kInferred; };
template <class T>
concept HasHole = requires { T::Hole; };
template <class T, class... Ts>
constexpr bool contains(std::variant<Ts...>*) {
  return (std::is_same_v<T, Ts> || ...);
}
static_assert(!MutableResolvedGraph<resolution::ResolvedProgram>);
static_assert(std::is_const_v<syntax::MetadataExprPtr::element_type>);
static_assert(std::is_const_v<syntax::SemanticExprPtr::element_type>);
static_assert(!HasInference<completed::Descriptor::Kind>);
static_assert(!HasHole<completed::Symbol::Kind>);
static_assert(!contains<OutputAxisGroupRef>(
    static_cast<decltype(completed::TypedExpr::value)*>(nullptr)));
static_assert(!std::is_convertible_v<TypedExprPtr, completed::TypedExprPtr>);

struct MemoryLoader : SourceLoader {
  std::map<std::string, std::string> files;
  std::map<std::string, std::size_t> reads;
  std::optional<std::string> read(const std::string& name) override {
    ++reads[name];
    if (auto i = files.find(name); i != files.end()) return i->second;
    return {};
  }
};
const std::string dialect = R"(
dialect D {
 op transpose(x:tensor)->tensor {
   attrs { permutation:index[]; }
   dtype(t) { yield t; }
   axes {
     bind S { x:all_axes(x) };
     assert is_valid_axis_list(attrs.permutation,rank(x));
     assert len(attrs.permutation)==rank(x);
     yield permute(S,attrs.permutation);
   }
   semantics { yield x; }
 }
}
)";
int main() {
  MemoryLoader loader;
  loader.files["/virtual/dialect.tepl"] = dialect;
  const SourceFile source{"/virtual/main.tepl", R"(
 from "dialect.tepl" import D as d;
 use d::{transpose};
 rule compose { (transpose (transpose X)) => (transpose X) }
 graph g { input X:f32[2,3,4]; yield (transpose[permutation=[2,0,1]] (transpose[permutation=[1,2,0]] X)); }
 )"};
  auto parsed = parse(source);
  assert(parsed.ok() && parsed.program->dialects.empty());
  // API callers can retain a mutable pointer when constructing source syntax.
  auto external_lhs =
      std::make_shared<syntax::GraphExpr>(*parsed.program->rules[0].lhs);
  parsed.program->rules[0].lhs = external_lhs;
  auto loaded = loadImports(*parsed.program, loader);
  assert(loaded.ok() && loaded.program->declarations.dialects.size() == 1);
  assert(parsed.program->dialects.empty());
  auto& operation = loaded.program->declarations.dialects[0].operations[0];
  auto external_dtype = std::make_shared<syntax::MetadataExpr>(
      *operation.dtype_definition->result.value);
  operation.dtype_definition->result.value = external_dtype;
  auto resolved = resolution::resolve(std::move(*loaded.program));
  assert(resolved.ok() && resolved.program->data().rules.size() == 1);
  external_lhs->value = syntax::NameRef{"changed_after_resolution"};
  external_dtype->value = syntax::NameRef{"invalid_dtype_after_resolution"};
  assert(std::holds_alternative<syntax::Operator>(
      resolved.program->data().sources->declarations.rules[0].lhs->value));
  assert(resolved.program->data().declarations.size() == 1);
  assert(resolved.program->data().declarations[0].id == OpId{0});
  auto typed = checking::check(*resolved.program);
  assert(typed.ok());
  const auto& typed_rule = typed.program->ir().rules[0];
  assert(typed_rule.attribute_holes.size() == 1);
  resolved.program.reset();
  loaded.program.reset();
  parsed.program.reset();
  auto verified = semantics::verify(*typed.program);
  assert(verified.ok());
  assert(verified.program->verification(RuleId{0}) ==
         semantics::VerificationKind::Semantic);
  assert(typed_rule.attribute_holes.size() == 1);

  assert(verified.program->ir().rules[0].derivations.size() == 1);
  auto executable = lowering::lower(*verified.program);
  assert(executable.ok());
  verified.program.reset();
  typed.program.reset();
  auto restored =
      tepl::executable::load(tepl::executable::serialize(*executable.program));
  auto out = rewrite::run(restored, restored.ir().graphs[0]);
  assert(out.rewrite_count == 1);
  const auto& permutation =
      out.graph.nodes[out.graph.root].attributes[0].elements;
  assert(permutation.size() == 3 && permutation[0].text == "0" &&
         permutation[1].text == "1" && permutation[2].text == "2");
  // A single project cache covers multiple roots and repeated import edges.
  MemoryLoader project_loader;
  project_loader.files = loader.files;
  const SourceFile second{"/virtual/second.tepl", R"(
    from "dialect.tepl" import D as a;
    from "dialect.tepl" import D as b;
    rule identity { X => X }
  )"};
  auto project =
      loadProject(std::vector<SourceFile>{source, second}, project_loader);
  assert(project.ok() && project.program->modules.size() == 3);
  assert(project_loader.reads.at("/virtual/dialect.tepl") == 1);
  assert(compile(*project.program).ok());
  // Root modules and imported modules share the cache, regardless of traversal
  // order.
  MemoryLoader seeded_loader;
  auto seeded = loadProject(
      std::vector<SourceFile>{source, {"/virtual/dialect.tepl", dialect}},
      seeded_loader);
  assert(seeded.ok() && seeded_loader.reads.empty());
  assert(compile(*seeded.program).ok());

  auto direct = compile(CompileRequest{source, {}}, loader);
  assert(direct.ok());
  assert(tepl::executable::serialize(*direct.program) ==
         tepl::executable::serialize(restored));
  // Output groups become runtime instructions during the single completion
  // pass.
  const SourceFile groups{"/virtual/groups.tepl", R"(
    from "dialect.tepl" import D as d;
    rule compose {
      (p: d.transpose (q: d.transpose X)) => (d.transpose X)
      where { p.axes.S.positions == q.axes.S.positions && len(p.axes.S.positions) == len(q.axes.S.positions) }
    }
    graph g { input X:f32[2,3,4]; yield (d.transpose[permutation=[2,0,1]] (d.transpose[permutation=[1,2,0]] X)); }
    graph identity { input X:f32[2,3,4]; yield (d.transpose[permutation=[0,1,2]] (d.transpose[permutation=[0,1,2]] X)); }
  )"};
  auto group_program = compile(CompileRequest{groups, {}}, loader);
  assert(group_program.ok());
  auto group_restored = tepl::executable::load(
      tepl::executable::serialize(*group_program.program));
  assert(rewrite::run(group_restored, group_restored.ir().graphs[0])
             .rewrite_count == 0);
  assert(rewrite::run(group_restored, group_restored.ir().graphs[1])
             .rewrite_count == 1);

  // Repeated shared subexpressions must compare without expanding into a tree.
  symbolic::Key left("leaf"), right("leaf");
  for (int i = 0; i < 40; ++i) {
    left = symbolic::Key("pair", {left, left});
    right = symbolic::Key("pair", {right, right});
  }
  assert(left == right);
  assert(!(left < right) && !(right < left));

  // Type checking succeeds independently of semantic equivalence.
  auto bad = parse(SourceFile{"/virtual/bad.tepl", R"(
 from "dialect.tepl" import D as d;
 rule invalid { (d.transpose X) => X }
 )"});
  auto bad_loaded = loadImports(*bad.program, loader);
  auto bad_resolved = resolution::resolve(std::move(*bad_loaded.program));
  auto bad_typed = checking::check(*bad_resolved.program);
  assert(bad_typed.ok());
  auto rejected = semantics::verify(*bad_typed.program);
  assert(!rejected.ok() && !rejected.program);
  auto inspected =
      semantics::inspect(*bad_typed.program, {SemanticPolicy::kStrict, true});
  assert(inspected.ok() &&
         inspected.program->rules[0].semantic_plan->proof.status ==
             semantics::ProofStatus::Unknown);
}
