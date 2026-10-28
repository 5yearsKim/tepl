#include <iostream>
#include <stdexcept>
#include <string>

#include "src/core/pipeline.h"
#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/symbolic/expr.h"
#include "src/executable/codec.h"
#include "src/rewrite/engine.h"

namespace {
using namespace tepl;
const std::string dialect = R"(
dialect D {
 op broadcast(x:tensor)->tensor {
  attrs { dimensions:axis[]; new_extents:extent[]; }
  dtype(t) { yield t; }
  axes { bind S {x:all_axes(x)}; bind N=group(attrs.new_extents);
         yield place(S,attrs.dimensions,N); }
  semantics { yield x; }
 }
 op reverse(x:tensor)->tensor {
  attrs { dimensions:axis[]; }
  dtype(t) {yield t;}
  axes {bind S {x:all_axes(x)}; yield S;}
  semantics {yield Read(x,IndexReverse(Coords(S),attrs.dimensions));}
 }
 op transpose(x:tensor)->tensor {
  attrs {permutation:axis[];}
  dtype(t){yield t;}
  axes {bind S {x:all_axes(x)};
    assert is_valid_axis_list(attrs.permutation,rank(x));
    assert len(attrs.permutation)==rank(x); yield permute(S,attrs.permutation);}
  semantics {yield x;}
 }
 op defaults(x:tensor)->tensor {
  attrs {permutation:axis[]; flags:bool[]=[]; note:string?;
         optional_axes:axis[]?; empty_axes:axis[]=[]; optional_flags:bool[]?=[];}
  dtype(t){yield t;}
  axes {bind S {x:all_axes(x)};
    assert is_valid_axis_list(attrs.permutation,rank(x));
    assert len(attrs.permutation)==rank(x); yield permute(S,attrs.permutation);}
  semantics {yield x;}
 }
 op shifted(x:tensor)->tensor {
  attrs {offset:coordinate[];}
  dtype(t){yield t;}
  axes {bind S {x:all_axes(x)}; yield S;}
  semantics {yield Read(x,IndexOffset(attrs.offset,Coords(S)));}
 }
 op settings(x:tensor)->tensor {
  attrs {count:u64; tag:i64;}
  dtype(t){yield t;}
  axes {bind S {x:all_axes(x)}; yield S;}
  semantics {yield x;}
 }
 op legacy(x:tensor)->tensor {
  attrs { permutation: u64[]; }
  dtype(t){yield t;}
  axes {bind S {x:all_axes(x)};
    assert is_valid_axis_list(attrs.permutation,rank(x));
    assert len(attrs.permutation)==rank(x); yield permute(S,attrs.permutation);}
  semantics {yield x;}
 }
}
)";
void expect(bool x, const std::string& message) {
  if (!x) throw std::runtime_error(message);
}
auto parsed(const std::string& body) {
  auto p = parse(dialect + body, "axis_auto.tepl");
  expect(p.ok(), "fixture must parse");
  return p;
}
void accepts(const std::string& body) {
  auto p = parsed(body);
  auto c = core::compile(*p.program, {core::SemanticPolicy::kAlgebraic});
  if (!c.ok())
    for (const auto& d : c.diagnostics) std::cerr << d.message << '\n';
  expect(c.ok(), "axis auto fixture compiles");
  expect(
      executable::load(executable::serialize(*c.program)).ir().rules.size() ==
          1,
      "completed holes survive executable round trip");
}
void rejects(const std::string& body, const std::string& needle) {
  auto p = parsed(body);
  auto c = core::compile(*p.program, {core::SemanticPolicy::kAlgebraic});
  expect(!c.ok(), "invalid inference request must fail");
  std::string messages;
  for (const auto& d : c.diagnostics) messages += d.message + "\n";
  expect(messages.find(needle) != std::string::npos, messages);
}
}  // namespace
int main() {
  try {
    accepts(R"(rule r {(s:transpose X)=>(t:transpose X)
      derive {t.attrs={permutation:auto};}})");
    accepts(R"(rule r {(s:transpose X)=>(transpose X)})");
    accepts(R"(rule r {(s:defaults X)=>(defaults X)})");
    accepts(R"(rule r {(s:defaults X)=>(t:defaults X)
      derive {t.attrs={permutation:auto};}})");
    accepts(R"(rule r {(s:shifted X)=>(shifted[s.attrs] X)})");
    rejects(R"(rule r {(s:defaults X)=>(t:defaults X)
      derive {t.attrs={permutation:auto,optional_axes:auto};}})",
            "non-optional axis");
    rejects(R"(rule r {(s:shifted X)=>(t:shifted X)
      derive {t.attrs={offset:auto};}})",
            "axis attribute");
    rejects(R"(rule r {(s:shifted X)=>(shifted X)})", "offset");
    accepts(R"(dialect DefaultsOnly {
      op f(x:tensor)->tensor {
        attrs {flags:bool[]=[]; note:string?; axes:axis[]=[];}
      }
    }
    rule r {(s:DefaultsOnly.f X)=>(DefaultsOnly.f X)})");
    const std::string partial = R"(rule r {(s:broadcast X)=>(t:broadcast X)
      derive {t.attrs={dimensions:auto,new_extents:s.attrs.new_extents};}})";
    accepts(partial);
    accepts(R"(rule r {(s:broadcast X)=>(t:broadcast X)
      derive {t.attrs={..s.attrs,dimensions:auto};}})");
    accepts(R"(rule r {(s:broadcast X)=>(broadcast[s.attrs] X)})");
    accepts(R"(rule r {(reverse (reverse X))=>(reverse X)})");
    rejects(R"(rule r {(s:broadcast X)=>(broadcast X)})", "new_extents");
    rejects(R"(rule r {(s:legacy X)=>(legacy X)})", "only axis");
    rejects(R"(rule r {(s:broadcast X)=>(t:broadcast X)
      derive {t.attrs={dimensions:auto,new_extents:auto};}})",
            "axis attribute");
    rejects(R"(rule r {(s:legacy X)=>(t:legacy X)
      derive {t.attrs={permutation:auto};}})",
            "axis attribute");
    rejects(R"(rule r {(s:transpose X)=>X where {auto==0}})",
            "only allowed directly");
    rejects(R"(rule r {(s:transpose X)=>(t:transpose X)
      derive {let a=auto; t.attrs={permutation:a};}})",
            "only allowed directly");
    rejects(R"(rule r {(s:broadcast X)=>(t:broadcast X)
      derive {t.attrs={dimensions:auto};}})",
            "new_extents");
    rejects(R"(rule r {(s:transpose X)=>(t:transpose X)
      derive {t.attrs={permutation:[auto]};}})",
            "only allowed directly");
    accepts(R"(rule r {(s:settings X)=>(t:settings X)
      derive {t.attrs={count:u64(7),tag:i64(7)};}})");
    accepts(R"(rule r {(s:settings X)=>(t:settings X)
      derive {t.attrs={count:18446744073709551615,tag:-1};}})");
    rejects(R"(rule r {(s:settings X)=>(t:settings X)
      derive {t.attrs={count:auto,tag:s.attrs.tag};}})",
            "axis attribute");
    rejects(R"(rule r {(s:settings X)=>(t:settings X)
      derive {t.attrs={count:s.attrs.count,tag:auto};}})",
            "axis attribute");
    rejects(R"(rule r {(s:settings X)=>(t:settings X)
      derive {t.attrs={count:u64(-1),tag:0};}})",
            "attribute type");
    rejects(R"(rule r {(s:settings X)=>(t:settings X)
      derive {t.attrs={count:0,tag:i64(18446744073709551615)};}})",
            "attribute type");
    auto removed = parse(R"(dialect Old {
      op f(x:tensor)->tensor {attrs {value:index;}}
    })",
                         "removed_index.tepl");
    expect(removed.ok(), "removed type fixture parses");
    expect(!core::compile(*removed.program).ok(),
           "index is rejected as a public attribute type");
    expect(!core::resolveType("index") && !core::resolveType("index_list"),
           "old index rule type aliases are removed");
    auto p = parsed(partial);
    auto inspected =
        core::inspect(*p.program, {core::SemanticPolicy::kAlgebraic});
    expect(inspected.ok(), "partial fixture inspects");
    const auto& plan = *inspected.program->rules[0].semantic_plan;
    expect(plan.proof.status == core::semantics::ProofStatus::Proved,
           "partial constructor has a semantic proof");
    auto candidate =
        static_cast<const core::semantics::ConstructorCandidate&>(plan.proof);
    const auto descriptor = plan.nodes[plan.rhs].descriptor->value;
    candidate.constructors[descriptor][1] =
        core::semantics::make(core::semantics::Symbol::Kind::List);
    expect(core::semantics::verify(plan, candidate).status ==
               core::semantics::ProofStatus::Unknown,
           "verifier rejects alteration of a supplied extent field");
    auto invalid_axis = candidate;
    invalid_axis.constructors[descriptor][0] =
        core::semantics::call("none", {});
    expect(core::semantics::verify(plan, invalid_axis).status ==
               core::semantics::ProofStatus::Unknown,
           "verifier rejects none for a non-optional axis field");
    auto bad = parse(R"(dialect D {op bad(x:tensor)->tensor {
      attrs {sizes:axis[];} dtype(t){yield t;}
      axes {bind O=group(attrs.sizes); yield O;}
      semantics {yield Read(x,Coords(O));}}})",
                     "bad_role.tepl");
    expect(bad.ok(), "invalid role fixture parses");
    expect(
        !core::compile(*bad.program, {core::SemanticPolicy::kAlgebraic}).ok(),
        "axis role cannot be used as an extent domain");
    std::cout << "axis-only auto tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
