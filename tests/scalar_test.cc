#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "src/core/pipeline.h"
#include "src/core/semantics/ir.h"
#include "src/core/semantics/proof/verify.h"
#include "src/core/semantics/symbolic/expr.h"
#include "src/core/syntax/parse.h"
#include "src/executable/codec.h"

namespace {
std::string operation(std::string name, std::string inputs, std::string types,
                      std::string bindings, std::string value,
                      std::string attrs = "") {
  return "op " + name + "(" + inputs + ")->tensor {" + attrs + types +
         " axes {bind S {" + bindings + "}; yield S;} semantics {yield " +
         value + ";}}";
}
std::string dialect() {
  const std::string binary = "lhs:tensor,rhs:tensor";
  const std::string numeric =
      "dtype(a,b){assert a==b; assert is_numeric(a); yield a;}";
  const std::string bindings = "lhs:all_axes(lhs),rhs:all_axes(rhs)";
  std::string d = "dialect D {";
  for (auto name : {"min", "max"})
    d += operation(name, binary, numeric, bindings,
                   std::string(name) + "(lhs,rhs)");
  d += operation("add", binary, numeric, bindings, "add(lhs,rhs)");
  d += operation("mul", binary, numeric, bindings, "mul(lhs,rhs)");
  for (auto name : {"exp", "log", "sqrt"})
    d += operation(name, "x:tensor", "dtype(a){assert is_float(a);yield a;}",
                   "x:all_axes(x)", std::string(name) + "(x)");
  d += operation("abs", "x:tensor", "dtype(a){assert is_numeric(a);yield a;}",
                 "x:all_axes(x)", "abs(x)");
  d += operation("zero", "x:tensor", "dtype(a){assert is_numeric(a);yield a;}",
                 "x:all_axes(x)", "Const(0)");
  d += operation("div", binary, numeric, bindings, "div(lhs,rhs)");
  d += operation("pow", binary, numeric, bindings, "pow(lhs,rhs)");
  d += operation("compare", binary,
                 "dtype(a,b){assert a==b;assert is_numeric(a);yield bool;}",
                 bindings, "compare(lhs,rhs,attrs.direction)",
                 "attrs {direction: u64;}");
  d += operation("not", "x:tensor", "dtype(a){assert a==bool;yield bool;}",
                 "x:all_axes(x)", "not(x)");
  d += "op transpose(x:tensor)->tensor {attrs {p:axis[];} dtype(a){yield a;} "
       "axes {bind S {x:all_axes(x)}; assert "
       "is_valid_axis_list(attrs.p,rank(x)); assert "
       "len(attrs.p)==rank(x);yield permute(S,attrs.p);} semantics {yield x;}}";
  d += "op reverse(x:tensor)->tensor {attrs {p:axis[];} dtype(a){yield a;} "
       "axes {bind S {x:all_axes(x)}; assert "
       "is_valid_axis_list(attrs.p,rank(x)); yield S;} "
       "semantics {yield Read(x,IndexReverse(Coords(S),attrs.p));}}";
  d += "op sum(x:tensor)->tensor {attrs {p:axis[];} dtype(a){assert "
       "is_numeric(a);yield a;} axes {bind K {x:attrs.p};bind F "
       "{x:ordered_complement(all_axes(x),attrs.p)};yield F;} semantics {yield "
       "Reduce(K,x);}}";
  return d + "}";
}
void check(std::string rule, bool expected, bool algebraic = true,
           std::string definitions = {}) {
  auto parsed = tepl::parse(
      (definitions.empty() ? dialect() : definitions) + rule, "scalar.tepl");
  if (!parsed.ok())
    throw std::runtime_error("scalar fixture did not parse: " + rule);
  auto result = tepl::core::inspect(
      *parsed.program, {algebraic ? tepl::core::SemanticPolicy::kAlgebraic
                                  : tepl::core::SemanticPolicy::kStrict,
                        true});
  bool proved = result.ok() && result.program->rules[0].semantic_plan &&
                result.program->rules[0].semantic_plan->proof.status ==
                    tepl::core::semantics::ProofStatus::Proved;
  if (proved != expected) {
    for (const auto& diagnostic : result.diagnostics)
      std::cerr << diagnostic.message << '\n';
    if (result.ok() && result.program->rules[0].semantic_plan)
      std::cerr << result.program->rules[0].semantic_plan->proof.reason << '\n';
    throw std::runtime_error("unexpected scalar verification result: " + rule);
  }
  if (proved) {
    const auto& plan = *result.program->rules[0].semantic_plan;
    for (const auto& [descriptor, fields] : plan.proof.constructors)
      for (std::size_t field = 0; field < fields.size(); ++field)
        if (fields[field]->kind == tepl::core::symbolic::Symbol::Kind::Field) {
          auto corrupted = plan.proof;
          auto value =
              std::make_shared<tepl::core::symbolic::Symbol>(*fields[field]);
          value->sort =
              value->sort == tepl::core::symbolic::Symbol::Sort::Integer
                  ? tepl::core::symbolic::Symbol::Sort::Boolean
                  : tepl::core::symbolic::Symbol::Sort::Integer;
          corrupted.constructors.at(descriptor)[field] = value;
          if (tepl::core::semantics::verify(plan, corrupted).status !=
              tepl::core::semantics::ProofStatus::Unknown)
            throw std::runtime_error("forged source field sort accepted");
        } else if (fields[field]->kind ==
                   tepl::core::symbolic::Symbol::Kind::Integer) {
          auto corrupted = plan.proof;
          corrupted.constructors.at(descriptor)[field] =
              tepl::core::semantics::boolean(true);
          if (tepl::core::semantics::verify(plan, corrupted).status !=
              tepl::core::semantics::ProofStatus::Unknown)
            throw std::runtime_error(
                "scalar constructor type corruption accepted");
          corrupted.constructors.at(descriptor)[field] =
              tepl::core::semantics::make(
                  tepl::core::symbolic::Symbol::Kind::Integer,
                  "18446744073709551616");
          if (tepl::core::semantics::verify(plan, corrupted).status !=
              tepl::core::semantics::ProofStatus::Unknown)
            throw std::runtime_error(
                "out-of-range scalar constructor accepted");
        }
    auto compiled = tepl::core::compile(
        *parsed.program, {algebraic ? tepl::core::SemanticPolicy::kAlgebraic
                                    : tepl::core::SemanticPolicy::kStrict});
    if (!compiled.ok()) {
      for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
      throw std::runtime_error("proved scalar rule did not compile");
    }
    (void)tepl::executable::load(
        tepl::executable::serialize(*compiled.program));
  }
}
void run() {
  check("rule r {(D.min (D.min X Y) X)=>(D.min Y X)}", true);
  check("rule r {(D.max X (D.max Y X))=>(D.max X Y)}", true);
  check("rule r {(D.min X (D.max X Y))=>X}", true);
  check("rule r {(D.min X Y)=>X}", false);
  check("rule r {X:i32[] (D.min X 2147483647:i32)=>X}", true);
  check("rule r {X:i32[] (D.min (D.add X X) 2147483647:i32)=>(D.add X X)}",
        false);
  check("rule r {(D.exp X)=>(D.exp X)}", true, false);
  check("rule r {(c: D.compare X Y)=>(D.compare[c.attrs] X Y)}", true);
  check("rule r {(c: D.compare X Y)=>(D.compare[c.attrs] X Y)}", true);
  auto typed_attributes = dialect() + R"(
    dialect Typed {
      op enabled(x:tensor)->tensor {
        attrs {flag:bool;}
        dtype(t){yield t;}
        axes {assert attrs.flag; bind S{x:all_axes(x)}; yield S;}
        semantics {yield x;}
      }
      op annotated(x:tensor)->tensor {
        attrs {tag:dtype;}
        dtype(t){yield t;}
        axes {bind S{x:all_axes(x)}; yield S;}
        semantics {yield x;}
      }
    })";
  check("rule r {(a:Typed.enabled X)=>(Typed.enabled[a.attrs] X)}", true, false,
        typed_attributes);
  check("rule r {(a:Typed.annotated X)=>(Typed.annotated[a.attrs] X)}", true,
        false, typed_attributes);

  check(
      "rule r {(c: D.compare (D.max X Y) X)=>(D.compare[c.attrs] Y X) "
      "where {c.attrs.direction==3}}",
      true);
  check(
      "rule r {(c: D.compare X (D.min X Y))=>(D.compare[c.attrs] X Y) "
      "where {c.attrs.direction==3}}",
      true);
  check(
      "rule r {(c: D.compare (D.max X Y) X)=>(D.compare[c.attrs] X Y) "
      "where {c.attrs.direction==3}}",
      false);
  check(
      "rule r {(c: D.compare (D.max X Y) X)=>(D.compare[c.attrs] Y X) "
      "where {c.attrs.direction==0}}",
      false);
  auto renamed = dialect();
  for (auto [from, to] : {std::pair{"op compare(", "op relation("},
                          std::pair{"direction", "mode"}})
    for (std::size_t pos = 0;
         (pos = renamed.find(from, pos)) != std::string::npos;
         pos += std::string(to).size())
      renamed.replace(pos, std::string(from).size(), to);
  check(
      "rule r {(c: D.relation X (D.min X Y))=>(D.relation[c.attrs] X Y) "
      "where {c.attrs.mode==3}}",
      true, true, renamed);
  check(
      "rule r {(c: D.compare X Y)=>(d: D.compare X Y) derive "
      "{d.attrs={direction:c.attrs.direction};}}",
      true);
  check("rule r {(c: D.compare X Y)=>(D.compare[d.attrs] X Y)}", false);
  check("rule r {(D.min (D.exp X) (D.exp Y))=>(D.exp (D.min X Y))}", true);

  check("rule r {(D.exp (D.add X (D.zero X)))=>(D.exp X)}", true);
  check("rule r {(D.exp (D.add X Y))=>(D.exp (D.add Y X))}", true);
  check("rule r {(D.exp (D.add X Y))=>(D.exp (D.add X Y))}", true, false);
  check("rule r {(D.exp (D.add X Y))=>(D.exp (D.add Y X))}", false, false);
  check("rule r {(D.exp (D.add (D.add X Y) Z))=>(D.exp (D.add X (D.add Y Z)))}",
        false, false);
  check("rule r {(t: D.transpose (D.exp X))=>(D.exp (D.transpose[t.attrs] X))}",
        true);
  check("rule r {(t: D.transpose (D.exp X))=>(D.exp (D.transpose[t.attrs] X))}",
        true, false);
  check(
      "rule r {X:[M,N] (t: D.transpose (D.exp X))=>(D.exp X) where "
      "{t.attrs.p==[1,0]}}",
      false);
  check("rule r {(r: D.reverse (D.exp X))=>(D.exp (D.reverse[r.attrs] X))}",
        true);
  check("rule r {(t: D.transpose (D.exp X))=>(D.exp (D.transpose X))}", true);
  check("rule r {(D.exp X)=>(D.log X)}", false);
  check(
      "rule r {(c: D.compare X Y)=>(out: D.compare X Y) where "
      "{c.attrs.direction==0} derive {out.attrs={direction:1};}}",
      false);
  check("rule r {(s: D.sum (D.exp X))=>(D.exp (D.sum[s.attrs] X))}", false);
  check("rule r {(D.min X X)=>X}", false, false);
  check("rule r {X:[] (D.mul X 1)=>X}", true);
  check("rule r {X:[] (D.pow X 2)=>(D.mul X X)}", true);
  check("rule r {X:[] (D.pow X 2.0)=>(D.mul X X)}", true);
  check("rule r {X:[] (D.pow X 2)=>(D.pow X 2)}", true, false);
  check("rule r {X:[] (D.mul X 0.5)=>(D.mul X 0.500)}", true);
  check("rule r {X:[] (D.mul X 0.5)=>(D.mul X 0.25)}", false);

  check("rule r {(D.not (D.not X))=>X}", true);
  check("rule r {X:[] (D.pow X 0)=>1}", true);

  check("rule r {(D.log (D.exp X))=>X}", true);
  check("rule r {(D.sqrt (D.mul X X))=>(D.abs X)}", true);
  check("rule r {(D.mul (D.abs X) (D.abs X))=>(D.mul X X)}", true);
  check("rule r {(D.mul (D.exp X) (D.exp Y))=>(D.exp (D.add X Y))}", true);
  check("rule r {X:[] (D.log (D.sqrt (D.exp X)))=>(D.mul 0.5 X)}", true);
  check("rule r {X:[] (D.log (D.sqrt X))=>(D.mul 0.5 (D.log X))}", false);
  check("rule r {(D.div (D.div X Y) Z)=>(D.div X (D.mul Y Z))}", false);
  // Reciprocal multiplication is valid only for a known nonzero divisor in
  // the algebraic floating policy, not for integer or strict arithmetic.
  check("rule r {X:f32[] (D.div X 2:f32)=>(D.mul X (D.div 1:f32 2:f32))}",
        true);
  check("rule r {X:f32[] (D.div X 0.5:f32)=>(D.mul X (D.div 1:f32 0.5:f32))}",
        true);
  check("rule r {X:f32[] (D.div X 0:f32)=>(D.mul X (D.div 1:f32 0:f32))}",
        false);
  check("rule r {X:i32[] (D.div X 2:i32)=>(D.mul X (D.div 1:i32 2:i32))}",
        false);
  check("rule r {X:f32[] (D.div X 2:f32)=>(D.mul X (D.div 1:f32 2:f32))}",
        false, false);
  check("rule r {X:f32[] Y:f32[] (D.div X Y)=>(D.mul X (D.div 1:f32 Y))}",
        false);
  check("rule r {(D.exp X)=>(D.exp X)}", false, true,
        "dialect D {op exp(x:tensor)->tensor {dtype(a){assert "
        "is_float(a);yield a;} axes {bind S{x:all_axes(x)};yield S;} "
        "semantics {yield exp(x,x);}}}");
  check("rule r {(D.exp X)=>(D.exp X)}", false, true,
        "dialect D {op exp(x:tensor)->tensor {dtype(a){assert "
        "is_integer(a);yield a;} axes {bind S{x:all_axes(x)};yield S;} "
        "semantics {yield exp(x);}}}");
  check("rule r {(D.exp X)=>(D.exp X)}", false, true,
        "dialect D {op exp(x:tensor)->tensor {dtype(a){assert a==bool;yield "
        "bool;} "
        "axes {bind S{x:all_axes(x)};yield S;} "
        "semantics {yield compare(x,Const(2),0);}}}");
}
}  // namespace
int main() {
  try {
    run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
