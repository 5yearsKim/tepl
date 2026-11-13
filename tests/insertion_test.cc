#include "src/inference/insertion.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/core/attr_synth/ordered/contract.h"
#include "src/core/pipeline.h"
#include "src/core/semantics/ir.h"
#include "src/executable/codec.h"
#include "src/inference/extensions.h"
#include "src/rewrite/engine.h"

namespace {
using namespace tepl::inference;
using tepl::core::semantics::Plan;
void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}
std::string read(const std::string& path) {
  std::ifstream in(path);
  expect(in.good(), "cannot read " + path);
  return {std::istreambuf_iterator<char>(in), {}};
}
std::string dialect;
Plan parse(const std::string& text, bool admitted = true) {
  auto parsed = tepl::parse(dialect + text, "insertion.tepl");
  if (!parsed.ok())
    for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  expect(parsed.ok(), "insertion fixture parses");
  tepl::core::InspectionOptions options{tepl::core::SemanticPolicy::kAlgebraic,
                                        true};
  options.core_backend = true;
  auto result = tepl::core::inspect(*parsed.program, options);
  if (!result.ok())
    for (const auto& d : result.diagnostics) std::cerr << d.message << '\n';
  expect(result.ok(), "insertion fixture prepares");
  auto plan = *result.program->rules[0].semantic_plan;
  if (admitted) {
    expect(plan.proof.core && plan.proof.core->restricted_insertion,
           "expected generic insertion path: " + plan.proof.reason);
    auto emission = emit(*plan.proof.core);
    expect(emission.insertion && emission.blocks.size() == 2,
           "compiled normalization and existing core blocks");
    auto compiled = tepl::core::compile(
        *parsed.program, {tepl::core::SemanticPolicy::kAlgebraic, true});
    if (!compiled.ok())
      for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
    expect(compiled.ok(), "generic insertion compiles to executable");
    const auto loaded =
        tepl::executable::load(tepl::executable::serialize(*compiled.program));
    expect(loaded.ir().rules[0].core == compiled.program->ir().rules[0].core,
           "portable round trip regenerates exactly the same two DAGs");
    expect(loaded.ir().rules[0].conditions.size() ==
               compiled.program->ir().rules[0].conditions.size(),
           "user guards survive materialization and serialization");
  }
  return plan;
}
std::size_t node(const Plan& p, const std::string& label) {
  for (std::size_t i = 0; i < p.nodes.size(); ++i)
    if (p.nodes[i].label == label) return i;
  throw std::runtime_error("unknown fixture label " + label);
}
void field(const Plan& p, Metadata& m, const std::string& label,
           std::vector<List> fields) {
  m.descriptors[p.nodes[node(p, label)].descriptor->value] = std::move(fields);
}
void dot(const Plan& p, Metadata& m, const std::string& label, List lb = {},
         List rb = {}, List lc = {}, List rc = {}) {
  const auto& code = p.proof.core->source;
  for (const auto& n : code)
    if (n.node == node(p, label)) {
      std::vector<List> fields(4);
      fields[n.fields[0]] = lb;
      fields[n.fields[1]] = rb;
      fields[n.fields[2]] = lc;
      fields[n.fields[3]] = rc;
      field(p, m, label, fields);
      return;
    }
  throw std::runtime_error("missing source dot");
}
void enumerate(const List& shape, const std::function<void(const List&)>& fn) {
  List at(shape.size());
  auto loop = [&](auto&& self, std::size_t d) -> void {
    if (d == shape.size()) {
      fn(at);
      return;
    }
    for (at[d] = 0; at[d] < shape[d]; ++at[d]) self(self, d + 1);
  };
  loop(loop, 0);
}
struct Tensor {
  List shape, values;
  std::int64_t at(const List& indices) const {
    expect(shape.size() == indices.size(), "oracle read rank");
    std::size_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      expect(indices[i] >= 0 && indices[i] < shape[i], "oracle read range");
      offset = offset * shape[i] + indices[i];
    }
    return values.at(offset);
  }
  bool operator==(const Tensor&) const = default;
};
// Independent physical-coordinate evaluator; no logical classes, normalizer,
// constructor, source replay, or rewrite schema is used by this oracle.
Tensor interpret(const std::vector<Instruction>& code, const Metadata& m,
                 const std::map<std::size_t, Tensor>& inputs) {
  std::vector<Tensor> values;
  for (const auto& n : code) {
    if (n.capture) {
      values.push_back(inputs.at(*n.capture));
      continue;
    }
    const auto& a = values[n.inputs[0]];
    const auto& fields = m.descriptors.at(*n.descriptor);
    const auto& f = fields[n.fields[0]];
    Tensor out;
    auto selected = [](const List& xs, std::size_t i) {
      return std::find(xs.begin(), xs.end(), i) != xs.end();
    };
    if (n.kind == Kind::Permutation) {
      for (auto axis : f) out.shape.push_back(a.shape.at(axis));
      enumerate(out.shape, [&](const List& at) {
        List old(at.size());
        for (std::size_t i = 0; i < f.size(); ++i) old.at(f[i]) = at[i];
        out.values.push_back(a.at(old));
      });
    } else if (n.kind == Kind::Broadcast) {
      const auto& extents = fields[n.fields[1]];
      expect(f.size() == a.shape.size(), "oracle insertion rank");
      out.shape.resize(f.size() + extents.size(), -1);
      for (std::size_t i = 0; i < f.size(); ++i) {
        expect(out.shape.at(f[i]) == -1, "oracle insertion injective");
        out.shape[f[i]] = a.shape[i];
      }
      std::size_t fresh = 0;
      for (auto& extent : out.shape)
        if (extent == -1) extent = extents.at(fresh++);
      enumerate(out.shape, [&](const List& at) {
        List old;
        for (auto axis : f) old.push_back(at.at(axis));
        out.values.push_back(a.at(old));
      });
    } else if (n.kind == Kind::Reverse) {
      out.shape = a.shape;
      enumerate(out.shape, [&](List at) {
        for (auto axis : f) at.at(axis) = a.shape.at(axis) - 1 - at.at(axis);
        out.values.push_back(a.at(at));
      });
    } else if (n.kind == Kind::Summation) {
      List reduced;
      for (std::size_t i = 0; i < a.shape.size(); ++i)
        (selected(f, i) ? reduced : out.shape).push_back(a.shape[i]);
      enumerate(out.shape, [&](const List& at) {
        std::int64_t total = 0;
        enumerate(reduced, [&](const List& k) {
          List old;
          std::size_t j = 0, r = 0;
          for (std::size_t i = 0; i < a.shape.size(); ++i)
            old.push_back(selected(f, i) ? k[r++] : at[j++]);
          total += a.at(old);
        });
        out.values.push_back(total);
      });
    } else {
      expect(n.kind == Kind::Contraction,
             "oracle supports the restricted fragment");
      const auto& b = values[n.inputs[1]];
      const auto& rb = fields[n.fields[1]];
      const auto& lc = fields[n.fields[2]];
      const auto& rc = fields[n.fields[3]];
      List lf, rf, reduced;
      for (std::size_t i = 0; i < f.size(); ++i) {
        expect(a.shape.at(f[i]) == b.shape.at(rb.at(i)),
               "oracle batch extents");
        out.shape.push_back(a.shape[f[i]]);
      }
      for (std::size_t i = 0; i < lc.size(); ++i) {
        expect(a.shape.at(lc[i]) == b.shape.at(rc.at(i)),
               "oracle contraction extents");
        reduced.push_back(a.shape[lc[i]]);
      }
      for (std::size_t i = 0; i < a.shape.size(); ++i)
        if (!selected(f, i) && !selected(lc, i)) {
          lf.push_back(i);
          out.shape.push_back(a.shape[i]);
        }
      for (std::size_t i = 0; i < b.shape.size(); ++i)
        if (!selected(rb, i) && !selected(rc, i)) {
          rf.push_back(i);
          out.shape.push_back(b.shape[i]);
        }
      enumerate(out.shape, [&](const List& at) {
        List left(a.shape.size()), right(b.shape.size());
        std::size_t j = 0;
        for (std::size_t i = 0; i < f.size(); ++i)
          left[f[i]] = right[rb[i]] = at[j++];
        for (auto i : lf) left[i] = at[j++];
        for (auto i : rf) right[i] = at[j++];
        std::int64_t total = 0;
        enumerate(reduced, [&](const List& k) {
          for (std::size_t i = 0; i < lc.size(); ++i)
            left[lc[i]] = right[rc[i]] = k[i];
          total += a.at(left) * b.at(right);
        });
        out.values.push_back(total);
      });
    }
    values.push_back(std::move(out));
  }
  return values.back();
}
std::size_t coefficient_checks = 0, cases = 0;
Result verify(const Plan& p, const Metadata& m,
              Status wanted = Status::Solved) {
  const auto& program = *p.proof.core;
  const auto emission = emit(program);
  auto result = execute(emission, m);
  expect(result.status == wanted,
         "unexpected result after " + std::to_string(cases) +
             " solved cases: " + std::to_string(int(result.status)) + ": " +
             result.reason);
  const auto decision = predicate(emission, m);
  expect(decision.status ==
                 (wanted == Status::Solved ? Status::Feasible : wanted) &&
             decision.attributes.empty(),
         "predicate does not execute attribute builders");
  if (wanted != Status::Solved) {
    expect(result.attributes.empty(), "failed result has no partial witness");
    return result;
  }
  Metadata completed = m;
  for (const auto& [id, attrs] : result.attributes)
    completed.descriptors[id] = attrs;
  std::size_t original_descriptors = 0;
  for (const auto& n : program.replacement)
    if (n.descriptor) {
      ++original_descriptors;
      expect(result.attributes.contains(*n.descriptor),
             "original RHS descriptor retained");
    }
  expect(result.attributes.size() == original_descriptors,
         "auxiliary transpose absorbed; no generated nodes escape");
  std::vector<std::size_t> captures;
  std::map<std::size_t, Tensor> inputs;
  for (const auto& [id, shape] : m.shapes) {
    captures.push_back(id);
    auto& t = inputs[id];
    t.shape.assign(shape.begin(), shape.end());
    std::size_t size = 1;
    for (auto d : shape) size *= d;
    t.values.resize(size);
  }
  // Exhaust all multilinear basis coefficients for these bounded metadata.
  auto basis = [&](auto&& self, std::size_t i) -> void {
    if (i == captures.size()) {
      expect(interpret(program.source, m, inputs) ==
                 interpret(program.replacement, completed, inputs),
             "constructed original RHS preserves ordered shape and every "
             "tested coefficient");
      ++coefficient_checks;
      return;
    }
    auto& values = inputs.at(captures[i]).values;
    for (auto& value : values) {
      value = 1;
      self(self, i + 1);
      value = 0;
    }
  };
  basis(basis, 0);
  ++cases;
  expect(execute(emission, m, {0}).status == Status::ResourceLimit,
         "bounded work is respected");
  return result;
}
void recorded(const std::string& directory) {
  for (const auto& name :
       {"scalar_broadcast_reverse", "scalar_broadcast_transpose",
        "broadcast_compose_append", "catalog_broadcast_transpose",
        "catalog_broadcast_inserted_unit_dims",
        "catalog_broadcast_transpose_broadcast", "catalog_dot_move_broadcast",
        "catalog_reduction_broadcast_original_axes"}) {
    auto p = parse(read(directory + "/" + name + ".tepl"));
    Metadata m;
    if (std::string(name) == "scalar_broadcast_reverse") {
      m.shapes = {{0, {}}};
      field(p, m, "attr_b", {{}, {2, 3}});
      field(p, m, "attr_r", {{0, 1}});
    } else if (std::string(name) == "scalar_broadcast_transpose") {
      m.shapes = {{0, {}}};
      field(p, m, "attr_b", {{}, {2, 3}});
      field(p, m, "attr_p", {{1, 0}});
    } else if (std::string(name) == "broadcast_compose_append") {
      m.shapes = {{0, {2, 3}}};
      field(p, m, "attr_inner", {{0, 1}, {4}});
      field(p, m, "attr_outer", {{0, 1, 2}, {1, 2}});
    } else if (std::string(name) == "catalog_broadcast_transpose") {
      m.shapes = {{0, {2, 3}}};
      field(p, m, "attr_b", {{1, 0}, {}});
    } else if (std::string(name) == "catalog_broadcast_inserted_unit_dims") {
      m.shapes = {{0, {3}}};
      field(p, m, "axis_eval_0", {{1}, {1}});
      field(p, m, "attr_b", {{0, 1}, {2, 3}});
    } else if (std::string(name) == "catalog_broadcast_transpose_broadcast") {
      m.shapes = {{0, {2, 3}}};
      field(p, m, "b", {{2, 0}, {4, 1}});
      field(p, m, "p", {{3, 2, 0, 1}});
    } else if (std::string(name) == "catalog_dot_move_broadcast") {
      m.shapes = {{0, {2}}, {1, {3, 2}}};
      field(p, m, "attr_b", {{0}, {3}});
      dot(p, m, "attr_d", {}, {}, {1}, {0});
    } else {
      m.shapes = {{0, {2}}};
      field(p, m, "attr_b", {{0}, {3}});
      field(p, m, "attr_r", {{0}});
    }
    verify(p, m);
    std::cout << name
              << ": generic reduction + exact bounded coefficient check\n";
  }
}
void boundaries() {
  auto p = parse(R"(rule r {(b:broadcast X)=>(o:broadcast X)
    derive {o.attrs={dimensions:auto,new_extents:[1]} }})");
  Metadata m{{{0, {2, 3}}}, {}};
  field(p, m, "b", {{2, 0}, {1}});
  verify(p, m);
  field(p, m, "b", {{2, 0}, {2}});
  verify(p, m, Status::Impossible);
  field(p, m, "b", {{0, 1}, {0}});
  verify(p, m, Status::OutsideDomain);
  field(p, m, "b", {{0, 1}, {1}});
  m.shapes[0] = {1, 3};
  verify(p, m, Status::OutsideDomain);
  m.shapes[0] = {0, 3};
  verify(p, m, Status::OutsideDomain);
  m.shapes[0] = {};
  field(p, m, "b", {{}, {1}});
  verify(p, m);
  // Reuse the same finite DAG at a rank far beyond the bounded oracle cases.
  auto reusable = emit(*p.proof.core);
  m.shapes[0] = std::vector<std::uint64_t>(128, 2);
  List embedding(128);
  std::iota(embedding.rbegin(), embedding.rend(), 1);
  field(p, m, "b", {embedding, {1}});
  auto high = execute(reusable, m);
  expect(
      high.status == Status::Solved &&
          high.attributes.at(p.nodes[p.rhs].descriptor->value)[0] == embedding,
      "rank-128 placement uses the same compiled DAG");
  auto bare = parse("rule r {(b:broadcast X)=>(p:transpose X)}");
  Metadata bm{{{0, {2, 3}}}, {}};
  field(bare, bm, "b", {{2, 0}, {1}});
  verify(bare, bm, Status::Impossible);
  field(bare, bm, "b", {{1, 0}, {}});
  verify(bare, bm);
  auto unwrapped = parse(R"(rule r {(p:transpose X)=>(b:broadcast X)
    derive {b.attrs={dimensions:auto,new_extents:[]} }})");
  Metadata plain{{{0, {2, 3, 4}}}, {}};
  field(unwrapped, plain, "p", {{2, 0, 1}});
  verify(unwrapped, plain);
  auto inputRhs = parse("rule r {(b:broadcast X)=>X}");
  Metadata input{{{0, {2, 3}}}, {}};
  field(inputRhs, input, "b", {{0, 1}, {}});
  verify(inputRhs, input);
  field(inputRhs, input, "b", {{1, 0}, {}});
  verify(inputRhs, input, Status::Impossible);
  auto internal =
      parse(R"(rule r {(b:broadcast X)=>(p:transpose (o:broadcast X))
    derive {o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})",
            false);
  auto admission =
      tepl::core::attr_synth::ordered::admit(internal, Mode::Synthesize, true);
  expect(admission.admitted() && !admission.program.restricted_insertion &&
             !insertionCandidate(admission.program),
         "internal RHS insertion is outside generic boundary");
  auto sourceSum = parse(
      R"(rule r {(s:reduce_sum (b:broadcast X))=>(o:broadcast (t:reduce_sum X))
    derive {o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})");
  Metadata sm{{{0, {2, 3}}}, {}};
  field(sourceSum, sm, "b", {{0, 2}, {2}});
  field(sourceSum, sm, "s", {{1}});
  verify(sourceSum, sm, Status::OutsideDomain);
  auto elim = parse(
      R"(rule r {(d:dot (b:broadcast X) Y)=>(q:dot X (s:reduce_sum Y))})");
  Metadata dm{{{0, {2, 3}}, {1, {3, 2}}}, {}};
  field(elim, dm, "b", {{2, 0}, {3}});
  dot(elim, dm, "d", {}, {}, {1}, {0});
  verify(elim, dm, Status::OutsideDomain);  // mapped free axes change order
  field(elim, dm, "b", {{0, 1}, {3}});
  dot(elim, dm, "d");
  verify(elim, dm, Status::OutsideDomain);  // created axis not contracted
  dot(elim, dm, "d", {}, {}, {2}, {0});
  dm.shapes[1] = {4, 2};
  verify(elim, dm, Status::InvalidSource);  // mismatched paired extents
  auto supplied =
      parse(R"(rule r {(b:broadcast X)=>(o:broadcast (p:transpose X))
    derive {p.attrs={permutation:[0]} o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})",
            false);
  auto restricted =
      tepl::core::attr_synth::ordered::admit(supplied, Mode::Synthesize, true);
  expect(!restricted.admitted() || !restricted.program.restricted_insertion,
         "supplied inner field cannot become an unknown");
  auto undefined = parse(R"(rule r {(b:broadcast X)=>(o:broadcast X)
    derive {let unused=gather(b.attrs.new_extents,[7])
      o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})");
  Metadata um{{{0, {2}}}, {}};
  field(undefined, um, "b", {{0}, {1}});
  auto invalid = execute(emit(*undefined.proof.core), um);
  expect(invalid.status != Status::Solved && invalid.attributes.empty(),
         "unused supplied expression definedness retained");
}
void runtimeGuards() {
  const std::string rule = R"(rule r {
    (r:reverse (b:broadcast X))=>(o:broadcast X)
    where b.attrs.new_extents==[1] && r.attrs.dimensions==[1]
    derive {o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }
  })";
  auto parsed = tepl::parse(dialect + rule + R"(
    graph yes {input X:f64[2] yield (reverse[dimensions=[1]] (broadcast[dimensions=[0],new_extents=[1]] X)) }
    graph no {input X:f64[2] yield (reverse[dimensions=[]] (broadcast[dimensions=[0],new_extents=[1]] X)) }
    graph outside {input X:f64[1] yield (reverse[dimensions=[1]] (broadcast[dimensions=[0],new_extents=[1]] X)) }
  )",
                            "runtime_insertion.tepl");
  expect(parsed.ok(), "runtime fixture parses");
  auto compiled = tepl::core::compile(
      *parsed.program, {tepl::core::SemanticPolicy::kAlgebraic, true});
  if (!compiled.ok())
    for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
  expect(compiled.ok(), "runtime fixture compiles");
  auto loaded =
      tepl::executable::load(tepl::executable::serialize(*compiled.program));
  for (std::size_t i = 0; i < 3; ++i) {
    auto rewritten = tepl::rewrite::run(loaded, loaded.ir().graphs[i]);
    expect(
        rewritten.rewrite_count == (i == 0 ? 1u : 0u),
        "runtime preserves source guard and skips outside the theorem domain");
  }
}
void general() {
  // Unsorted embeddings require a fresh output permutation; deletion of the
  // broadcast would incorrectly reject this reversed operand order.
  auto p = parse(R"(rule r {(b:broadcast (d:dot A B))=>(o:broadcast (q:dot B A))
    derive {o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})");
  Metadata m{{{0, {2}}, {1, {3}}}, {}};
  field(p, m, "b", {{2, 0}, {1}});
  dot(p, m, "d");
  auto r = verify(p, m);
  expect(r.attributes.at(p.nodes[p.rhs].descriptor->value)[0] == List({0, 2}),
         "auxiliary transpose absorbed into embedding");
  auto nested = parse(R"(rule r {(v:broadcast (u:broadcast X))=>(o:broadcast X)
    derive {o.attrs={dimensions:auto,new_extents:[3,2,1]} }})");
  Metadata nm{{{0, {2, 3}}}, {}};
  field(nested, nm, "u", {{2, 0}, {2}});
  field(nested, nm, "v", {{4, 1, 2}, {3, 1}});
  verify(nested, nm);
  auto signedRule = parse(
      R"(rule r {(r:reverse (b:broadcast (d:dot A B)))=>(o:broadcast (v:reverse (q:dot B A)))
    derive {o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})");
  Metadata rm{{{0, {2}}, {1, {3}}}, {}};
  dot(signedRule, rm, "d");
  field(signedRule, rm, "b", {{2, 0}, {1}});
  for (List axes : {List{}, List{0}, List{1}, List{2}, List{0, 1, 2}}) {
    field(signedRule, rm, "r", {axes});
    verify(signedRule, rm);
  }
  auto impossible =
      parse(R"(rule r {(r:reverse (b:broadcast X))=>(o:broadcast X)
    derive {o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})");
  Metadata im{{{0, {2}}}, {}};
  field(impossible, im, "b", {{0}, {1}});
  field(impossible, im, "r", {{0}});
  verify(impossible, im, Status::Impossible);
  field(impossible, im, "r", {{1}});
  verify(impossible, im);
  auto contraction = parse(
      R"(rule r {(b:broadcast (d:dot (r:reverse A) B))=>(o:broadcast (q:dot A (v:reverse B)))
    derive {o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})");
  Metadata cm{{{0, {2, 3}}, {1, {3, 2}}}, {}};
  field(contraction, cm, "r", {{1}});
  dot(contraction, cm, "d", {}, {}, {1}, {0});
  field(contraction, cm, "b", {{2, 0}, {1}});
  verify(contraction, cm);
  // No operation-name dispatch: identical place semantics under an arbitrary
  // name.
  auto start = dialect.find("op broadcast_in_dim");
  expect(start != std::string::npos, "fixture insertion definition");
  auto saved = dialect;
  dialect.replace(start, std::string("op broadcast_in_dim").size(),
                  "op inflate_axes");
  auto alias = dialect.find("alias: broadcast", start);
  dialect.replace(alias, std::string("alias: broadcast").size(),
                  "alias: inflate");
  auto renamed = parse(R"(rule r {(b:inflate X)=>(o:inflate X)
    derive {o.attrs={dimensions:auto,new_extents:b.attrs.new_extents} }})");
  Metadata am{{{0, {2, 3}}}, {}};
  field(renamed, am, "b", {{1, 2}, {1}});
  verify(renamed, am);
  dialect = saved;
}
// Reuse the independent dense oracle for signed programs without insertion.
void signedCore() {
  auto prepare = [](const std::string& rule) {
    auto p = parse(rule, false);
    expect(p.proof.core && !p.proof.core->restricted_insertion,
           "pure signed core admission");
    const auto e = emit(*p.proof.core);
    expect(!e.insertion && e.blocks.size() == 1 &&
               !usesSpecializedSchema(*p.proof.core),
           "signed programs dispatch to generic construction");
    const auto& expressions = e.blocks[0].expressions;
    expect(std::any_of(expressions.begin(), expressions.end(), [](auto& x) {
             return x.op == Op::OrientationCheck;
           }), "signed core emits orientation feasibility");
    expect(validatePortable(portable(e)), "signed DAG portable validation");
    return p;
  };
  auto chain = prepare(
      "rule r {(outer:reverse (inner:reverse X))=>(out:reverse X)}");
  for (unsigned rank = 0; rank <= 3; ++rank) {
    Metadata m{{{0, std::vector<std::uint64_t>(rank, 2)}}, {}};
    for (unsigned a = 0; a < (1u << rank); ++a)
      for (unsigned b = 0; b < (1u << rank); ++b) {
        List left, right, expected;
        for (unsigned i = 0; i < rank; ++i) {
          if (a & (1u << i)) left.push_back(i);
          if (b & (1u << i)) right.push_back(i);
          if ((a ^ b) & (1u << i)) expected.push_back(i);
        }
        field(chain, m, "inner", {left});
        field(chain, m, "outer", {right});
        auto r = verify(chain, m);
        expect(r.attributes.at(chain.nodes[node(chain, "out")].descriptor->value)[0]
                   == expected, "generic reversal chain computes parity");
      }
  }
  auto transfer = prepare(
      "rule r {(d:dot (r:reverse A) B)=>(q:dot A (v:reverse B))}");
  for (unsigned extent : {2u, 3u}) {
    Metadata m{{{0, {extent, extent}}, {1, {extent, extent}}}, {}};
    dot(transfer, m, "d", {}, {}, {1}, {0});
    for (List axes : {List{}, List{1}}) {
      field(transfer, m, "r", {axes});
      verify(transfer, m);
    }
    field(transfer, m, "r", {{0}});
    verify(transfer, m, Status::Impossible);
  }
  auto eliminated = prepare(
      "rule r {(s:reduce_sum (r:reverse X))=>(t:reduce_sum X)}");
  Metadata m{{{0, {2, 3}}}, {}};
  field(eliminated, m, "s", {{0}});
  field(eliminated, m, "r", {{0}});
  verify(eliminated, m);
  field(eliminated, m, "r", {{1}});
  verify(eliminated, m, Status::Impossible);
}

}  // namespace
int main(int argc, char** argv) {
  try {
    auto runfiles = std::unique_ptr<rules_cc::cc::runfiles::Runfiles>(
        rules_cc::cc::runfiles::Runfiles::Create(argv[0]));
    expect(argc == 3 && bool(runfiles), "fixture arguments");
    dialect = read(runfiles->Rlocation(argv[1]));
    auto fixture = runfiles->Rlocation(argv[2]);
    recorded(fixture.substr(0, fixture.rfind('/')));
    signedCore();
    boundaries();
    general();
    runtimeGuards();
    std::cout << cases << " solved metadata, " << coefficient_checks
              << " exact multilinear basis checks\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
