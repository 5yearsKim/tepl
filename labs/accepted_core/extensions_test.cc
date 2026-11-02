#include "src/inference/extensions.h"

#include "labs/accepted_core/dense_reference.h"
using namespace tepl;
std::string definitions;
unsigned checks = 0, negative = 0, graph_id = 0;
std::string xs(const Shape& s) {
  std::string out = "[";
  for (auto n : s) {
    if (out.size() > 1) out += ",";
    out += std::to_string(n);
  }
  return out + "]";
}
std::string graph(const Shape& shape, const Shape& a, const Shape& b) {
  return "graph g" + std::to_string(graph_id++) + " { input X:f32" + xs(shape) +
         "; yield (reverse[dimensions=" + xs(a) +
         "] (reverse[dimensions=" + xs(b) + "] X)); }\n";
}
executable::ExecutableProgram compiled(const std::string& text) {
  auto parsed = parse(definitions + text, "extension_test.tepl");
  for (const auto& d : parsed.diagnostics) std::cerr << d.message << '\n';
  require(parsed.ok(), "extension parses");
  auto result = core::compile(*parsed.program,
                              {core::SemanticPolicy::kAlgebraic, false, true});
  for (const auto& d : result.diagnostics) std::cerr << d.message << '\n';
  require(result.ok(), "extension compiles");
  return std::move(*result.program);
}
void rejected(const std::string& text, const std::string& defs = {}) {
  auto parsed =
      parse((defs.empty() ? definitions : defs) + text, "negative.tepl");
  require(parsed.ok(), "negative parses");
  require(!core::compile(*parsed.program,
                         {core::SemanticPolicy::kAlgebraic, false, true})
               .ok(),
          "unsound or unsupported schema accepted");
  ++negative;
}
void invalid(Json j) {
  bool bad = false;
  try {
    executable::load(j.dump());
  } catch (const std::exception&) {
    bad = true;
  }
  require(bad, "edited certificate accepted");
  ++negative;
}
void test(const std::string& source, bool requires_guard = false) {
  std::cerr << "testing " << source.substr(0, 70) << "\n";
  auto p = compiled(source);
  auto wire = executable::serialize(p);
  auto loaded = executable::load(wire);
  require(wire == executable::serialize(loaded), "schema roundtrip");
  require(loaded.ir().rules[0].core &&
              inference::hasExtensions(loaded.ir().rules[0].core->program),
          "schema capability missing");
  for (const auto& g : loaded.ir().graphs) {
    auto before = reference(loaded.ir(), g);
    rewrite::Options opts;
    opts.max_steps = 1;
    auto r = rewrite::run(loaded, g, opts);
    require(r.rewrite_count == 1, "one rewrite: " + g.name + " " + r.detail);
    require(before == reference(loaded.ir(), r.graph),
            "dense values and ordered shape: " + g.name);
    ++checks;
  }
  auto j = Json::parse(wire), q = j;
  auto& c = q["program"]["rules"][0]["core"];
  c["nodes"][0][0]["capture"] = 999999;
  invalid(q);
  q = j;
  q["program"]["rules"][0]["core"]["attributes"][0][0][1] = 999999;
  invalid(q);
  q = j;
  q["program"]["rules"][0]["core"]["program"]["replacement"].back()["fields"] =
      Json::array({99});
  invalid(q);
  q = j;
  q["capabilities"].erase(q["capabilities"].end() - 1);
  invalid(q);
  q = j;
  q["policy"] = "strict";
  invalid(q);
  if (requires_guard) {
    q = j;
    q["program"]["rules"][0]["conditions"] = Json::array();
    invalid(q);
  }
}
int main(int argc, char** argv) {
  try {
    definitions = read(argv[1]);
    std::string reverse =
        "rule compose {(a:reverse (b:reverse X)) => (c:reverse X) derive "
        "{c.attrs={dimensions:auto};}}\n";
    std::string graphs;
    for (unsigned rank = 0; rank <= 4; ++rank)
      for (unsigned a = 0; a < (1u << rank); ++a)
        for (unsigned b = 0; b < (1u << rank); ++b) {
          Shape shape(rank), aa, bb;
          for (unsigned i = 0; i < rank; ++i) {
            shape[i] = (a + b + i) % 3;
            if (a & (1u << i)) aa.push_back(i);
            if (b & (1u << i)) bb.push_back(i);
          }
          std::reverse(aa.begin(), aa.end());
          graphs += graph(shape, aa, bb);
        }
    test(reverse + graphs);
    test(
        "rule cancel {(a:reverse (b:reverse X))=>X where "
        "{a.attrs.dimensions==b.attrs.dimensions}} graph g {input "
        "X:f32[2,3];yield(reverse[dimensions=[1,0]](reverse[dimensions=[1,0]]X)"
        ");}",
        true);
    for (auto op : {"add", "multiply"}) {
      std::string rule =
          "rule distribute {(" + std::string(op) +
          " (a:broadcast X) (b:broadcast Y)) => (c:broadcast (" + op +
          " X Y)) where {a.attrs.dimensions==b.attrs.dimensions && "
          "a.attrs.new_extents==b.attrs.new_extents} derive "
          "{c.attrs={dimensions:auto,new_extents:a.attrs.new_extents};}}";
      graphs.clear();
      for (unsigned rank = 0; rank <= 3; ++rank)
        for (unsigned extent = 0; extent <= 2; ++extent) {
          Shape shape(rank);
          for (unsigned i = 0; i < rank; ++i) shape[i] = (i + extent) % 3;
          Shape dims(rank);
          std::iota(dims.begin(), dims.end(), 1);
          std::reverse(dims.begin(), dims.end());
          graphs += "graph d" + std::to_string(rank) + std::to_string(extent) +
                    " {input X:f32" + xs(shape) + ";input Y:f32" + xs(shape) +
                    ";yield(" + op + "(broadcast[dimensions=" + xs(dims) +
                    ",new_extents=[" + std::to_string(extent) +
                    "]]X)(broadcast[dimensions=" + xs(dims) + ",new_extents=[" +
                    std::to_string(extent) + "]]Y));}";
        }
      test(rule + graphs, true);
      auto pos = rule.find(" && a.attrs.new_extents==b.attrs.new_extents");
      rule.erase(
          pos,
          std::string(" && a.attrs.new_extents==b.attrs.new_extents").size());
      rejected(rule);
    }
    test(
        "rule compose {X:[D...] (a:broadcast(b:broadcast X))=>(c:broadcast X) "
        "where {a.attrs.dimensions==range(len(D)+len(b.attrs.new_extents))} "
        "derive "
        "{c.attrs={dimensions:auto,new_extents:concat(b.attrs.new_extents,a."
        "attrs.new_extents)};}} graph g {input "
        "X:f32[2,3];yield(broadcast[dimensions=[0,1,2],new_extents=[2]]("
        "broadcast[dimensions=[2,0],new_extents=[1]]X));}",
        true);
    test(
        "rule push {(a:reverse(b:broadcast X))=>(c:broadcast(d:reverse X)) "
        "derive "
        "{d.attrs={dimensions:auto};c.attrs={dimensions:auto,new_extents:b."
        "attrs.new_extents};}} graph g {input "
        "X:f32[2,3];yield(reverse[dimensions=[0,1]](broadcast[dimensions=[2,0],"
        "new_extents=[2]]X));} graph z {input "
        "X:f32[0,3];yield(reverse[dimensions=[0,1]](broadcast[dimensions=[2,0],"
        "new_extents=[0]]X));}");
    test(
        "rule scalar {(a:reverse(b:broadcast X))=>(c:broadcast X) where "
        "{b.attrs.dimensions==[]} derive "
        "{c.attrs={dimensions:auto,new_extents:b.attrs.new_extents};}} graph g "
        "{input "
        "X:f32[];yield(reverse[dimensions=[0,1]](broadcast[dimensions=[],new_"
        "extents=[2,3]]X));}",
        true);
    test(
        "rule scalar {(a:transpose(b:broadcast X))=>(c:broadcast X) where "
        "{b.attrs.dimensions==[]} derive "
        "{c.attrs={dimensions:auto,new_extents:gather(b.attrs.new_extents,a."
        "attrs.permutation)};}} graph g {input "
        "X:f32[];yield(transpose[permutation=[2,0,1]](broadcast[dimensions=[],"
        "new_extents=[2,0,3]]X));}",
        false);
    test(
        "rule perm {(a:broadcast X)=>(b:transpose X) where "
        "{a.attrs.new_extents==[]} derive {b.attrs={permutation:auto};}} graph "
        "g {input "
        "X:f32[2,3,1];yield(broadcast[dimensions=[2,0,1],new_extents=[]]X);}",
        true);
    test(
        "rule reduce {(a:reduce_sum(b:broadcast X))=>(c:broadcast(d:reduce_sum "
        "X)) where {b.attrs.dimensions==[2,0] && a.attrs.dimensions==[0]} "
        "derive "
        "{d.attrs={dimensions:auto};c.attrs={dimensions:auto,new_extents:b."
        "attrs.new_extents};}} graph g {input "
        "X:f32[2,3];yield(reduce_sum[dimensions=[0]](broadcast[dimensions=[2,0]"
        ",new_extents=[2]]X));} graph z {input "
        "X:f32[2,0];yield(reduce_sum[dimensions=[0]](broadcast[dimensions=[2,0]"
        ",new_extents=[0]]X));}",
        true);
    test(
        "rule negate {(a:reduce_sum(negate X))=>(negate(b:reduce_sum X)) "
        "derive {b.attrs={dimensions:auto};}} graph g {input "
        "X:f32[2,0];yield(reduce_sum[dimensions=[1]](negate X));}");
    test(
        "rule move {(d:dot(r:reverse X)Y)=>(e:dot X(t:reverse Y)) where "
        "{r.attrs.dimensions==[2] && d.attrs.lhs_contracting_dimensions==[2]} "
        "derive "
        "{t.attrs={dimensions:auto};e.attrs={lhs_contracting_dimensions:auto,"
        "rhs_contracting_dimensions:auto,lhs_batching_dimensions:auto,rhs_"
        "batching_dimensions:auto};}} graph g {input X:f32[2,3,4];input "
        "Y:f32[2,4,5];yield(dot[lhs_contracting_dimensions=[2],rhs_contracting_"
        "dimensions=[1],lhs_batching_dimensions=[0],rhs_batching_dimensions=[0]"
        "](reverse[dimensions=[2]]X)Y);}",
        true);
    test(
        "rule full {X:[D...] Y:[D...] (d:dot X Y)=>(s:reduce_sum(multiply X "
        "Y)) where {d.attrs.lhs_batching_dimensions==[] && "
        "d.attrs.rhs_batching_dimensions==[] && "
        "d.attrs.lhs_contracting_dimensions==range(len(D)) && "
        "d.attrs.rhs_contracting_dimensions==range(len(D))} derive "
        "{s.attrs={dimensions:auto};}} graph g {input X:f32[2,3];input "
        "Y:f32[2,3];yield(dot[lhs_contracting_dimensions=[0,1],rhs_contracting_"
        "dimensions=[0,1],lhs_batching_dimensions=[],rhs_batching_dimensions=[]"
        "]X Y);}",
        true);
    test(
        "rule outer {X:[A...] Y:[B...] (d:dot X Y)=>(multiply(l:broadcast "
        "X)(r:broadcast Y)) where {d.attrs.lhs_batching_dimensions==[] && "
        "d.attrs.rhs_batching_dimensions==[] && "
        "d.attrs.lhs_contracting_dimensions==[] && "
        "d.attrs.rhs_contracting_dimensions==[]} derive "
        "{l.attrs={dimensions:auto,new_extents:B};r.attrs={dimensions:auto,new_"
        "extents:A};}} graph g {input X:f32[2,1];input "
        "Y:f32[3];yield(dot[lhs_contracting_dimensions=[],rhs_contracting_"
        "dimensions=[],lhs_batching_dimensions=[],rhs_batching_dimensions=[]]X "
        "Y);}",
        true);
    rejected(
        "rule bad {(a:reverse(b:broadcast X))=>(c:broadcast X) derive "
        "{c.attrs={dimensions:auto,new_extents:b.attrs.new_extents};}}");
    rejected("rule bad {(a:reverse(b:reverse X))=>X}");
    rejected(
        "rule bad {(a:broadcast X)=>(b:transpose X) derive "
        "{b.attrs={permutation:auto};}}");
    rejected(
        "rule bad {(add(a:broadcast X)(b:broadcast Y))=>(c:broadcast(multiply "
        "X Y)) where {a.attrs.dimensions==b.attrs.dimensions && "
        "a.attrs.new_extents==b.attrs.new_extents} derive "
        "{c.attrs={dimensions:auto,new_extents:a.attrs.new_extents};}}");
    // High rank and an empty selector are not finite-rank inference premises.
    test(reverse + graph(Shape(128, 1), {127, 0}, {64, 127}));
    auto reverse_program = compiled(reverse);
    auto emitted = inference::emit(reverse_program.ir().rules[0].core->program);
    inference::Metadata metadata;
    for (const auto& n : emitted.original.source) {
      if (n.capture) metadata.shapes[*n.capture] = {2, 3};
      if (n.descriptor) metadata.descriptors[*n.descriptor] = {{0, 1}};
    }
    require(inference::executeSound(emitted, metadata, {1, 128}).status ==
                inference::Status::ResourceLimit,
            "work budget");
    ++negative;
    for (auto& [id, fields] : metadata.descriptors) fields[0] = {0, 0};
    require(inference::executeSound(emitted, metadata).status ==
                inference::Status::InvalidSource,
            "duplicate selector");
    ++negative;
    test(
        "rule parity {(negate(negate X))=>X} graph g {input "
        "X:f32[2,3];yield(negate(negate X));}");
    auto altered = definitions;
    auto at = altered.find(
        "yield Read(input, IndexReverse(Coords(S), attrs.dimensions));");
    require(at != std::string::npos, "reverse semantic mutation fixture");
    altered.replace(
        at,
        std::string(
            "yield Read(input, IndexReverse(Coords(S), attrs.dimensions));")
            .size(),
        "yield input;");
    rejected(reverse, altered);
    altered = definitions;
    at = altered.find("op reverse(");
    at = altered.find("dtype(t) { yield t; }", at);
    require(at != std::string::npos, "reverse dtype mutation fixture");
    altered.replace(at, std::string("dtype(t) { yield t; }").size(),
                    "dtype(t) { yield f64; }");
    rejected(reverse, altered);
    rejected(
        "rule bad {(a:reverse(b:reverse X))=>(c:reverse X) derive "
        "{c.attrs={dimensions:[0]};}}");
    // Five additional schema families: the frozen inventory sources are
    // exercised separately by the CLI rerun; these are independent variants.
    const std::string dotattrs =
        "lhs_contracting_dimensions:auto,rhs_contracting_dimensions:auto,lhs_"
        "batching_dimensions:auto,rhs_batching_dimensions:auto";
    auto broadcastText = [](const Shape& dims, const Shape& ext,
                            const std::string& input) {
      return "(broadcast[dimensions=" + xs(dims) + ",new_extents=" + xs(ext) +
             "] " + input + ")";
    };
    auto inputText = [](const std::string& name, const Shape& shape) {
      return "input " + name + ":f32" + xs(shape) + ";";
    };
    std::string nested =
        "rule nested {Y:[D...] C:[] (multiply (multiply X(c:broadcast "
        "C))(y:broadcast Y)) => (multiply X(out:broadcast(multiply "
        "Y(small:broadcast C)))) where {c.attrs.dimensions==[]} derive "
        "{small.attrs={dimensions:auto,new_extents:D};out.attrs={dimensions:"
        "auto,new_extents:y.attrs.new_extents};}}";
    graphs.clear();
    for (unsigned rank = 0; rank <= 3; ++rank) {
      Shape shape(rank);
      for (unsigned i = 0; i < rank; ++i) shape[i] = i + 1;
      Shape order(rank + 1);
      std::iota(order.begin(), order.end(), 0);
      do {
        Shape dims(order.begin(), order.begin() + rank);
        for (unsigned e = 0; e <= 2; ++e) {
          Shape out(rank + 1, e);
          for (unsigned i = 0; i < rank; ++i) out[dims[i]] = shape[i];
          graphs += "graph n" + std::to_string(graph_id++) + " {" +
                    inputText("X", out) + inputText("Y", shape) +
                    inputText("C0", {}) + "yield(multiply(multiply X" +
                    broadcastText({}, out, "C0") + ")" +
                    broadcastText(dims, {e}, "Y") + ");}";
        }
      } while (std::next_permutation(order.begin(), order.end()));
    }
    test(nested + graphs, true);
    // Mixed original and created contraction axes, with batch dimensions.
    std::string moved =
        "rule move {X:[B,M,K] Y:[B,J,K,N] (d:dot(b:broadcast X)Y)=>(e:dot "
        "X(s:reduce_sum Y)) where {b.attrs.dimensions==[0,1,2] && "
        "b.attrs.new_extents==[J] && d.attrs.lhs_batching_dimensions==[0] && "
        "d.attrs.rhs_batching_dimensions==[0] && "
        "d.attrs.lhs_contracting_dimensions==[2,3] && "
        "d.attrs.rhs_contracting_dimensions==[2,1]} derive "
        "{s.attrs={dimensions:auto};e.attrs={" +
        dotattrs + "};}}";
    graphs.clear();
    for (unsigned j = 0; j <= 3; ++j) {
      graphs += "graph m" + std::to_string(j) + " {" +
                inputText("X", {2, 3, 2}) + inputText("Y", {2, j, 2, 4}) +
                "yield(dot[lhs_contracting_dimensions=[2,3],rhs_contracting_"
                "dimensions=[2,1],lhs_batching_dimensions=[0],rhs_batching_"
                "dimensions=[0]]" +
                broadcastText({0, 1, 2}, {j}, "X") + " Y);}";
    }
    test(moved + graphs, true);
    // A non-scalar factor is not eligible for scalar extraction.
    std::string scalar =
        "rule factor {X:[B,M,K] Y:[B,K,N] C:[] (d:dot(multiply(b:broadcast "
        "C)X)Y)=>(multiply(out:broadcast C)(e:dot X Y)) where "
        "{b.attrs.dimensions==[] && d.attrs.lhs_batching_dimensions==[0] && "
        "d.attrs.rhs_batching_dimensions==[0] && "
        "d.attrs.lhs_contracting_dimensions==[2] && "
        "d.attrs.rhs_contracting_dimensions==[1]} derive "
        "{out.attrs={dimensions:auto,new_extents:[B,M,N]};e.attrs={" +
        dotattrs + "};}}";
    graphs.clear();
    for (unsigned k = 0; k <= 3; ++k) {
      graphs += "graph s" + std::to_string(k) + " {" +
                inputText("X", {2, 3, k}) + inputText("Y", {2, k, 4}) +
                inputText("C0", {}) +
                "yield(dot[lhs_contracting_dimensions=[2],rhs_contracting_"
                "dimensions=[1],lhs_batching_dimensions=[0],rhs_batching_"
                "dimensions=[0]](multiply" +
                broadcastText({}, {2, 3, k}, "C0") + " X)Y);}";
    }
    test(scalar + graphs, true);
    std::string unit =
        "rule unit {X:[A,N] (b:broadcast X)=>(out:broadcast(u:remove_unit_axes "
        "X)) where {A==1 && b.attrs.dimensions==[0,1]} derive "
        "{u.attrs={dimensions:[0]};out.attrs={dimensions:auto,new_extents:"
        "concat([1],b.attrs.new_extents)};}}";
    graphs.clear();
    for (unsigned n = 0; n <= 3; ++n)
      for (unsigned e = 0; e <= 2; ++e) {
        graphs += "graph u" + std::to_string(n) + std::to_string(e) + " {" +
                  inputText("X", {1, n}) + "yield" +
                  broadcastText({0, 1}, {e}, "X") + ";}";
      }
    test(unit + graphs, true);
    const std::string permute = R"(
rule permute {X:[D...] (p:transpose(b:broadcast X))=>(out:broadcast X)
 derive {
 let n=len(p.attrs.permutation);
 let inv=[sum([if p.attrs.permutation[j]==i then j else 0 for j in range(n)]) for i in range(n)];
 let d=gather(inv,b.attrs.dimensions);
 let oldnew=exclude(range(n),b.attrs.dimensions);
 let allpos=concat(b.attrs.dimensions,oldnew);
 let allshape=concat(D,b.attrs.new_extents);
 let oldshape=[sum([if allpos[j]==i then allshape[j] else 0 for j in range(n)]) for i in range(n)];
 let newpos=exclude(range(n),d);
 out.attrs={dimensions:auto,new_extents:gather(gather(oldshape,p.attrs.permutation),newpos)};
 }}
)";
    graphs.clear();
    for (unsigned rank = 0; rank <= 2; ++rank) {
      Shape shape(rank);
      for (unsigned i = 0; i < rank; ++i) shape[i] = i + 1;
      std::vector<Shape> dims = {Shape(rank)};
      std::iota(dims[0].begin(), dims[0].end(), 0);
      if (rank) {
        auto other = dims[0];
        other[0] = rank + 1;
        std::reverse(other.begin(), other.end());
        dims.push_back(other);
      }
      for (const auto& embedding : dims) {
        Shape p(rank + 2);
        std::iota(p.begin(), p.end(), 0);
        do {
          for (unsigned e = 0; e <= 2; ++e)
            graphs += "graph p" + std::to_string(graph_id++) + " {" +
                      inputText("X", shape) +
                      "yield(transpose[permutation=" + xs(p) + "]" +
                      broadcastText(embedding, {e, 3}, "X") + ");}";
        } while (std::next_permutation(p.begin(), p.end()));
      }
    }
    test(permute + graphs);
    // The general helper must preserve both permutation bounds and values.
    auto alteredPermute = permute;
    auto badAt = alteredPermute.find("then j else 0");
    alteredPermute.replace(badAt, std::string("then j else 0").size(),
                           "then i else 0");
    rejected(alteredPermute);
    alteredPermute = permute;
    badAt = alteredPermute.find("let n=len(p.attrs.permutation);");
    alteredPermute.replace(
        badAt, std::string("let n=len(p.attrs.permutation);").size(),
        "let n=len(p.attrs.permutation)+1;");
    rejected(alteredPermute);
    auto badUnit = unit;
    badAt = badUnit.find("A==1");
    badUnit.replace(badAt, 4, "A==2");
    rejected(badUnit);
    auto badNested = nested;
    badAt = badNested.find("c.attrs.dimensions==[]");
    badNested.replace(badAt, std::string("c.attrs.dimensions==[]").size(),
                      "true");
    rejected(badNested);
    auto badMoved = moved;
    badAt = badMoved.find("lhs_contracting_dimensions==[2,3]");
    badMoved.replace(badAt,
                     std::string("lhs_contracting_dimensions==[2,3]").size(),
                     "lhs_contracting_dimensions==[2]");
    rejected(badMoved);
    auto badScalar = scalar;
    badAt = badScalar.find("b.attrs.dimensions==[]");
    badScalar.replace(badAt, std::string("b.attrs.dimensions==[]").size(),
                      "true");
    rejected(badScalar);
    alteredPermute = permute;
    badAt = alteredPermute.find("then allshape[j] else 0");
    alteredPermute.replace(badAt, std::string("then allshape[j] else 0").size(),
                           "then allshape[i] else 0");
    rejected(alteredPermute);
    alteredPermute = permute;
    badAt = alteredPermute.find("let n=");
    alteredPermute.insert(badAt, "let unused=gather([], [0]); ");
    rejected(alteredPermute);
    auto changedUnitDefinition = definitions;
    badAt = changedUnitDefinition.find("extent_sub(sizes, sizes), 1");
    require(badAt != std::string::npos, "unit declaration fixture");
    changedUnitDefinition.replace(
        badAt, std::string("extent_sub(sizes, sizes), 1").size(),
        "extent_sub(sizes, sizes), 2");
    rejected(unit, changedUnitDefinition);
    auto partialNested = nested;
    badAt = partialNested.find("new_extents:D");
    partialNested.replace(badAt, std::string("new_extents:D").size(),
                          "new_extents:concat(D,gather(gather([], [0]), []))");
    rejected(partialNested);
    alteredPermute = permute;
    badAt = alteredPermute.find("let n=");
    alteredPermute.insert(badAt, "let hidden=gather(gather([], [0]), []); ");
    rejected(alteredPermute);
    auto strict = parse(definitions + nested, "strict_core.tepl");
    require(strict.ok(), "strict core parses");
    require(!core::compile(*strict.program,
                           {core::SemanticPolicy::kAlgebraic, true, false})
                 .ok(),
            "canonical core admitted extension");
    ++negative;
    std::cout << "schema tests: " << checks << " dense/graph checks, "
              << negative << " rejection checks\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
