#include "src/core/inference/constraints.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

#include "src/core/inference/constraint_normalize.h"

namespace tepl::core::inference {
namespace {
struct Ref {
  std::size_t id;
  bool coordinate;
};
using Environment = std::map<std::string, Ref>;
using Terms = std::vector<ConstraintTerm>;
struct Compiler {
  const SemanticPlan& source;
  ConstraintPlan out;
  std::map<std::pair<std::string, std::vector<std::size_t>>, std::size_t>
      intern;
  std::size_t work = 0, depth = 0;
  struct Depth {
    Compiler& compiler;
    explicit Depth(Compiler& c) : compiler(c) {
      if (++c.work > 20000 || ++c.depth > 256)
        throw std::runtime_error("constraint expansion limit");
    }
    ~Depth() { --compiler.depth; }
  };
  [[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }
  Ref emit(std::string op, std::vector<std::size_t> args = {},
           bool coordinate = false) {
    auto key = std::pair(op, args);
    if (auto i = intern.find(key); i != intern.end())
      return {i->second, coordinate};
    if (out.expressions.size() >= 10000) fail("constraint expression limit");
    auto id = out.expressions.size();
    out.expressions.push_back({std::move(op), std::move(args)});
    intern.emplace(std::move(key), id);
    return {id, coordinate};
  }
  static std::int64_t coefficient(__int128 x) {
    if (x < std::numeric_limits<std::int64_t>::min() ||
        x > std::numeric_limits<std::int64_t>::max())
      throw std::runtime_error("constraint coefficient limit");
    return static_cast<std::int64_t>(x);
  }
  Ref index(std::size_t node, std::size_t id, const Environment& env) {
    Depth guard(*this);
    const auto& n = source.nodes.at(node);
    const auto& e = n.expressions.at(id);
    if (e.args.empty()) {
      if (auto i = env.find(e.op); i != env.end()) return i->second;
      if (e.op.starts_with('a') && e.op.size() > 1)
        return emit("@attr:" + std::to_string(node) + ":" + e.op.substr(1));
      if (e.op == "list" || e.op == "true" || e.op == "false" ||
          (!e.op.empty() &&
           e.op.find_first_not_of("0123456789") == std::string::npos))
        return emit(e.op);
      fail("unbound coordinate expression: " + e.op);
    }
    if (e.op == "shape" || e.op == "rank") {
      auto operand = n.expressions.at(e.args.at(0)).op;
      if (!operand.starts_with('t'))
        fail("tensor metadata requires an operand");
      auto child = n.inputs.at(std::stoul(operand.substr(1)));
      auto shape = emit("@shape:" + std::to_string(child));
      return e.op == "shape" ? shape : emit("len", {shape.id});
    }
    std::vector<Ref> args;
    std::vector<std::size_t> ids;
    for (auto a : e.args) {
      args.push_back(index(node, a, env));
      ids.push_back(args.back().id);
    }
    bool coord = std::any_of(args.begin(), args.end(),
                             [](auto a) { return a.coordinate; });
    if (coord) {
      // Labels represent coordinate identities, never numeric coordinates.
      // Permit routing only, so equality of labels proves equality of accesses.
      bool routing = false;
      if (e.op == "concat" || e.op == "list") {
        routing = std::all_of(args.begin(), args.end(), [&](auto a) {
          return a.coordinate || (out.expressions[a.id].op == "list" &&
                                  out.expressions[a.id].args.empty());
        });
      } else if (e.op == "gather" || e.op == "index" || e.op == "take" ||
                 e.op == "slice") {
        routing = args[0].coordinate;
        for (std::size_t j = 1; j < args.size(); ++j)
          routing &= !args[j].coordinate;
      } else if (e.op == "scatter") {
        routing =
            !args[0].coordinate && !args[1].coordinate && args[2].coordinate;
      }
      if (!routing)
        fail("coordinate arithmetic outside routing fragment: " + e.op);
    }
    return emit(e.op, std::move(ids), coord);
  }
  Terms expand(std::size_t node, Ref output) {
    Depth guard(*this);
    const auto& n = source.nodes.at(node);
    if (n.capture >= 0)
      return {{1, {}, {{static_cast<std::size_t>(n.capture), output.id}}}};
    return scalar(node, n.root, {{"o", output}});
  }
  Terms scalar(std::size_t node, std::size_t id, const Environment& env) {
    Depth guard(*this);
    const auto& n = source.nodes.at(node);
    const auto& e = n.expressions.at(id);
    if (e.op == "read") {
      auto operand = n.expressions.at(e.args.at(0)).op;
      if (!operand.starts_with('t')) fail("read requires an operand");
      auto at = index(node, e.args.at(1), env);
      if (!at.coordinate)
        fail("constant input indexing outside routing fragment");
      return expand(n.inputs.at(std::stoul(operand.substr(1))), at);
    }
    if (e.op.starts_with("sum:")) {
      auto domain = index(node, e.args.at(0), env);
      if (domain.coordinate) fail("nonrectangular reduction domain");
      auto binder = out.domains.size();
      out.domains.push_back(domain.id);
      auto inner = env;
      inner.insert_or_assign(
          e.op.substr(4), emit("@bound:" + std::to_string(binder), {}, true));
      auto terms = scalar(node, e.args.at(1), inner);
      for (auto& t : terms) t.reductions.push_back(binder);
      return terms;
    }
    if (e.args.empty() && !e.op.empty() &&
        e.op.find_first_not_of("0123456789") == std::string::npos)
      return {{std::stoll(e.op), {}, {}}};
    if (e.op != "+" && e.op != "-" && e.op != "*" && e.op != "unary:+" &&
        e.op != "unary:-")
      fail("scalar expression outside polynomial fragment: " + e.op);
    auto a = scalar(node, e.args.at(0), env);
    if (e.op == "unary:+") return a;
    if (e.op == "unary:-") {
      for (auto& t : a) t.coefficient = coefficient(-__int128(t.coefficient));
      return a;
    }
    auto b = scalar(node, e.args.at(1), env);
    if (e.op == "+" || e.op == "-") {
      for (auto& t : b) {
        if (e.op == "-") t.coefficient = coefficient(-__int128(t.coefficient));
        a.push_back(std::move(t));
      }
      if (a.size() > 64) fail("additive term limit");
      return a;
    }
    if (a.size() * b.size() > 64) fail("product expansion limit");
    Terms product;
    for (auto& x : a)
      for (auto& y : b) {
        auto t = x;
        t.coefficient = coefficient(__int128(x.coefficient) * y.coefficient);
        t.reductions.insert(t.reductions.end(), y.reductions.begin(),
                            y.reductions.end());
        t.reads.insert(t.reads.end(), y.reads.begin(), y.reads.end());
        product.push_back(std::move(t));
      }
    return product;
  }
  ConstraintPlan run() {
    if (!source.algebraic) fail("strict policy uses trace verification");
    auto output = emit("@output", {}, true);
    out.lhs = expand(source.source_root, output);
    out.rhs = expand(source.rhs_root, output);
    normalizeConstraints(out);
    out.available = true;
    out.reason =
        "sufficient routing, factor-occurrence, and reduction-bijection "
        "constraints";
    return std::move(out);
  }
};
}  // namespace
ConstraintPlan compileConstraints(const SemanticPlan& p) {
  try {
    return Compiler{p}.run();
  } catch (const std::exception& e) {
    ConstraintPlan out;
    out.reason = e.what();
    return out;
  }
}
}  // namespace tepl::core::inference
