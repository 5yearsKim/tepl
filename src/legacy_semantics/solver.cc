#include "src/legacy_semantics/solver.h"

#include <z3++.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "src/legacy_semantics/axis_domains.h"
#include "src/legacy_semantics/constraint_match.h"
#include "src/legacy_semantics/early_verification.h"
#include "src/legacy_semantics/ordered.h"

namespace tepl::core::semantics {
namespace {
using Node = inference::SemanticNode;
struct Limit : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Illegal {};
struct RetryUnrestricted {};
struct Timer {
  double* destination;
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  explicit Timer(double& out) : destination(&out) {}
  void stop() {
    if (destination) {
      *destination += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      destination = nullptr;
    }
  }
  ~Timer() { stop(); }
};
struct Vector {
  z3::expr length;
  std::vector<z3::expr> entries;
};
struct Value {
  z3::expr scalar;
  std::optional<Vector> vector;
  std::optional<std::size_t> upper;
  Value(z3::expr e, std::optional<std::size_t> cap = {})
      : scalar(e), upper(cap) {}
  Value(Vector v) : scalar(v.length), vector(v), upper(v.entries.size()) {}
};
struct Engine {
  const inference::SemanticPlan& plan;
  const std::vector<List>& captures;
  const std::vector<List>& known;
  const std::vector<List>& pinned;
  z3::context ctx;
  z3::solver solver{ctx};
  std::size_t budget;
  const SolveOptions& options;
  SolveStatistics& statistics;
  const AxisPropagation* propagated;
  std::size_t symbolic_axes = 0;
  std::chrono::steady_clock::time_point deadline;
  std::vector<Vector> shapes;
  std::vector<std::vector<Vector>> attrs;
  std::map<std::size_t, Vector> variables;
  std::map<unsigned, List> index_domains;
  bool generating = true;
  Engine(const inference::SemanticPlan& p, const std::vector<List>& s,
         const std::vector<List>& k, const std::vector<List>& pins,
         std::size_t b, const SolveOptions& opts, SolveStatistics& stats,
         const AxisPropagation* domains = nullptr)
      : plan(p),
        captures(s),
        known(k),
        pinned(pins),
        budget(b),
        options(opts),
        statistics(stats),
        propagated(domains),
        deadline(std::chrono::steady_clock::now() +
                 std::chrono::milliseconds(p.timeout_ms)) {
    solver.set("timeout", p.timeout_ms);
    solver.set("random_seed", options.random_seed);
  }
  void tick() {
    if (!budget || std::chrono::steady_clock::now() >= deadline)
      throw Limit("resource limit");
    --budget;
  }
  z3::check_result checkSolver() {
    tick();
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now())
                          .count();
    solver.set("timeout",
               static_cast<unsigned>(std::max<std::int64_t>(1, left)));
    ++statistics.solver_calls;
    Timer timer(statistics.z3_ms);
    return solver.check();
  }
  bool verifyMeasured() {
    ++statistics.candidates;
    Timer timer(statistics.verification_ms);
    auto result = verify();
    if (!result) ++statistics.rejected_candidates;
    return result;
  }
  z3::expr integer(std::uint64_t n) {
    return ctx.int_val(std::to_string(n).c_str());
  }
  Vector literal(const List& xs) {
    if (xs.size() > options.max_vector_capacity) throw Limit("vector capacity");
    Vector v{integer(xs.size()), {}};
    for (auto x : xs) v.entries.push_back(integer(x));
    return v;
  }
  void require(z3::expr c) {
    c = c.simplify();
    if (generating)
      solver.add(c);
    else if (c.is_false())
      throw Illegal{};
    else if (!c.is_true())
      throw Limit("nonconcrete candidate index");
  }
  z3::expr checked(z3::expr value) {
    require(value >= ctx.int_val("-170141183460469231731687303715884105728") &&
            value <= ctx.int_val("170141183460469231731687303715884105727"));
    return value.simplify();
  }
  Vector vector(const Value& v) {
    if (!v.vector) throw Limit("vector required");
    auto out = *v.vector;
    auto length = out.length.simplify();
    if (length.is_numeral()) {
      auto n = length.get_numeral_uint64();
      if (n > out.entries.size()) throw Illegal{};
      out.entries.resize(n, integer(0));
    }
    return out;
  }
  z3::expr at(const Vector& v, const z3::expr& i, bool checked = true) {
    if (checked) require(i >= 0 && i < v.length);
    auto position = i.simplify();
    if (position.is_numeral()) {
      if ((position < 0 || position >= integer(v.entries.size()))
              .simplify()
              .is_true())
        return integer(0);
      return v.entries.at(position.get_numeral_uint64());
    }
    auto x = integer(0);
    if (auto domain = index_domains.find(position.id());
        domain != index_domains.end()) {
      for (auto it = domain->second.rbegin(); it != domain->second.rend(); ++it)
        if (*it < v.entries.size())
          x = z3::ite(position == integer(*it), v.entries[*it], x);
    } else {
      for (std::size_t j = v.entries.size(); j-- > 0;)
        x = z3::ite(position == integer(j), v.entries[j], x);
    }
    return x.simplify();
  }
  z3::expr contains(const Vector& v, z3::expr x) {
    auto length = v.length.simplify();
    auto value = x.simplify();
    if (length.is_numeral() && value.is_numeral() &&
        (length >= 0 && length <= integer(v.entries.size()))
            .simplify()
            .is_true()) {
      auto count = length.get_numeral_uint64();
      bool concrete = true, found = false;
      for (std::size_t j = 0; j < count; ++j) {
        auto entry = v.entries[j].simplify();
        concrete &= entry.is_numeral();
        found |= z3::eq(entry, value);
      }
      if (concrete) return ctx.bool_val(found);
    }
    auto b = ctx.bool_val(false);
    for (std::size_t j = 0; j < v.entries.size(); ++j)
      b = b || (integer(j) < v.length && v.entries[j] == x);
    return b.simplify();
  }
  z3::expr equal(const Vector& a, const Vector& b) {
    auto out = a.length == b.length;
    for (std::size_t j = 0; j < std::max(a.entries.size(), b.entries.size());
         ++j)
      out = out &&
            z3::implies(integer(j) < a.length,
                        at(a, integer(j), false) == at(b, integer(j), false));
    return out.simplify();
  }
  z3::expr valid(const Vector& v, z3::expr n) {
    auto out = v.length >= 0 && v.length <= n;
    auto length = v.length.simplify();
    auto bound = n.simplify();
    if (length.is_numeral() && bound.is_numeral() &&
        (length >= 0 && length <= integer(v.entries.size()))
            .simplify()
            .is_true()) {
      auto count = length.get_numeral_uint64();
      bool concrete = true, unique = true;
      std::set<unsigned> seen;
      for (std::size_t i = 0; i < count; ++i) {
        auto entry = v.entries[i].simplify();
        concrete &= entry.is_numeral();
        unique &= seen.insert(entry.id()).second;
        out = out && entry >= 0 && entry < n;
      }
      if (concrete) return (out && ctx.bool_val(unique)).simplify();
    }
    z3::expr_vector distinct(ctx);
    for (std::size_t i = 0; i < v.entries.size(); ++i) {
      auto active = integer(i) < v.length;
      out = out && z3::implies(active, v.entries[i] >= 0 && v.entries[i] < n);
      // Inactive slots receive different sentinels outside the active range.
      distinct.push_back(z3::ite(active, v.entries[i], n + integer(i)));
    }
    if (distinct.size() > 1) out = out && z3::distinct(distinct);
    return out.simplify();
  }
  using Environment = std::map<std::string, Vector>;
  std::map<std::tuple<std::size_t, std::size_t, std::string>, Value>
      concrete_cache;
  std::map<std::pair<std::size_t, std::size_t>, Value> metadata_cache;
  Value eval(std::size_t node, std::size_t id, const Environment& env = {}) {
    const auto key = std::pair(node, id);
    if (generating && env.empty()) {
      if (metadata_cache.contains(key)) return metadata_cache.at(key);
      auto value = evaluate(node, id, env);
      metadata_cache.emplace(key, value);
      return value;
    }
    std::string environment;
    for (auto& [name, v] : env) {
      environment += std::to_string(name.size()) + ":" + name + ":" +
                     std::to_string(v.length.id()) + ":";
      for (auto& x : v.entries) environment += std::to_string(x.id()) + ",";
      environment += ";";
    }
    auto concrete_key = std::tuple(node, id, environment);
    if (concrete_cache.contains(concrete_key))
      return concrete_cache.at(concrete_key);
    auto value = evaluate(node, id, env);
    if (concrete_cache.size() > 20000) throw Limit("expression cache limit");
    concrete_cache.emplace(concrete_key, value);
    return value;
  }
  Value evaluate(std::size_t node, std::size_t id, const Environment& env) {
    tick();
    const auto& expressions = node == plan.nodes.size()
                                  ? plan.constraints.expressions
                                  : plan.nodes.at(node).expressions;
    const auto& e = expressions.at(id);
    auto arg = [&](std::size_t i) { return eval(node, e.args.at(i), env); };
    if (e.args.empty()) {
      if (env.contains(e.op)) return Value(env.at(e.op));
      if (e.op.starts_with("@shape:"))
        return Value(shapes.at(std::stoul(e.op.substr(7))));
      if (e.op.starts_with("@attr:")) {
        auto colon = e.op.find(':', 6);
        return Value(attrs.at(std::stoul(e.op.substr(6, colon - 6)))
                         .at(std::stoul(e.op.substr(colon + 1))));
      }
      if (e.op == "true" || e.op == "false")
        return Value(ctx.bool_val(e.op == "true"));
      if (e.op.starts_with('a'))
        return Value(attrs.at(node).at(std::stoul(e.op.substr(1))));
      if (!e.op.empty() &&
          e.op.find_first_not_of("0123456789") == std::string::npos)
        return Value(ctx.int_val(e.op.c_str()), std::stoull(e.op));
      if (e.op == "list") return Value(literal({}));
      throw Limit("unbound metadata name: " + e.op);
    }
    if (e.op == "shape" || e.op == "rank") {
      const auto& n = plan.nodes.at(node);
      auto tensor = n.expressions.at(e.args[0]).op;
      if (!tensor.starts_with('t')) throw Limit("shape requires an operand");
      auto v = shapes.at(n.inputs.at(std::stoul(tensor.substr(1))));
      return e.op == "shape" ? Value(v) : Value(v.length, v.entries.size());
    }
    if (e.op == "index_space") return arg(0);
    if (e.op == "len") {
      auto v = vector(arg(0));
      return Value(v.length, v.entries.size());
    }
    if (e.op == "list") {
      if (e.args.size() > options.max_vector_capacity)
        throw Limit("vector capacity");
      Vector v{integer(e.args.size()), {}};
      for (std::size_t i = 0; i < e.args.size(); ++i)
        v.entries.push_back(arg(i).scalar);
      return Value(v);
    }
    if (e.op == "range") {
      auto n = arg(0);
      if (!n.upper || *n.upper > options.max_vector_capacity)
        throw Limit("range lacks a practical finite bound");
      require(n.scalar >= 0 && n.scalar <= integer(*n.upper));
      Vector v{n.scalar, {}};
      for (std::size_t i = 0; i < *n.upper; ++i)
        v.entries.push_back(integer(i));
      return Value(v);
    }
    if (e.op == "index") return Value(at(vector(arg(0)), arg(1).scalar));
    if (e.op == "gather") {
      auto v = vector(arg(0)), a = vector(arg(1));
      Vector out{a.length, {}};
      for (std::size_t i = 0; i < a.entries.size(); ++i) {
        require(z3::implies(integer(i) < a.length,
                            a.entries[i] >= 0 && a.entries[i] < v.length));
        out.entries.push_back(at(v, a.entries[i], false));
      }
      return Value(out);
    }
    if (e.op == "exclude") {
      auto v = vector(arg(0)), a = vector(arg(1));
      auto all_concrete = [](const Vector& x) {
        return x.length.simplify().is_numeral() &&
               std::all_of(x.entries.begin(), x.entries.end(), [](auto& entry) {
                 return entry.simplify().is_numeral();
               });
      };
      if (all_concrete(v) && all_concrete(a)) {
        auto values = constants(v), removed = constants(a);
        std::set<std::uint64_t> selected(removed.begin(), removed.end());
        std::erase_if(values, [&](auto x) { return selected.contains(x); });
        return Value(literal(values));
      }
      std::vector<z3::expr> positions, keep;
      auto length = integer(0);
      for (std::size_t i = 0; i < v.entries.size(); ++i) {
        tick();
        positions.push_back(length);
        // Match the metadata builtin: remove selected values, preserving order.
        keep.push_back(integer(i) < v.length && !contains(a, v.entries[i]));
        length = length + z3::ite(keep.back(), integer(1), integer(0));
      }
      Vector out{length.simplify(), {}};
      for (std::size_t j = 0; j < v.entries.size(); ++j) {
        tick();
        auto x = integer(0);
        for (std::size_t i = 0; i < v.entries.size(); ++i)
          x = z3::ite(keep[i] && positions[i] == integer(j), v.entries[i], x);
        out.entries.push_back(x.simplify());
      }
      return Value(out);
    }
    if (e.op == "concat") {
      Vector out = vector(arg(0));
      for (std::size_t j = 1; j < e.args.size(); ++j) {
        auto v = vector(arg(j));
        if (out.entries.size() + v.entries.size() > options.max_vector_capacity)
          throw Limit("vector capacity");
        if (out.length.simplify().is_numeral() &&
            v.length.simplify().is_numeral()) {
          out.entries.insert(out.entries.end(), v.entries.begin(),
                             v.entries.end());
          out.length = (out.length + v.length).simplify();
          continue;
        }
        Vector next{(out.length + v.length).simplify(), {}};
        for (std::size_t i = 0; i < out.entries.size() + v.entries.size(); ++i)
          next.entries.push_back(z3::ite(integer(i) < out.length,
                                         at(out, integer(i), false),
                                         at(v, integer(i) - out.length, false))
                                     .simplify());
        out = next;
      }
      return Value(out);
    }
    if (e.op == "take" || e.op == "slice") {
      auto v = vector(arg(0));
      auto start = arg(1).scalar, count = arg(2).scalar;
      require(start >= 0 && count >= 0 && start + count <= v.length);
      Vector out{count, {}};
      for (std::size_t i = 0; i < v.entries.size(); ++i)
        out.entries.push_back(at(v, start + integer(i), false));
      return Value(out);
    }
    if (e.op == "scatter") {
      auto size = arg(0);
      auto axes = vector(arg(1)), values = vector(arg(2));
      if (!size.upper || *size.upper > options.max_vector_capacity)
        throw Limit("scatter lacks a finite bound");
      require(valid(axes, size.scalar) && axes.length == size.scalar &&
              values.length == axes.length);
      Vector out{size.scalar, {}};
      if (size.scalar.simplify().is_numeral() &&
          axes.length.simplify().is_numeral() &&
          std::all_of(
              axes.entries.begin(), axes.entries.end(),
              [](auto& entry) { return entry.simplify().is_numeral(); })) {
        auto count = size.scalar.simplify().get_numeral_uint64();
        out.entries.resize(count, integer(0));
        for (std::size_t j = 0; j < axes.entries.size(); ++j) {
          auto position = axes.entries[j].simplify().get_numeral_uint64();
          if (position >= count) throw Illegal{};
          out.entries[position] = at(values, integer(j), false);
        }
        return Value(out);
      }
      for (std::size_t i = 0; i < *size.upper; ++i) {
        tick();
        auto x = integer(0);
        for (std::size_t j = 0; j < axes.entries.size(); ++j)
          x = z3::ite(integer(j) < axes.length && axes.entries[j] == integer(i),
                      at(values, integer(j), false), x);
        out.entries.push_back(x.simplify());
      }
      return Value(out);
    }
    if (e.op == "is_valid_axis_list")
      return Value(valid(vector(arg(0)), arg(1).scalar));
    if (e.op == "is_disjoint") {
      auto a = vector(arg(0)), b = vector(arg(1));
      auto out = ctx.bool_val(true);
      for (std::size_t i = 0; i < a.entries.size(); ++i)
        out = out &&
              z3::implies(integer(i) < a.length, !contains(b, a.entries[i]));
      return Value(out.simplify());
    }
    if (e.op == "if") {
      auto c = arg(0).scalar.simplify();
      // Branch-local legality cannot be flattened into unconditional
      // assertions.
      if (c.is_true()) return arg(1);
      if (c.is_false()) return arg(2);
      throw Limit("symbolic conditional requires guarded lowering");
    }
    auto a = arg(0);
    if (e.op == "unary:!") return Value(!a.scalar);
    if (e.op == "unary:+") return a;
    if (e.op == "unary:-") return Value(checked(-a.scalar));
    if (e.op == "&&" || e.op == "||") {
      auto condition = a.scalar.simplify();
      if (!condition.is_true() && !condition.is_false())
        throw Limit(
            "symbolic short-circuit metadata requires guarded lowering");
      if ((e.op == "&&" && condition.is_false()) ||
          (e.op == "||" && condition.is_true()))
        return Value(condition);
      return arg(1);
    }
    auto b = arg(1);
    if (e.op == "==" || e.op == "!=") {
      auto q = a.vector && b.vector ? equal(*a.vector, *b.vector)
                                    : a.scalar == b.scalar;
      return Value(e.op == "==" ? q : !q);
    }
    if (e.op == "+") {
      std::optional<std::size_t> upper;
      if (a.upper && b.upper) {
        if (*a.upper > std::numeric_limits<std::size_t>::max() - *b.upper)
          throw Limit("metadata capacity overflow");
        upper = *a.upper + *b.upper;
      }
      return Value(checked(a.scalar + b.scalar), upper);
    }
    if (e.op == "-") return Value(checked(a.scalar - b.scalar));
    if (e.op == "*") return Value(checked(a.scalar * b.scalar));
    if (e.op == "/" || e.op == "%") {
      require(b.scalar != 0);
      auto magnitude = z3::ite(a.scalar < 0, -a.scalar, a.scalar) /
                       z3::ite(b.scalar < 0, -b.scalar, b.scalar);
      auto quotient =
          z3::ite((a.scalar < 0) != (b.scalar < 0), -magnitude, magnitude);
      return Value(
          checked(e.op == "/" ? quotient : a.scalar - b.scalar * quotient));
    }
    if (e.op == "<") return Value(a.scalar < b.scalar);
    if (e.op == "<=") return Value(a.scalar <= b.scalar);
    if (e.op == ">") return Value(a.scalar > b.scalar);
    if (e.op == ">=") return Value(a.scalar >= b.scalar);
    throw Limit("unsupported primitive: " + e.op);
  }
  List concrete(const Vector& v, const z3::model& m) {
    auto size = m.eval(v.length, true).get_numeral_uint64();
    if (size > v.entries.size()) throw Illegal{};
    List out;
    for (std::size_t i = 0; i < size; ++i)
      out.push_back(m.eval(v.entries[i], true).get_numeral_uint64());
    return out;
  }
  // Polynomial over independent tensor entries. Monomials are sorted multisets
  // of (capture, concrete coordinates); coefficients are exact Z3 rationals.
  using Monomial = std::vector<List>;
  using Polynomial = std::map<Monomial, z3::expr>;
  struct Computation {
    Polynomial polynomial;
    std::string trace;
  };
  void accumulate(Polynomial& out, const Monomial& key, z3::expr coefficient) {
    auto i = out.find(key);
    if (i != out.end()) coefficient = coefficient + i->second;
    coefficient = coefficient.simplify();
    if (coefficient.to_string() == "0" || coefficient.to_string() == "0.0")
      out.erase(key);
    else
      out.insert_or_assign(key, coefficient);
    if (out.size() > 4096) throw Limit("polynomial term limit");
  }
  List constants(const Vector& v) {
    auto len = v.length.simplify().get_numeral_uint64();
    if (len > v.entries.size()) throw Illegal{};
    List out;
    for (std::size_t i = 0; i < len; ++i)
      out.push_back(v.entries[i].simplify().get_numeral_uint64());
    return out;
  }
  Computation expand(std::size_t id, const List& output) {
    tick();
    const auto& n = plan.nodes.at(id);
    auto dims = constants(shapes[id]);
    if (output.size() != dims.size()) throw Illegal{};
    for (std::size_t i = 0; i < dims.size(); ++i)
      if (output[i] >= dims[i]) throw Illegal{};
    if (n.capture >= 0) {
      List entry{static_cast<std::uint64_t>(n.capture)};
      entry.insert(entry.end(), output.begin(), output.end());
      std::string trace = "read:";
      for (auto v : entry) trace += std::to_string(v) + ",";
      return {{{Monomial{entry}, ctx.real_val(1)}}, trace};
    }
    Environment env{{"o", literal(output)}};
    return scalar(id, n.root, env);
  }
  Computation scalar(std::size_t node, std::size_t id, Environment& env) {
    tick();
    const auto& n = plan.nodes[node];
    const auto& e = n.expressions[id];
    if (e.op == "read") {
      auto t = n.expressions[e.args[0]].op;
      auto child = n.inputs.at(std::stoul(t.substr(1)));
      return expand(child, constants(vector(eval(node, e.args[1], env))));
    }
    if (e.op.starts_with("sum:")) {
      auto dims = constants(vector(eval(node, e.args[0], env)));
      List index(dims.size());
      Computation out{{}, "sum("};
      for (auto d : dims) out.trace += std::to_string(d) + ",";
      out.trace += ")[";
      auto name = e.op.substr(4);
      if (dims.empty()) {
        env.insert_or_assign(name, literal({}));
        auto body = scalar(node, e.args[1], env);
        env.erase(name);
        return body;
      }
      auto saved =
          env.find(name) == env.end() ? std::optional<Vector>{} : env.at(name);
      std::function<void(std::size_t)> loop = [&](std::size_t axis) {
        tick();
        if (axis < dims.size()) {
          for (std::uint64_t i = 0; i < dims[axis]; ++i) {
            index[axis] = i;
            loop(axis + 1);
          }
          return;
        }
        env.insert_or_assign(name, literal(index));
        auto body = scalar(node, e.args[1], env);
        for (auto& [k, c] : body.polynomial) accumulate(out.polynomial, k, c);
        out.trace += body.trace + ";";
        if (out.trace.size() > 1000000) throw Limit("trace limit");
      };
      loop(0);
      if (saved)
        env.insert_or_assign(name, *saved);
      else
        env.erase(name);
      out.trace += "]";
      return out;
    }
    if (e.op == "if") {
      auto c = eval(node, e.args[0], env).scalar.simplify();
      if (!c.is_true() && !c.is_false())
        throw Limit("value-dependent condition");
      return scalar(node, e.args[c.is_true() ? 1 : 2], env);
    }
    if (e.args.empty() &&
        e.op.find_first_not_of("0123456789") == std::string::npos)
      return {{{Monomial{}, ctx.real_val(e.op.c_str())}}, e.op};
    if (e.op == "unary:+") return scalar(node, e.args[0], env);
    if (e.op == "unary:-") {
      auto a = scalar(node, e.args[0], env);
      for (auto& [k, c] : a.polynomial) c = (-c).simplify();
      a.trace = "neg(" + a.trace + ")";
      return a;
    }
    auto a = scalar(node, e.args.at(0), env),
         b = scalar(node, e.args.at(1), env);
    Computation out{{}, e.op + "(" + a.trace + "," + b.trace + ")"};
    if (e.op == "+" || e.op == "-") {
      out.polynomial = a.polynomial;
      for (auto& [k, c] : b.polynomial)
        accumulate(out.polynomial, k, e.op == "+" ? c : -c);
    } else if (e.op == "*") {
      for (auto& [x, c] : a.polynomial)
        for (auto& [y, d] : b.polynomial) {
          tick();
          auto key = x;
          key.insert(key.end(), y.begin(), y.end());
          std::sort(key.begin(), key.end());
          accumulate(out.polynomial, key, c * d);
        }
    } else
      throw Limit("scalar division and nonpolynomial arithmetic unsupported");
    return out;
  }
  struct Factor {
    std::size_t capture;
    List coordinates;
  };
  struct IndexedTerm {
    std::vector<Factor> factors;
    List bound;
    z3::expr coefficient;
  };
  using Indexed = std::vector<IndexedTerm>;
  List coordinate_dims;
  Vector coordinates(const List& ids) {
    Vector v{integer(ids.size()), {}};
    for (auto id : ids)
      v.entries.push_back(
          ctx.int_const(("coord_" + std::to_string(id)).c_str()));
    return v;
  }
  Indexed indexed(std::size_t node, const Vector& output) {
    tick();
    const auto& n = plan.nodes[node];
    auto dims = constants(shapes[node]);
    if (output.entries.size() != dims.size()) throw Illegal{};
    if (n.capture >= 0) {
      Factor f{static_cast<std::size_t>(n.capture), {}};
      for (std::size_t i = 0; i < dims.size(); ++i) {
        auto c = output.entries[i].simplify();
        if (c.is_numeral()) {
          auto k = c.get_numeral_uint64();
          if (k >= dims[i]) throw Illegal{};
          // High bit distinguishes fixed positions from coordinate identities.
          if (k >= (std::uint64_t(1) << 63)) throw Limit("coordinate capacity");
          f.coordinates.push_back(k | (std::uint64_t(1) << 63));
        } else {
          auto name = c.to_string();
          if (!name.starts_with("coord_"))
            throw Limit("index arithmetic needs finite verification");
          auto id = std::stoull(name.substr(6));
          if (id >= coordinate_dims.size() || coordinate_dims[id] != dims[i])
            throw Limit("coordinate domain mismatch");
          f.coordinates.push_back(dims[i] == 1 ? (std::uint64_t(1) << 63) : id);
        }
      }
      return {{{f}, {}, ctx.real_val(1)}};
    }
    Environment env{{"o", output}};
    return indexedScalar(node, n.root, env);
  }
  Indexed indexedScalar(std::size_t node, std::size_t id, Environment& env) {
    tick();
    const auto& n = plan.nodes[node];
    const auto& e = n.expressions[id];
    if (e.op == "read") {
      auto operand = std::stoul(n.expressions[e.args[0]].op.substr(1));
      auto out = vector(eval(node, e.args[1], env));
      auto count = out.length.simplify().get_numeral_uint64();
      if (count > out.entries.size()) throw Illegal{};
      out.entries.erase(out.entries.begin() + count, out.entries.end());
      return indexed(n.inputs.at(operand), out);
    }
    if (e.op.starts_with("sum:")) {
      auto dims = constants(vector(eval(node, e.args[0], env)));
      if (std::find(dims.begin(), dims.end(), 0) != dims.end()) return {};
      List bound;
      for (auto dim : dims) {
        bound.push_back(coordinate_dims.size());
        coordinate_dims.push_back(dim);
      }
      auto name = e.op.substr(4);
      env.insert_or_assign(name, coordinates(bound));
      auto out = indexedScalar(node, e.args[1], env);
      env.erase(name);
      for (auto& term : out)
        term.bound.insert(term.bound.end(), bound.begin(), bound.end());
      return out;
    }
    if (e.op == "if") {
      auto c = eval(node, e.args[0], env).scalar.simplify();
      if (!c.is_true() && !c.is_false())
        throw Limit("symbolic index predicate needs finite verification");
      return indexedScalar(node, e.args[c.is_true() ? 1 : 2], env);
    }
    if (e.args.empty() &&
        e.op.find_first_not_of("0123456789") == std::string::npos)
      return {{{}, {}, ctx.real_val(e.op.c_str())}};
    auto a = indexedScalar(node, e.args.at(0), env);
    if (e.op == "unary:+") return a;
    if (e.op == "unary:-") {
      for (auto& t : a) t.coefficient = -t.coefficient;
      return a;
    }
    auto b = indexedScalar(node, e.args.at(1), env);
    if (e.op == "+" || e.op == "-") {
      if (e.op == "-")
        for (auto& t : b) t.coefficient = -t.coefficient;
      a.insert(a.end(), b.begin(), b.end());
    } else if (e.op == "*") {
      Indexed out;
      for (const auto& x : a)
        for (const auto& y : b) {
          tick();
          auto t = x;
          t.factors.insert(t.factors.end(), y.factors.begin(), y.factors.end());
          t.bound.insert(t.bound.end(), y.bound.begin(), y.bound.end());
          t.coefficient = t.coefficient * y.coefficient;
          out.push_back(t);
          if (out.size() > 4096) throw Limit("indexed polynomial term limit");
        }
      a = std::move(out);
    } else
      throw Limit("unsupported scalar expression");
    if (a.size() > 4096) throw Limit("indexed polynomial term limit");
    return a;
  }
  std::map<List, z3::expr> normalized(std::size_t root) {
    coordinate_dims = constants(shapes[root]);
    List output(coordinate_dims.size());
    std::iota(output.begin(), output.end(), 0);
    auto terms = indexed(root, coordinates(output));
    std::map<List, z3::expr> result;
    for (auto term : terms) {
      std::sort(
          term.factors.begin(), term.factors.end(),
          [](const auto& a, const auto& b) {
            return a.capture < b.capture ||
                   (a.capture == b.capture && a.coordinates < b.coordinates);
          });
      List best;
      bool first = true;
      z3::expr best_coefficient = term.coefficient;
      std::function<void(std::size_t)> visit = [&](std::size_t begin) {
        tick();
        if (begin < term.factors.size()) {
          auto end = begin + 1;
          while (end < term.factors.size() &&
                 term.factors[end].capture == term.factors[begin].capture)
            ++end;
          do {
            visit(end);
          } while (std::next_permutation(term.factors.begin() + begin,
                                         term.factors.begin() + end,
                                         [](const auto& a, const auto& b) {
                                           return a.coordinates < b.coordinates;
                                         }));
          return;
        }
        std::map<std::uint64_t, std::uint64_t> names;
        List key{term.factors.size()};
        for (auto& f : term.factors) {
          key.push_back(f.capture);
          key.push_back(f.coordinates.size());
          for (auto c : f.coordinates) {
            if (c >> 63) {
              key.push_back(2);
              key.push_back(c & ~(std::uint64_t(1) << 63));
            } else if (c < output.size()) {
              key.push_back(0);
              key.push_back(c);
            } else {
              if (std::find(term.bound.begin(), term.bound.end(), c) ==
                  term.bound.end())
                throw Limit("unbound coordinate");
              auto [i, inserted] = names.emplace(c, names.size());
              key.push_back(1);
              key.push_back(i->second);
              key.push_back(coordinate_dims.at(c));
            }
          }
        }
        auto coefficient = term.coefficient;
        for (auto b : term.bound)
          if (!names.contains(b))
            coefficient = coefficient * integer(coordinate_dims[b]);
        if (first || key < best) {
          best = key;
          first = false;
          best_coefficient = coefficient;
        }
      };
      visit(0);
      auto i = result.find(best);
      if (i != result.end()) best_coefficient = best_coefficient + i->second;
      best_coefficient = best_coefficient.simplify();
      if ((best_coefficient == 0).simplify().is_true())
        result.erase(best);
      else
        result.insert_or_assign(best, best_coefficient);
    }
    return result;
  }
  bool indexedEqual() {
    if (!plan.algebraic) return false;
    const auto a = normalized(plan.source_root), b = normalized(plan.rhs_root);
    if (a.size() != b.size()) return false;
    auto x = a.begin(), y = b.begin();
    for (; x != a.end(); ++x, ++y)
      if (x->first != y->first ||
          !(x->second == y->second).simplify().is_true())
        return false;
    return true;
  }
  bool verify() {
    auto dims = constants(shapes[plan.source_root]);
    if (dims != constants(shapes[plan.rhs_root])) return false;
    if (std::find(dims.begin(), dims.end(), 0) != dims.end()) return true;
    try {
      if (indexedEqual()) return true;
    } catch (const Limit&) {
      // Index isomorphism is only sufficient. A failed match is not invalid.
    }
    List index(dims.size());
    bool okay = true;
    std::function<void(std::size_t)> visit = [&](std::size_t axis) {
      tick();
      if (!okay) return;
      if (axis < dims.size()) {
        for (std::uint64_t i = 0; i < dims[axis] && okay; ++i) {
          index[axis] = i;
          visit(axis + 1);
        }
        return;
      }
      auto a = expand(plan.source_root, index),
           b = expand(plan.rhs_root, index);
      if (!plan.algebraic) {
        okay = a.trace == b.trace;
        return;
      }
      if (a.polynomial.size() != b.polynomial.size()) {
        okay = false;
        return;
      }
      auto x = a.polynomial.begin(), y = b.polynomial.begin();
      for (; x != a.polynomial.end(); ++x, ++y)
        if (x->first != y->first ||
            !(x->second == y->second).simplify().is_true()) {
          okay = false;
          break;
        }
    };
    visit(0);
    return okay;
  }
  bool sameProgram(const Node& a, const Node& b) {
    if (a.capture >= 0 || b.capture >= 0 || a.root != b.root ||
        a.shape != b.shape || a.assertions != b.assertions ||
        a.expressions.size() != b.expressions.size() ||
        a.fields.size() != b.fields.size())
      return false;
    for (std::size_t i = 0; i < a.expressions.size(); ++i)
      if (a.expressions[i].op != b.expressions[i].op ||
          a.expressions[i].args != b.expressions[i].args)
        return false;
    return true;
  }
  Solution witness() {
    std::vector<List> out;
    for (std::size_t i = 0; i < attrs.size(); ++i)
      for (std::size_t j = 0; j < attrs[i].size(); ++j) {
        auto f = plan.nodes[i].fields[j].output;
        if (out.size() <= f) out.resize(f + 1);
        out[f] = constants(attrs[i][j]);
      }
    return {Status::Found, out, "exact verification succeeded"};
  }
  [[noreturn]] void constraintLimit(const std::string& reason) {
    throw Limit(reason);
  }
  // Instantiate the existing generic source-copy/identity preference first.
  // All primitive constraints and tensor equivalence must still hold. This
  // avoids constructing padded symbolic vectors for an already valid witness.
  bool preferredCandidate() {
    generating = false;
    std::map<std::size_t, Vector> shared;
    for (std::size_t id = 0; id < plan.nodes.size(); ++id) {
      tick();
      const auto& n = plan.nodes[id];
      attrs.emplace_back();
      if (n.capture >= 0) {
        shapes.push_back(literal(captures.at(n.capture)));
        continue;
      }
      for (auto& f : n.fields)
        attrs.back().push_back(f.known >= 0   ? literal(known.at(f.known))
                               : f.known < -1 ? literal(pinned.at(-2 - f.known))
                                              : literal({}));
      for (std::size_t f = 0; f < n.fields.size(); ++f) {
        if (n.fields[f].known != -1) continue;
        if (shared.contains(n.fields[f].output)) {
          attrs[id][f] = shared.at(n.fields[f].output);
          concrete_cache.clear();
          continue;
        }
        bool copied = false;
        for (std::size_t source = 0; source < plan.source_end && source < id;
             ++source)
          if (sameProgram(n, plan.nodes[source])) {
            attrs[id][f] = attrs[source][f];
            copied = true;
            break;
          }
        if (!copied) {
          auto size = eval(id, n.fields[f].bound)
                          .scalar.simplify()
                          .get_numeral_uint64();
          if (size > options.max_vector_capacity)
            throw Limit("attribute domain capacity");
          List identity(size);
          std::iota(identity.begin(), identity.end(), 0);
          attrs[id][f] = literal(identity);
        }
        shared.emplace(n.fields[f].output, attrs[id][f]);
        concrete_cache.clear();
      }
      for (auto a : n.assertions) require(eval(id, a).scalar);
      shapes.push_back(literal(constants(vector(eval(id, n.shape)))));
    }
    return verifyMeasured();
  }
  void initializeMetadata(std::size_t count, bool check_source = false) {
    for (std::size_t id = 0; id < count; ++id) {
      if (check_source && id == plan.source_end && checkSolver() != z3::sat)
        throw Limit("illegal or undecidable source metadata");
      const auto& n = plan.nodes[id];
      attrs.emplace_back();
      if (n.capture >= 0) {
        shapes.push_back(literal(captures.at(n.capture)));
        continue;
      }
      // Known fields are installed before unknown domain expressions are read.
      for (const auto& f : n.fields)
        attrs.back().push_back(f.known >= 0   ? literal(known.at(f.known))
                               : f.known < -1 ? literal(pinned.at(-2 - f.known))
                                              : literal({}));
      std::set<std::size_t> initialized, active;
      std::function<void(std::size_t)> install = [&](std::size_t j) {
        const auto& f = n.fields[j];
        if (f.known != -1 || initialized.contains(j)) return;
        if (!active.insert(j).second)
          throw Limit("cyclic attribute domain dependency");
        if (variables.contains(f.output)) {
          attrs[id][j] = variables.at(f.output);
          metadata_cache.clear();
          initialized.insert(j);
          active.erase(j);
          return;
        }
        std::set<std::size_t> visited;
        std::function<void(std::size_t)> dependencies = [&](std::size_t e) {
          if (!visited.insert(e).second) return;
          const auto& expr = n.expressions.at(e);
          if (expr.args.empty() && expr.op.starts_with('a'))
            install(std::stoul(expr.op.substr(1)));
          for (auto arg : expr.args) dependencies(arg);
        };
        dependencies(f.bound);
        metadata_cache.clear();
        auto bound = eval(id, f.bound);
        if (!bound.upper || *bound.upper > options.max_vector_capacity)
          throw Limit("attribute domain bound exceeds supported capacity");
        const AxisDomain* domain =
            propagated ? &propagated->fields.at(f.output) : nullptr;
        auto capacity = domain ? domain->entries.size() : *bound.upper;
        auto choose = [&](const std::string& name, const List* allowed) {
          if (allowed && allowed->size() == 1) return integer(allowed->front());
          if (++symbolic_axes > options.max_symbolic_axis_variables)
            throw Limit("unresolved attribute variable budget");
          ++statistics.symbolic_axis_variables;
          auto x = ctx.int_const(name.c_str());
          if (allowed) {
            index_domains.emplace(x.id(), *allowed);
            auto condition = ctx.bool_val(false);
            for (auto n : *allowed) condition = condition || x == integer(n);
            require(condition);
          }
          return x;
        };
        Vector v{capacity == 0 ? integer(0)
                               : choose("len_" + std::to_string(f.output),
                                        domain ? &domain->lengths : nullptr),
                 {}};
        for (std::size_t k = 0; k < capacity; ++k) {
          auto name =
              "axis_" + std::to_string(f.output) + "_" + std::to_string(k);
          auto minimum_length = domain ? domain->lengths.front() : 0;
          // Domain entries constrain active slots only. Use an explicit
          // inactive zero rather than restricting a padded variable to its
          // active domain.
          auto entry = choose(name, domain ? &domain->entries[k] : nullptr);
          if (domain && k >= minimum_length)
            entry = z3::ite(integer(k) < v.length, entry, integer(0));
          v.entries.push_back(entry);
        }
        require(v.length >= 0 && v.length <= integer(v.entries.size()));
        for (std::size_t k = 0; k < v.entries.size(); ++k)
          require(z3::implies(integer(k) >= v.length, v.entries[k] == 0));
        require(valid(v, bound.scalar));
        for (std::size_t k = 0; k < v.entries.size(); ++k)
          require(z3::implies(
              integer(k) < v.length,
              v.entries[k] <=
                  integer(std::numeric_limits<std::uint64_t>::max())));
        variables.emplace(f.output, v);
        attrs[id][j] = v;
        metadata_cache.clear();
        initialized.insert(j);
        active.erase(j);
      };
      for (std::size_t j = 0; j < n.fields.size(); ++j) install(j);
      for (auto a : n.assertions) require(eval(id, a).scalar);
      shapes.push_back(vector(eval(id, n.shape)));
      auto length = shapes.back().length.simplify();
      if (length.is_numeral()) {
        auto size = length.get_numeral_uint64();
        if (size > shapes.back().entries.size()) throw Illegal{};
        shapes.back().entries.erase(shapes.back().entries.begin() + size,
                                    shapes.back().entries.end());
      }
      for (std::size_t k = 0; k < shapes.back().entries.size(); ++k)
        require(z3::implies(
            integer(k) < shapes.back().length,
            shapes.back().entries[k] >= 0 &&
                shapes.back().entries[k] <=
                    integer(std::numeric_limits<std::uint64_t>::max())));
    }
  }
  Solution run() {
    if (propagated && propagated->residual_entries == 0 &&
        std::all_of(
            propagated->fields.begin(), propagated->fields.end(),
            [](const auto& item) { return item.second.lengths.size() == 1; })) {
      generating = false;
      for (std::size_t id = 0; id < plan.nodes.size(); ++id) {
        tick();
        const auto& n = plan.nodes[id];
        attrs.emplace_back();
        if (n.capture >= 0) {
          shapes.push_back(literal(captures.at(n.capture)));
          continue;
        }
        for (const auto& f : n.fields) {
          if (f.known != -1) {
            attrs.back().push_back(literal(
                f.known >= 0 ? known.at(f.known) : pinned.at(-2 - f.known)));
          } else {
            List xs;
            for (const auto& entry : propagated->fields.at(f.output).entries)
              xs.push_back(entry.at(0));
            attrs.back().push_back(literal(xs));
          }
        }
        for (auto a : n.assertions) require(eval(id, a).scalar);
        shapes.push_back(literal(constants(vector(eval(id, n.shape)))));
      }
      if (!verifyMeasured()) throw RetryUnrestricted{};
      statistics.propagation_direct = true;
      statistics.constraint_attempted = true;
      auto out = witness();
      out.method =
          "constraints-z3";  // Keep the public inference method stable.
      out.reason =
          "axis propagation and exact verification succeeded without SMT "
          "search";
      return out;
    }
    if (options.semantic_constraints && !plan.constraints.available) {
      statistics.constraint_fallback = true;
      statistics.fallback_reason = plan.constraints.reason;
    }
    try {
      if (options.preferred_candidate &&
          (!options.semantic_constraints || !plan.constraints.available)) {
        statistics.preferred_attempted = true;
        if (preferredCandidate()) return witness();
      }
    } catch (const Illegal&) {
      // A preference can be illegal; it never restricts the solver domain.
    } catch (const Limit&) {
      // An unsupported preference must not be treated as impossibility.
    }
    generating = true;
    shapes.clear();
    attrs.clear();
    metadata_cache.clear();
    concrete_cache.clear();
    Timer metadata_timer(statistics.metadata_ms);
    initializeMetadata(plan.nodes.size(), true);
    if (plan.nodes.size() == plan.source_end && checkSolver() != z3::sat)
      throw Limit("illegal or undecidable source metadata");
    require(equal(shapes[plan.source_root], shapes[plan.rhs_root]));
    auto symbolic_shapes = shapes;
    auto symbolic_attrs = attrs;
    metadata_timer.stop();
    bool compiled = false;
    if (options.semantic_constraints && plan.constraints.available &&
        plan.algebraic) {
      statistics.constraint_attempted = true;
      Timer timer(statistics.constraints_ms);
      solver.push();
      try {
        detail::ConstraintMatcher<Engine>{*this}.instantiate();
        compiled = true;
      } catch (const Limit& e) {
        solver.pop();
        statistics.constraint_fallback = true;
        statistics.fallback_reason = e.what();
        metadata_cache.clear();
        concrete_cache.clear();
      }
    }
    bool inconclusive = false;
    while (true) {
      tick();
      auto check = checkSolver();
      if (check == z3::unknown) throw Limit(solver.reason_unknown());
      if (check == z3::unsat) {
        if (compiled) {
          if (propagated) throw RetryUnrestricted{};
          solver.pop();
          compiled = false;
          statistics.constraint_fallback = true;
          statistics.fallback_reason = "sufficient constraints exhausted";
          // Sufficient matching constraints cannot justify impossibility.
          metadata_cache.clear();
          concrete_cache.clear();
          continue;
        }
        if (propagated) throw RetryUnrestricted{};
        return {
            plan.algebraic && !inconclusive ? Status::Invalid : Status::Unknown,
            {},
            "no verified attribute assignment"};
      }
      if (options.max_candidates &&
          statistics.candidates >= options.max_candidates)
        throw Limit("candidate limit");
      auto model = solver.get_model();
      generating = false;
      for (std::size_t i = 0; i < shapes.size(); ++i)
        shapes[i] = literal(concrete(symbolic_shapes[i], model));
      for (std::size_t i = 0; i < attrs.size(); ++i)
        for (std::size_t j = 0; j < attrs[i].size(); ++j)
          attrs[i][j] = literal(concrete(symbolic_attrs[i][j], model));
      concrete_cache.clear();
      bool valid_candidate = false;
      try {
        valid_candidate = verifyMeasured();
      } catch (const Illegal&) {
        ++statistics.rejected_candidates;
        inconclusive = true;
      }
      if (valid_candidate) {
        auto out = witness();
        if (compiled) {
          out.method = "constraints-z3";
          out.reason =
              "compiled semantic constraints and exact verification succeeded";
        }
        return out;
      }
      // Exclude precisely this joint candidate, preserving all other models.
      auto block = ctx.bool_val(false);
      for (auto& [output, v] : variables) {
        block = block || v.length != model.eval(v.length, true);
        for (auto& x : v.entries) block = block || x != model.eval(x, true);
      }
      shapes = symbolic_shapes;
      attrs = symbolic_attrs;
      generating = true;
      metadata_cache.clear();
      solver.add(block);
    }
  }
};
void writeNumbers(std::ostream& out, const auto& xs) {
  out << xs.size() << ' ';
  for (auto x : xs) out << x << ' ';
}
std::vector<std::size_t> readNumbers(std::istream& in) {
  std::size_t n;
  if (!(in >> n) || n > 1000000) throw Limit("malformed request");
  std::vector<std::size_t> xs(n);
  for (auto& x : xs)
    if (!(in >> x)) throw Limit("malformed request");
  return xs;
}
inference::SemanticPlan readPlan(std::istream& in) {
  inference::SemanticPlan p;
  std::size_t count;
  std::string version;
  if (!(in >> version) || version != "tepl-constraints-2")
    throw Limit("unsupported constraint plan format");
  if (!(in >> p.source_root >> p.rhs_root >> p.source_end >> p.algebraic >>
        p.timeout_ms >> count) ||
      count > 10000)
    throw Limit("malformed plan");
  p.nodes.resize(count);
  for (auto& n : p.nodes) {
    in >> n.capture;
    n.inputs = readNumbers(in);
    in >> n.root >> n.shape;
    n.assertions = readNumbers(in);
    in >> count;
    if (count > 10000) throw Limit("malformed fields");
    n.fields.resize(count);
    for (auto& f : n.fields) in >> f.known >> f.output >> f.bound;
    in >> count;
    if (count > 100000) throw Limit("malformed expressions");
    n.expressions.resize(count);
    for (auto& e : n.expressions) {
      in >> std::quoted(e.op);
      e.args = readNumbers(in);
    }
  }
  auto& c = p.constraints;
  in >> c.available >> std::quoted(c.reason) >> count;
  if (!in || count > 10000) throw Limit("malformed constraint expressions");
  c.expressions.resize(count);
  for (std::size_t i = 0; i < c.expressions.size(); ++i) {
    auto& e = c.expressions[i];
    in >> std::quoted(e.op);
    e.args = readNumbers(in);
    for (auto arg : e.args)
      if (arg >= i) throw Limit("constraint expressions must be a DAG");
  }
  c.domains = readNumbers(in);
  for (auto domain : c.domains)
    if (domain >= c.expressions.size())
      throw Limit("malformed reduction domain");
  for (auto* terms : {&c.lhs, &c.rhs}) {
    in >> count;
    if (!in || count > 64) throw Limit("malformed constraint terms");
    terms->resize(count);
    for (auto& t : *terms) {
      in >> t.coefficient;
      t.reductions = readNumbers(in);
      for (auto b : t.reductions)
        if (b >= c.domains.size()) throw Limit("malformed reduction binder");
      t.unused_reductions = readNumbers(in);
      for (auto b : t.unused_reductions)
        if (std::find(t.reductions.begin(), t.reductions.end(), b) ==
            t.reductions.end())
          throw Limit("malformed unused reduction");
      in >> count;
      if (!in || count > 10000) throw Limit("malformed tensor reads");
      t.reads.resize(count);
      for (auto& r : t.reads) {
        in >> r.capture >> r.indices;
        if (r.indices >= c.expressions.size())
          throw Limit("malformed access map");
      }
    }
  }
  if (!in) throw Limit("malformed plan");
  return p;
}
#include "src/legacy_semantics/early_verification.inc"
}  // namespace
Solution solve(const inference::SemanticPlan& p,
               const std::vector<List>& shapes, const std::vector<List>& known,
               const std::vector<List>& pinned, std::size_t budget) {
  const auto began = std::chrono::steady_clock::now();
  const auto ordered = std::chrono::steady_clock::now();
  auto direct = solveOrdered(p, shapes, known, pinned, budget);
  const auto ordered_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - ordered)
                              .count();
  if (auto result = std::move(direct)) {
    result->statistics.ordered_ms = ordered_ms;
    result->statistics.total_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - began)
                                      .count();
    return std::move(*result);
  }
  auto remaining = std::chrono::milliseconds(p.timeout_ms) -
                   (std::chrono::steady_clock::now() - began);
  if (remaining <= std::chrono::steady_clock::duration::zero()) {
    Solution result{
        Status::Unknown, {}, "contract admission exhausted timeout"};
    result.statistics.ordered_ms = ordered_ms;
    result.statistics.total_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - began)
                                     .count();
    return result;
  }
  auto fallback = p;
  fallback.timeout_ms = std::max<std::int64_t>(
      1,
      std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count());
  auto result = solveGeneric(fallback, shapes, known, pinned, budget);
  result.statistics.ordered_ms = ordered_ms;
  result.statistics.total_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - began)
                                   .count();
  return result;
}
Solution solveGeneric(const inference::SemanticPlan& p,
                      const std::vector<List>& shapes,
                      const std::vector<List>& known,
                      const std::vector<List>& pinned, std::size_t budget,
                      const SolveOptions& options) {
  SolveStatistics statistics;
  Timer timer(statistics.total_ms);
  Solution result;
  try {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(p.timeout_ms);
    AxisPropagation domains;
    if (options.axis_propagation && options.semantic_constraints &&
        p.constraints.available && p.algebraic) {
      statistics.propagation_attempted = true;
      Timer propagation_timer(statistics.propagation_ms);
      domains = propagateAxes(p, shapes, known, pinned, [&] {
        if (!budget || std::chrono::steady_clock::now() >= deadline)
          throw Limit("axis propagation resource limit");
        --budget;
      });
      statistics.propagation_reason = domains.reason;
      statistics.propagation_applied = domains.available;
      statistics.original_axis_entries = domains.original_entries;
      statistics.forced_axis_entries = domains.forced_entries;
      statistics.residual_axis_entries = domains.residual_entries;
      statistics.propagation_instructions = domains.instructions;
    }
    auto attempt = [&](const AxisPropagation* reduced) {
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                           deadline - std::chrono::steady_clock::now())
                           .count();
      if (remaining <= 0) throw Limit("resource limit");
      auto current = p;
      current.timeout_ms = remaining;
      Engine engine(current, shapes, known, pinned, budget, options, statistics,
                    reduced);
      try {
        auto answer = engine.run();
        budget = engine.budget;
        return answer;
      } catch (...) {
        budget = engine.budget;
        throw;
      }
    };
    if (domains.available) {
      try {
        result = attempt(&domains);
      } catch (const RetryUnrestricted&) {
        statistics.propagation_fallback = true;
        statistics.propagation_reason =
            "reduced sufficient constraints exhausted; rebuilt unrestricted "
            "metadata";
        result = attempt(nullptr);
      } catch (const Illegal&) {
        statistics.propagation_fallback = true;
        statistics.propagation_reason =
            "propagated candidate failed legality; rebuilt unrestricted "
            "metadata";
        result = attempt(nullptr);
      }
    } else {
      result = attempt(nullptr);
    }
  } catch (const Limit& e) {
    result = {Status::Unknown, {}, e.what()};
  } catch (const Illegal&) {
    result = {
        Status::Unknown, {}, "unsupported or illegal source/shape metadata"};
  } catch (const z3::exception& e) {
    result = {Status::Unknown, {}, e.msg()};
  } catch (const std::exception& e) {
    result = {Status::Unknown, {}, e.what()};
  }
  timer.stop();
  result.statistics = std::move(statistics);
  return result;
}
std::string solveRequest(const std::string& request) {
  try {
    std::istringstream in(request);
    auto p = readPlan(in);
    std::size_t count;
    auto lists = [&]() {
      in >> count;
      if (count > 100000) throw Limit("malformed lists");
      std::vector<List> out;
      for (std::size_t i = 0, size = count; i < size; ++i) {
        auto xs = readNumbers(in);
        out.emplace_back(xs.begin(), xs.end());
      }
      return out;
    };
    auto shapes = lists(), known = lists(), pins = lists();
    if (!in) throw Limit("malformed metadata");
    auto result = solve(p, shapes, known, pins);
    std::ostringstream out;
    out << (result.status == Status::Found     ? 0
            : result.status == Status::Invalid ? 1
                                               : 2)
        << ' ';
    out << result.fields.size() << ' ';
    for (auto& f : result.fields) writeNumbers(out, f);
    // Rust exposes the selected inference path and the diagnostic.
    out << result.method << ' ';
    if (result.reason.empty())
      out << '-';
    else {
      constexpr char hex[] = "0123456789abcdef";
      for (unsigned char c : result.reason) out << hex[c >> 4] << hex[c & 15];
    }
    out << ' ';
    return out.str();
  } catch (...) {
    return "2 0 ";
  }
}
}  // namespace tepl::core::semantics

namespace tepl::core::semantics {
MetadataSample sampleMetadata(const inference::SemanticPlan& p,
                              const std::vector<List>& shapes,
                              const std::vector<std::vector<List>>& blocked,
                              std::size_t budget) {
  try {
    auto source = p;
    if (source.source_end > source.nodes.size())
      throw Limit("invalid source boundary");
    source.nodes.resize(source.source_end);
    std::size_t count = 0;
    for (auto& n : p.nodes)
      for (auto& f : n.fields)
        if (f.known >= 0) count = std::max(count, std::size_t(f.known + 1));
    for (auto& n : source.nodes)
      for (auto& f : n.fields) {
        if (f.known < 0 || f.bound >= n.expressions.size())
          throw Limit("source sampling requires bounded index[] attributes");
        f.known = -1;
      }
    const std::vector<List> empty;
    SolveOptions options;
    SolveStatistics statistics;
    EarlyEngine engine(source, shapes, empty, empty, budget, options,
                       statistics);
    engine.initializeMetadata(source.nodes.size());
    for (auto& assignment : blocked) {
      if (assignment.size() != count) throw Limit("malformed blocked sample");
      auto different = engine.ctx.bool_val(false);
      for (std::size_t id = 0; id < source.nodes.size(); ++id)
        for (std::size_t f = 0; f < source.nodes[id].fields.size(); ++f)
          different =
              different ||
              !engine.equal(
                  engine.attrs[id][f],
                  engine.literal(assignment.at(p.nodes[id].fields[f].known)));
      engine.solver.add(different);
    }
    auto status = engine.checkSolver();
    if (status == z3::unknown) throw Limit(engine.solver.reason_unknown());
    if (status == z3::unsat)
      return {Status::Invalid, {}, {}, "source attribute domain exhausted"};
    auto model = engine.solver.get_model();
    MetadataSample result;
    result.status = Status::Found;
    result.known.resize(count);
    std::map<std::size_t, List> fields;
    for (std::size_t id = 0; id < source.nodes.size(); ++id) {
      result.node_shapes.push_back(engine.concrete(engine.shapes[id], model));
      for (std::size_t f = 0; f < source.nodes[id].fields.size(); ++f)
        fields[p.nodes[id].fields[f].output] =
            engine.concrete(engine.attrs[id][f], model);
    }
    // Captured descriptors reused on the RHS refer to the same assignment.
    for (auto& n : p.nodes)
      for (auto& f : n.fields)
        if (f.known >= 0 && fields.contains(f.output))
          result.known[f.known] = fields.at(f.output);
    return result;
  } catch (const std::exception& e) {
    return {Status::Unknown, {}, {}, e.what()};
  } catch (...) {
    return {Status::Unknown, {}, {}, "unsupported source metadata"};
  }
}
VerificationResult verifySparse(const inference::SemanticPlan& p,
                                const std::vector<List>& shapes,
                                const std::vector<List>& known,
                                const std::vector<List>& pinned,
                                std::size_t budget, std::size_t max_degree) {
  try {
    std::map<std::size_t, List> shared;
    for (const auto& node : p.nodes)
      for (const auto& field : node.fields) {
        if (field.known == -1)
          throw Limit("early verification requires explicit attributes");
        const auto& value = field.known >= 0 ? known.at(field.known)
                                             : pinned.at(-2 - field.known);
        if (shared.contains(field.output) && shared.at(field.output) != value)
          throw Limit("inconsistent shared attribute binding");
        shared[field.output] = value;
      }
    SolveOptions options;
    SolveStatistics statistics;
    EarlyEngine engine(p, shapes, known, pinned, budget, options, statistics);
    return engine.verify(max_degree);
  } catch (const std::exception& e) {
    VerificationResult out;
    out.reason = e.what();
    return out;
  } catch (...) {
    VerificationResult out;
    out.reason = "unsupported or illegal semantic indexing";
    return out;
  }
}
std::string verificationRequest(const std::string& request) {
  try {
    std::istringstream in(request);
    std::string operation;
    std::size_t budget, degree;
    unsigned timeout;
    if (!(in >> operation >> timeout >> budget >> degree) ||
        (operation != "sample-v1" && operation != "sparse-v1") || degree > 64 ||
        !timeout || timeout > 60000 || budget > 10000000)
      throw Limit("malformed verification request");
    auto p = readPlan(in);
    p.timeout_ms = timeout;
    auto readLists = [&]() {
      std::size_t count;
      if (!(in >> count) || count > 10000) throw Limit("malformed lists");
      std::vector<List> xs;
      for (std::size_t i = 0; i < count; ++i) {
        auto list = readNumbers(in);
        xs.emplace_back(list.begin(), list.end());
      }
      return xs;
    };
    auto shapes = readLists();
    std::ostringstream out;
    auto hex = [&](const std::string& text) {
      if (text.empty()) {
        out << "- ";
        return;
      }
      constexpr char digits[] = "0123456789abcdef";
      for (unsigned char c : text) out << digits[c >> 4] << digits[c & 15];
      out << ' ';
    };
    auto writeLists = [&](const std::vector<List>& xs) {
      out << xs.size() << ' ';
      for (auto& x : xs) writeNumbers(out, x);
    };
    if (operation == "sample-v1") {
      std::size_t count;
      if (!(in >> count) || count > 10000)
        throw Limit("malformed blocked samples");
      std::vector<std::vector<List>> blocked;
      for (std::size_t i = 0; i < count; ++i) blocked.push_back(readLists());
      if (!in) throw Limit("malformed sample request");
      auto result = sampleMetadata(p, shapes, blocked, budget);
      out << static_cast<int>(result.status) << ' ';
      writeLists(result.known);
      writeLists(result.node_shapes);
      hex(result.reason);
    } else {
      auto known = readLists(), pinned = readLists();
      if (!in) throw Limit("malformed sparse request");
      auto result = verifySparse(p, shapes, known, pinned, budget, degree);
      out << static_cast<int>(result.status) << ' ' << result.degree << ' ';
      writeNumbers(out, result.output);
      out << result.entries.size() << ' ';
      for (auto& entry : result.entries) {
        out << entry.capture << ' ' << entry.value << ' ';
        writeNumbers(out, entry.coordinates);
      }
      hex(result.lhs_value);
      hex(result.rhs_value);
      hex(result.reason);
    }
    return out.str();
  } catch (...) {
    return "";  // The adapter treats malformed/unavailable replies as unknown.
  }
}
}  // namespace tepl::core::semantics
