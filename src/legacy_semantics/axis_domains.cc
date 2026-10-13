#include "src/legacy_semantics/axis_domains.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

namespace tepl::core::semantics {
namespace {
using Number = std::int64_t;
using Set = std::vector<Number>;
using Id = std::size_t;
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Inconsistent {};
Set sorted(Set xs) {
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  return xs;
}
Set unite(const Set& a, const Set& b) {
  Set out;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                 std::back_inserter(out));
  return out;
}
bool overlaps(const Set& a, const Set& b) {
  auto x = a.begin(), y = b.begin();
  while (x != a.end() && y != b.end()) {
    if (*x == *y) return true;
    if (*x < *y)
      ++x;
    else
      ++y;
  }
  return false;
}
Set interval(Number low, Number high) {
  if (high < low) return {};
  // This limits materialized uncertainty, not the rank of a concrete tensor.
  if (high - low > 8192) throw Unsupported("axis domain uncertainty capacity");
  Set out(high - low + 1);
  std::iota(out.begin(), out.end(), low);
  return out;
}
struct Vec {
  Id length;
  std::vector<Id> entries;
};
struct Value {
  Id scalar;
  std::optional<Vec> vector;
  explicit Value(Id x) : scalar(x) {}
  explicit Value(Vec v) : scalar(v.length), vector(std::move(v)) {}
};
// A work-list of monotone, solution-preserving relational propagators. A
// propagator can leave unsupported deductions untouched; it cannot guess.
struct Domains {
  struct Instruction {
    std::function<void()> apply;
    bool queued = false;
  };
  const std::function<void()>& tick;
  std::vector<Set> values;
  std::vector<std::vector<Id>> users;
  std::vector<Instruction> instructions;
  std::deque<Id> pending;
  std::map<Number, Id> constants;
  std::size_t stored_values = 0;
  explicit Domains(const std::function<void()>& t) : tick(t) {}
  Id variable(Set xs) {
    if (values.size() >= 200000)
      throw Unsupported("axis domain storage budget");
    xs = sorted(std::move(xs));
    if (xs.empty()) throw Inconsistent{};
    if (xs.size() > 4000000 - stored_values)
      throw Unsupported("axis domain value storage budget");
    stored_values += xs.size();
    values.push_back(std::move(xs));
    users.emplace_back();
    return values.size() - 1;
  }
  Id constant(Number x) {
    if (constants.contains(x)) return constants.at(x);
    return constants[x] = variable({x});
  }
  const Set& get(Id x) const { return values.at(x); }
  bool fixed(Id x) const { return get(x).size() == 1; }
  Number minimum(Id x) const { return get(x).front(); }
  Number maximum(Id x) const { return get(x).back(); }
  void enqueue(Id i) {
    if (!instructions[i].queued) {
      instructions[i].queued = true;
      pending.push_back(i);
    }
  }
  void narrow(Id x, const Set& allowed) {
    Set next;
    std::set_intersection(get(x).begin(), get(x).end(), allowed.begin(),
                          allowed.end(), std::back_inserter(next));
    if (next.empty()) throw Inconsistent{};
    if (next == get(x)) return;
    stored_values -= get(x).size() - next.size();
    values[x] = std::move(next);
    for (auto i : users[x]) enqueue(i);
  }
  template <class Predicate>
  void filter(Id x, Predicate p) {
    Set next;
    for (auto v : get(x))
      if (p(v)) next.push_back(v);
    narrow(x, next);
  }
  void instruction(std::vector<Id> deps, std::function<void()> apply) {
    auto i = instructions.size();
    instructions.push_back({std::move(apply)});
    for (auto x : sortedIds(std::move(deps))) users[x].push_back(i);
    enqueue(i);
  }
  static std::vector<Id> sortedIds(std::vector<Id> xs) {
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    return xs;
  }
  static void append(std::vector<Id>& xs, const Vec& v) {
    xs.push_back(v.length);
    xs.insert(xs.end(), v.entries.begin(), v.entries.end());
  }
  void propagate() {
    while (!pending.empty()) {
      tick();
      auto i = pending.front();
      pending.pop_front();
      instructions[i].queued = false;
      // Copy: a propagator may install additional instructions in the future.
      auto apply = instructions[i].apply;
      apply();
    }
  }
  void equality(Id a, Id b) {
    if (a == b) return;
    instruction({a, b}, [=, this] {
      narrow(a, get(b));
      narrow(b, get(a));
    });
  }
  Id binary(Id a, Id b, const std::string& op) {
    auto calculate = [&](Number x, Number y) -> Number {
      __int128 z;
      if (op == "+")
        z = __int128(x) + y;
      else if (op == "-")
        z = __int128(x) - y;
      else if (op == "*")
        z = __int128(x) * y;
      else if (op == "==")
        return x == y;
      else if (op == "!=")
        return x != y;
      else if (op == "<")
        return x < y;
      else if (op == "<=")
        return x <= y;
      else if (op == ">")
        return x > y;
      else if (op == ">=")
        return x >= y;
      else
        throw Unsupported("axis propagation primitive: " + op);
      if (z < std::numeric_limits<Number>::min() ||
          z > std::numeric_limits<Number>::max())
        throw Unsupported("axis propagation arithmetic capacity");
      return Number(z);
    };
    if (fixed(a) && fixed(b))
      return constant(calculate(minimum(a), minimum(b)));
    auto supports = [&](Id out, bool install) {
      if (get(a).size() * get(b).size() > 1000000)
        throw Unsupported("axis propagation relation budget");
      Set ax, bx, cx;
      auto av = get(a), bv = get(b);
      std::size_t rows = 0;
      for (auto x : av) {
        // Charge relational work in bounded chunks. The work-list itself also
        // charges every instruction; charging every scalar value would make
        // the native runtime's existing instruction budget rank dependent
        // twice.
        if (++rows % 64 == 0) tick();
        for (auto y : bv) {
          if (a == b && x != y) continue;
          auto z = calculate(x, y);
          if (!install ||
              std::binary_search(get(out).begin(), get(out).end(), z)) {
            ax.push_back(x);
            bx.push_back(y);
            cx.push_back(z);
          }
        }
      }
      if (install) {
        narrow(a, sorted(ax));
        narrow(b, sorted(bx));
      }
      return sorted(std::move(cx));
    };
    auto c = variable(supports(0, false));
    // Keep the arithmetic implementation independent of stack-local lambdas.
    instruction({a, b, c}, [=, this] {
      Set ax, bx, cx;
      auto av = get(a), bv = get(b);
      std::size_t rows = 0;
      for (auto x : av) {
        if (++rows % 64 == 0) tick();
        for (auto y : bv) {
          if (a == b && x != y) continue;
          __int128 z = 0;
          if (op == "+")
            z = __int128(x) + y;
          else if (op == "-")
            z = __int128(x) - y;
          else if (op == "*")
            z = __int128(x) * y;
          else if (op == "==")
            z = x == y;
          else if (op == "!=")
            z = x != y;
          else if (op == "<")
            z = x < y;
          else if (op == "<=")
            z = x <= y;
          else if (op == ">")
            z = x > y;
          else if (op == ">=")
            z = x >= y;
          auto n = Number(z);
          if (std::binary_search(get(c).begin(), get(c).end(), n)) {
            ax.push_back(x);
            bx.push_back(y);
            cx.push_back(n);
          }
        }
      }
      narrow(a, sorted(ax));
      narrow(b, sorted(bx));
      narrow(c, sorted(cx));
    });
    return c;
  }
  Id active(Id length, std::size_t i) {
    return binary(constant(i), length, "<");
  }
  Vec literal(const std::vector<std::uint64_t>& xs) {
    Vec v{constant(xs.size()), {}};
    for (auto x : xs) {
      if (x > std::uint64_t(std::numeric_limits<Number>::max()))
        throw Unsupported("axis propagation unsigned metadata capacity");
      v.entries.push_back(constant(x));
    }
    return v;
  }
  Set contents(const Vec& v) {
    Set out{0};
    for (std::size_t i = 0;
         i < v.entries.size() && Number(i) < maximum(v.length); ++i)
      out = unite(out, get(v.entries[i]));
    return out;
  }
  Vec vector(Id length, std::size_t capacity, const Set& domain) {
    Vec out{length, {}};
    for (std::size_t i = 0; i < capacity; ++i)
      out.entries.push_back(variable(domain));
    return out;
  }
  std::optional<std::vector<std::uint64_t>> concrete(const Vec& v) const {
    if (!fixed(v.length)) return {};
    auto n = minimum(v.length);
    if (n < 0 || std::size_t(n) > v.entries.size()) throw Inconsistent{};
    std::vector<std::uint64_t> out;
    for (Number i = 0; i < n; ++i) {
      if (!fixed(v.entries[i])) return {};
      if (minimum(v.entries[i]) < 0)
        throw Unsupported("negative vector metadata");
      out.push_back(minimum(v.entries[i]));
    }
    return out;
  }
  void equal(const Vec& a, const Vec& b) {
    equality(a.length, b.length);
    auto deps = std::vector<Id>{};
    append(deps, a);
    append(deps, b);
    instruction(deps, [=, this] {
      filter(a.length, [&](auto n) {
        return n >= 0 &&
               std::size_t(n) <= std::min(a.entries.size(), b.entries.size());
      });
      narrow(b.length, get(a.length));
      for (Number i = 0; i < maximum(a.length); ++i) {
        if (!overlaps(get(a.entries[i]), get(b.entries[i]))) {
          filter(a.length, [&](auto n) { return n <= i; });
          continue;
        }
        if (i < minimum(a.length)) {
          narrow(a.entries[i], get(b.entries[i]));
          narrow(b.entries[i], get(a.entries[i]));
        }
      }
    });
  }
  // value = source[index] when guard is true. Enumerate compatible positions,
  // never assignments to complete vectors. Optional entries remain untouched.
  void route(const Vec& source, Id index, Id value, Id guard,
             bool slot_value = false, bool slot_index = false) {
    std::vector<Id> deps{index, value, guard};
    append(deps, source);
    instruction(deps, [=, this] {
      if (maximum(guard) == 0) return;
      Set positions, possible;
      for (auto p : get(index)) {
        if (p < 0 || std::size_t(p) >= source.entries.size() ||
            p >= maximum(source.length))
          continue;
        if (!overlaps(get(value), get(source.entries[p]))) continue;
        positions.push_back(p);
        possible = unite(possible, get(source.entries[p]));
      }
      if (positions.empty()) {
        narrow(guard, {0});
        return;
      }
      // Entry domains describe the value WHEN its vector slot is active. It is
      // safe to narrow such a slot before its length is known. Length-derived
      // scalar indices, in contrast, must retain their inactive alternatives.
      if (slot_value) narrow(value, possible);
      if (slot_index) narrow(index, positions);
      if (minimum(guard) == 0) return;
      narrow(index, positions);
      narrow(value, possible);
      filter(source.length, [&](auto n) { return n > minimum(index); });
      if (fixed(index)) narrow(source.entries[minimum(index)], get(value));
    });
  }
  void valid(const Vec& v, Id bound) {
    std::vector<Id> deps{bound};
    append(deps, v);
    instruction(deps, [=, this] {
      filter(v.length, [&](auto n) {
        return n >= 0 && n <= maximum(bound) &&
               std::size_t(n) <= v.entries.size();
      });
      filter(bound, [&](auto n) { return n >= minimum(v.length); });
      std::set<Number> forced;
      Set prefix;
      for (Number i = 0; i < maximum(v.length); ++i) {
        Set choices;
        for (auto x : get(v.entries[i]))
          if (x >= 0 && x < maximum(bound)) choices.push_back(x);
        prefix = unite(prefix, choices);
        // Hall's necessary condition on each prefix. This also detects repeated
        // singleton domains before any of those slots is mandatory.
        if (prefix.size() < std::size_t(i + 1)) {
          filter(v.length, [&](auto n) { return n <= i; });
          break;
        }
      }
      for (Number i = 0; i < minimum(v.length); ++i) {
        filter(v.entries[i],
               [&](auto x) { return x >= 0 && x < maximum(bound); });
        filter(bound, [&](auto n) { return n > minimum(v.entries[i]); });
        if (fixed(v.entries[i]) && !forced.insert(minimum(v.entries[i])).second)
          throw Inconsistent{};
      }
      for (Number i = 0; i < minimum(v.length); ++i)
        if (!fixed(v.entries[i]))
          filter(v.entries[i], [&](auto x) { return !forced.contains(x); });
      for (Number i = minimum(v.length); i < maximum(v.length); ++i) {
        bool possible = false;
        for (auto x : get(v.entries[i]))
          possible |= x >= 0 && x < maximum(bound) && !forced.contains(x);
        if (!possible) {
          filter(v.length, [&](auto n) { return n <= i; });
          break;
        }
      }
    });
  }
  void disjoint(const Vec& a, const Vec& b) {
    std::vector<Id> deps;
    append(deps, a);
    append(deps, b);
    instruction(deps, [=, this] {
      for (auto [x, y] : {std::pair(a, b), std::pair(b, a)}) {
        std::set<Number> forced;
        for (Number i = 0; i < minimum(x.length); ++i)
          if (fixed(x.entries[i])) forced.insert(minimum(x.entries[i]));
        for (Number i = 0; i < maximum(y.length); ++i) {
          if (i < minimum(y.length))
            filter(y.entries[i], [&](auto n) { return !forced.contains(n); });
          else if (std::all_of(get(y.entries[i]).begin(),
                               get(y.entries[i]).end(),
                               [&](auto n) { return forced.contains(n); }))
            filter(y.length, [&](auto n) { return n <= i; });
        }
      }
    });
  }
  Vec gather(const Vec& v, const Vec& axes) {
    if (auto a = concrete(axes)) {
      Vec out{axes.length, {}};
      for (auto i : *a) {
        if (i >= v.entries.size()) throw Inconsistent{};
        filter(v.length, [&](auto n) { return n > Number(i); });
        out.entries.push_back(v.entries[i]);
      }
      return out;
    }
    auto out = vector(axes.length, axes.entries.size(), contents(v));
    for (std::size_t i = 0; i < out.entries.size(); ++i)
      route(v, axes.entries[i], out.entries[i], active(out.length, i), true,
            true);
    return out;
  }
  Vec take(const Vec& v, Id start, Id count) {
    narrow(count, interval(0, v.entries.size()));
    auto end = binary(start, count, "+");
    equality(binary(end, v.length, "<="), constant(1));
    equality(binary(start, constant(0), ">="), constant(1));
    if (fixed(start) && fixed(count)) {
      auto first = minimum(start), size = minimum(count);
      if (first < 0 || size < 0 || std::size_t(first + size) > v.entries.size())
        throw Inconsistent{};
      return {count, std::vector<Id>(v.entries.begin() + first,
                                     v.entries.begin() + first + size)};
    }
    auto out = vector(count, maximum(count), contents(v));
    for (std::size_t i = 0; i < out.entries.size(); ++i)
      route(v, binary(start, constant(i), "+"), out.entries[i],
            active(count, i), true);
    return out;
  }
  Vec concat(const Vec& a, const Vec& b) {
    auto length = binary(a.length, b.length, "+");
    if (fixed(a.length) && fixed(b.length)) {
      if (minimum(a.length) < 0 || minimum(b.length) < 0 ||
          std::size_t(minimum(a.length)) > a.entries.size() ||
          std::size_t(minimum(b.length)) > b.entries.size())
        throw Inconsistent{};
      Vec out{length, {}};
      out.entries.insert(out.entries.end(), a.entries.begin(),
                         a.entries.begin() + minimum(a.length));
      out.entries.insert(out.entries.end(), b.entries.begin(),
                         b.entries.begin() + minimum(b.length));
      return out;
    }
    auto out = vector(length, maximum(length), unite(contents(a), contents(b)));
    for (std::size_t i = 0; i < a.entries.size(); ++i)
      route(out, constant(i), a.entries[i], active(a.length, i), true);
    for (std::size_t i = 0; i < b.entries.size(); ++i)
      route(out, binary(a.length, constant(i), "+"), b.entries[i],
            active(b.length, i), true);
    // Forward support prevents an unconstrained output slot from hiding a
    // contradiction. Start/length correlations may be overapproximated safely.
    std::vector<Id> deps;
    append(deps, a);
    append(deps, b);
    append(deps, out);
    instruction(deps, [=, this] {
      for (Number i = 0; i < minimum(out.length); ++i) {
        Set support;
        if (i < maximum(a.length) && std::size_t(i) < a.entries.size())
          support = get(a.entries[i]);
        for (auto start : get(a.length)) {
          auto j = i - start;
          if (j >= 0 && j < maximum(b.length) &&
              std::size_t(j) < b.entries.size())
            support = unite(support, get(b.entries[j]));
        }
        narrow(out.entries[i], support);
      }
    });
    return out;
  }
  Vec scatter(Id size, const Vec& axes, const Vec& vals) {
    narrow(size, interval(0, 8192));
    equality(axes.length, size);
    equality(vals.length, size);
    valid(axes, size);
    if (auto a = concrete(axes); a && fixed(size)) {
      if (a->size() != std::size_t(minimum(size)) ||
          vals.entries.size() < a->size())
        throw Inconsistent{};
      Vec out{size, std::vector<Id>(a->size(), constant(0))};
      for (Id j = 0; j < a->size(); ++j) {
        if ((*a)[j] >= a->size()) throw Inconsistent{};
        out.entries[(*a)[j]] = vals.entries[j];
      }
      return out;
    }
    auto out = vector(size, maximum(size), contents(vals));
    for (std::size_t i = 0;
         i < std::min(axes.entries.size(), vals.entries.size()); ++i)
      route(out, axes.entries[i], vals.entries[i], active(size, i), true, true);
    return out;
  }
  Id member(const Vec& v, Id item) {
    auto out = variable({0, 1});
    std::vector<Id> deps{item, out};
    append(deps, v);
    instruction(deps, [=, this] {
      // Variable membership is left to SMT until the searched value is fixed.
      if (!fixed(item)) return;
      auto x = minimum(item);
      std::vector<Number> positions;
      bool forced = false;
      for (Number i = 0; i < maximum(v.length); ++i)
        if (std::binary_search(get(v.entries[i]).begin(),
                               get(v.entries[i]).end(), x)) {
          positions.push_back(i);
          forced |= i < minimum(v.length) && fixed(v.entries[i]);
        }
      if (forced) narrow(out, {1});
      if (positions.empty()) narrow(out, {0});
      if (!fixed(out)) return;
      if (minimum(out) == 0) {
        for (auto i : positions) {
          if (i < minimum(v.length))
            filter(v.entries[i], [&](auto n) { return n != x; });
          else if (fixed(v.entries[i]))
            filter(v.length, [&](auto n) { return n <= i; });
        }
      } else if (positions.size() == 1) {
        auto i = positions.front();
        filter(v.length, [&](auto n) { return n > i; });
        narrow(v.entries[i], {x});
      }
    });
    return out;
  }
  Vec exclude(const Vec& v, const Vec& axes) {
    auto source = concrete(v), selected = concrete(axes);
    if (source && selected) {
      std::set<std::uint64_t> removed(selected->begin(), selected->end());
      std::erase_if(*source, [&](auto x) { return removed.contains(x); });
      return literal(*source);
    }
    auto out = vector(variable(interval(0, maximum(v.length))),
                      maximum(v.length), contents(v));
    auto prefix = constant(0);
    for (std::size_t i = 0; i < v.entries.size(); ++i) {
      auto absent = binary(member(axes, v.entries[i]), constant(0), "==");
      auto keep = binary(active(v.length, i), absent, "*");
      route(out, prefix, v.entries[i], keep);
      prefix = binary(prefix, keep, "+");
    }
    equality(out.length, prefix);
    return out;
  }
};

struct Interpreter {
  const inference::SemanticPlan& plan;
  Domains d;
  std::vector<Vec> shapes;
  std::vector<std::vector<Vec>> attrs;
  std::map<Id, Vec> fields;
  std::map<std::tuple<Id, Id, Id>, Value> cache;
  using Environment = std::map<std::string, Vec>;
  Id context = 0;
  Interpreter(const inference::SemanticPlan& p,
              const std::function<void()>& tick)
      : plan(p), d(tick) {}
  Vec vec(const Value& v) {
    if (!v.vector) throw Unsupported("axis propagation requires a vector");
    return *v.vector;
  }
  Value eval(Id node, Id id, const Environment& env = {}, Id scope = 0) {
    auto key = std::tuple(node, id, scope);
    if (cache.contains(key)) return cache.at(key);
    d.tick();
    auto value = evaluate(node, id, env, scope);
    cache.emplace(key, value);
    return value;
  }
  Value evaluate(Id node, Id id, const Environment& env, Id scope) {
    const auto& es = node == plan.nodes.size()
                         ? plan.constraints.expressions
                         : plan.nodes.at(node).expressions;
    const auto& e = es.at(id);
    auto arg = [&](Id i) { return eval(node, e.args.at(i), env, scope); };
    if (e.args.empty()) {
      if (env.contains(e.op)) return Value(env.at(e.op));
      if (e.op.starts_with("@shape:"))
        return Value(shapes.at(std::stoul(e.op.substr(7))));
      if (e.op.starts_with("@attr:")) {
        auto colon = e.op.find(':', 6);
        return Value(attrs.at(std::stoul(e.op.substr(6, colon - 6)))
                         .at(std::stoul(e.op.substr(colon + 1))));
      }
      if (e.op.starts_with('a'))
        return Value(attrs.at(node).at(std::stoul(e.op.substr(1))));
      if (e.op == "true" || e.op == "false")
        return Value(d.constant(e.op == "true"));
      if (e.op == "list") return Value(d.literal({}));
      if (!e.op.empty() &&
          e.op.find_first_not_of("0123456789") == std::string::npos)
        return Value(d.constant(std::stoll(e.op)));
      throw Unsupported("axis propagation unbound expression: " + e.op);
    }
    if (e.op == "shape" || e.op == "rank") {
      const auto& n = plan.nodes.at(node);
      auto tensor = n.expressions.at(e.args[0]).op;
      if (!tensor.starts_with('t'))
        throw Unsupported("axis propagation shape operand");
      auto shape = shapes.at(n.inputs.at(std::stoul(tensor.substr(1))));
      return e.op == "shape" ? Value(shape) : Value(shape.length);
    }
    if (e.op == "len") return Value(vec(arg(0)).length);
    if (e.op == "index_space") return arg(0);
    if (e.op == "list") {
      Vec v{d.constant(e.args.size()), {}};
      for (Id i = 0; i < e.args.size(); ++i) v.entries.push_back(arg(i).scalar);
      return Value(v);
    }
    if (e.op == "range") {
      auto n = arg(0).scalar;
      d.narrow(n, interval(0, 8192));
      Vec v{n, {}};
      for (Number i = 0; i < d.maximum(n); ++i)
        v.entries.push_back(d.constant(i));
      return Value(v);
    }
    if (e.op == "gather") return Value(d.gather(vec(arg(0)), vec(arg(1))));
    if (e.op == "exclude") return Value(d.exclude(vec(arg(0)), vec(arg(1))));
    if (e.op == "concat") {
      auto out = vec(arg(0));
      for (Id i = 1; i < e.args.size(); ++i) out = d.concat(out, vec(arg(i)));
      return Value(out);
    }
    if (e.op == "take" || e.op == "slice")
      return Value(d.take(vec(arg(0)), arg(1).scalar, arg(2).scalar));
    if (e.op == "scatter")
      return Value(d.scatter(arg(0).scalar, vec(arg(1)), vec(arg(2))));
    if (e.op == "index") {
      auto v = vec(arg(0));
      auto out = d.variable(d.contents(v));
      d.route(v, arg(1).scalar, out, d.constant(1));
      return Value(out);
    }
    if (e.op == "==") {
      auto a = arg(0), b = arg(1);
      if (a.vector && b.vector) {
        throw Unsupported("vector comparison outside an asserted equality");
      }
      return Value(d.binary(a.scalar, b.scalar, e.op));
    }
    if (e.op == "if" || e.op == "&&" || e.op == "||")
      throw Unsupported("axis propagation guarded metadata");
    if (e.op == "unary:+") return arg(0);
    if (e.op == "unary:-")
      return Value(d.binary(d.constant(0), arg(0).scalar, "-"));
    if (e.op == "unary:!")
      return Value(d.binary(arg(0).scalar, d.constant(0), "=="));
    return Value(d.binary(arg(0).scalar, arg(1).scalar, e.op));
  }
  // Only asserted Boolean expressions are evaluated as relational obligations.
  // Equality and validity used inside arbitrary Boolean expressions must not
  // become unconditional restrictions.
  void assertion(Id node, Id id) {
    const auto& e = plan.nodes.at(node).expressions.at(id);
    if (e.op == "==") {
      auto a = eval(node, e.args.at(0)), b = eval(node, e.args.at(1));
      if (a.vector && b.vector)
        d.equal(*a.vector, *b.vector);
      else
        d.equality(a.scalar, b.scalar);
    } else if (e.op == "is_valid_axis_list") {
      d.valid(vec(eval(node, e.args.at(0))), eval(node, e.args.at(1)).scalar);
    } else if (e.op == "is_disjoint") {
      d.disjoint(vec(eval(node, e.args.at(0))), vec(eval(node, e.args.at(1))));
    } else if (e.op == "<" || e.op == "<=" || e.op == ">" || e.op == ">=" ||
               e.op == "true")
      d.equality(eval(node, id).scalar, d.constant(1));
    else
      throw Unsupported("axis propagation assertion fragment");
  }
  void metadata(const std::vector<std::vector<std::uint64_t>>& captures,
                const std::vector<std::vector<std::uint64_t>>& known,
                const std::vector<std::vector<std::uint64_t>>& pinned) {
    for (Id node = 0; node < plan.nodes.size(); ++node) {
      const auto& n = plan.nodes[node];
      attrs.emplace_back();
      if (n.capture >= 0) {
        shapes.push_back(d.literal(captures.at(n.capture)));
        continue;
      }
      for (const auto& f : n.fields)
        attrs.back().push_back(d.literal(f.known >= 0 ? known.at(f.known)
                                         : f.known < -1
                                             ? pinned.at(-2 - f.known)
                                             : std::vector<std::uint64_t>{}));
      std::set<Id> installed, active;
      std::function<void(Id)> install = [&](Id j) {
        const auto& f = n.fields[j];
        if (f.known != -1 || installed.contains(j)) return;
        if (!active.insert(j).second)
          throw Unsupported("axis propagation cyclic attribute bounds");
        if (!fields.contains(f.output)) {
          std::set<Id> visited;
          std::function<void(Id)> dependencies = [&](Id id) {
            if (!visited.insert(id).second) return;
            const auto& e = n.expressions.at(id);
            if (e.args.empty() && e.op.starts_with('a'))
              install(std::stoul(e.op.substr(1)));
            for (auto a : e.args) dependencies(a);
          };
          dependencies(f.bound);
          auto bound = eval(node, f.bound).scalar;
          auto high = d.maximum(bound);
          auto v = d.vector(d.variable(interval(0, high)), high,
                            interval(0, std::max<Number>(0, high - 1)));
          d.valid(v, bound);
          fields.emplace(f.output, v);
        }
        attrs[node][j] = fields.at(f.output);
        cache.clear();
        installed.insert(j);
        active.erase(j);
      };
      for (Id j = 0; j < n.fields.size(); ++j) install(j);
      for (auto a : n.assertions) assertion(node, a);
      shapes.push_back(vec(eval(node, n.shape)));
      // Source metadata must be concrete before logical coordinate identities
      // are assigned. RHS shapes remain domains.
      if (node + 1 == plan.source_end) d.propagate();
    }
    d.equal(shapes.at(plan.source_root), shapes.at(plan.rhs_root));
    d.propagate();
  }
  static bool compatible(const inference::ConstraintTerm& a,
                         const inference::ConstraintTerm& b) {
    if (a.coefficient != b.coefficient || a.reads.size() != b.reads.size())
      return false;
    for (Id i = 0; i < a.reads.size(); ++i)
      if (a.reads[i].capture != b.reads[i].capture) return false;
    return true;
  }
  void matching() {
    const auto& c = plan.constraints;
    if (c.lhs.size() != c.rhs.size()) throw Inconsistent{};
    std::vector<Id> pairing;
    std::set<Id> used;
    for (const auto& lhs : c.lhs) {
      std::vector<Id> candidates;
      for (Id j = 0; j < c.rhs.size(); ++j)
        if (compatible(lhs, c.rhs[j])) candidates.push_back(j);
      if (candidates.empty()) throw Inconsistent{};
      if (candidates.size() != 1)
        throw Unsupported("ambiguous term correspondence retained for SMT");
      if (!used.insert(candidates[0]).second) throw Inconsistent{};
      for (Id i = 1; i < lhs.reads.size(); ++i)
        if (lhs.reads[i - 1].capture == lhs.reads[i].capture)
          throw Unsupported("repeated-factor correspondence retained for SMT");
      pairing.push_back(candidates[0]);
    }
    auto output = d.concrete(shapes.at(plan.source_root));
    if (!output) throw Unsupported("nonconcrete source metadata");
    std::vector<std::uint64_t> free(output->size());
    std::iota(free.begin(), free.end(), 0);
    const auto node = plan.nodes.size();
    for (Id t = 0; t < c.lhs.size(); ++t) {
      const auto& lhs = c.lhs[t];
      const auto& rhs = c.rhs[pairing[t]];
      Environment left{{"@output", d.literal(free)}}, right = left;
      std::vector<std::uint64_t> extents;
      auto lscope = ++context;
      for (auto binder : lhs.reductions) {
        auto dims = vec(eval(node, c.domains.at(binder), left, lscope));
        d.propagate();
        auto concrete = d.concrete(dims);
        if (!concrete) throw Unsupported("nonconcrete source reduction domain");
        if (std::find(lhs.unused_reductions.begin(),
                      lhs.unused_reductions.end(),
                      binder) != lhs.unused_reductions.end() &&
            std::all_of(concrete->begin(), concrete->end(),
                        [](auto n) { return n == 1; }))
          continue;
        std::vector<std::uint64_t> labels(concrete->size());
        std::iota(labels.begin(), labels.end(), free.size() + extents.size());
        left.emplace("@bound:" + std::to_string(binder), d.literal(labels));
        extents.insert(extents.end(), concrete->begin(), concrete->end());
      }
      // The same unknown reduction bijection as the SMT matcher, represented
      // by logical labels instead of a dense integer permutation expression.
      auto rscope = ++context;
      auto labels = d.literal({}), dimensions = d.literal({});
      for (auto binder : rhs.reductions) {
        auto dims = vec(eval(node, c.domains.at(binder), right, rscope));
        if (std::find(rhs.unused_reductions.begin(),
                      rhs.unused_reductions.end(),
                      binder) != rhs.unused_reductions.end())
          throw Unsupported("optional unused RHS reduction retained for SMT");
        if (extents.empty()) d.equality(dims.length, d.constant(0));
        auto domain =
            extents.empty()
                ? Set{Number(free.size())}
                : interval(Number(free.size()),
                           Number(free.size()) + Number(extents.size()) - 1);
        auto bound = d.vector(dims.length, dims.entries.size(), domain);
        Vec local_positions{bound.length, {}};
        for (auto label : bound.entries)
          local_positions.entries.push_back(
              d.binary(label, d.constant(free.size()), "-"));
        d.valid(local_positions, d.constant(extents.size()));
        right.emplace("@bound:" + std::to_string(binder), bound);
        labels = d.concat(labels, bound);
        dimensions = d.concat(dimensions, dims);
      }
      d.equality(labels.length, d.constant(extents.size()));
      // Convert labels to zero-based positions for validity and extent routing.
      Vec positions{labels.length, {}};
      for (auto label : labels.entries)
        positions.entries.push_back(
            d.binary(label, d.constant(free.size()), "-"));
      d.valid(positions, d.constant(extents.size()));
      d.equal(dimensions, d.gather(d.literal(extents), positions));
      for (Id i = 0; i < lhs.reads.size(); ++i)
        d.equal(vec(eval(node, lhs.reads[i].indices, left, lscope)),
                vec(eval(node, rhs.reads[i].indices, right, rscope)));
      d.propagate();
    }
  }
  AxisPropagation result() {
    AxisPropagation out;
    out.available = true;
    out.instructions = d.instructions.size();
    for (const auto& [id, v] : fields) {
      AxisDomain field;
      for (auto n : d.get(v.length)) field.lengths.push_back(n);
      auto size = d.maximum(v.length);
      for (Number i = 0; i < size; ++i) {
        std::vector<std::uint64_t> entry;
        for (auto n : d.get(v.entries[i])) {
          if (n < 0) throw Unsupported("negative inferred axis");
          entry.push_back(n);
        }
        field.entries.push_back(std::move(entry));
        if (i < d.minimum(v.length) && d.fixed(v.entries[i]))
          ++out.forced_entries;
        else
          ++out.residual_entries;
      }
      out.original_entries += v.entries.size();
      out.fields.emplace(id, std::move(field));
    }
    return out;
  }
};
}  // namespace
AxisPropagation propagateAxes(
    const inference::SemanticPlan& plan,
    const std::vector<std::vector<std::uint64_t>>& shapes,
    const std::vector<std::vector<std::uint64_t>>& known,
    const std::vector<std::vector<std::uint64_t>>& pinned,
    const std::function<void()>& tick) {
  AxisPropagation out;
  if (!plan.algebraic || !plan.constraints.available) {
    out.reason = "no algebraic constraint plan";
    return out;
  }
  try {
    Interpreter i(plan, tick);
    i.metadata(shapes, known, pinned);
    i.matching();
    return i.result();
  } catch (const Unsupported& e) {
    out.reason = e.what();
  } catch (const Inconsistent&) {
    out.inconsistent = true;
    out.reason =
        "sufficient axis relations inconsistent; unrestricted fallback "
        "required";
  }
  return out;
}
}  // namespace tepl::core::semantics
