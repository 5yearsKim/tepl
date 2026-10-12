#include "src/core/index_semantics/expr.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <tuple>
namespace tepl::core::index_semantics {
namespace {
std::string intern(const Symbol& s) {
  std::string signature = std::to_string(static_cast<int>(s.kind)) + ":" +
                          std::to_string(static_cast<int>(s.sort)) + ":";
  if (s.kind == Symbol::Kind::Field || s.kind == Symbol::Kind::Hole ||
      s.kind == Symbol::Kind::Shape || s.kind == Symbol::Kind::DType ||
      s.kind == Symbol::Kind::AxisGroup)
    signature += std::to_string(s.owner) + ":" + std::to_string(s.index);
  else
    signature += s.name;
  if (s.kind != Symbol::Kind::AxisGroup)
    for (const auto& a : s.args) signature += "{" + key(a) + "}";
  static std::mutex mutex;
  static std::map<std::string, std::size_t> identities;
  std::lock_guard<std::mutex> lock(mutex);
  auto [it, added] =
      identities.emplace(std::move(signature), identities.size() + 1);
  return "id" + std::to_string(it->second);
}
}  // namespace
S make(Symbol::Kind k, std::string name, std::size_t owner, std::size_t index,
       std::vector<S> args) {
  if (k == Symbol::Kind::AxisGroup && args.size() == 1 &&
      args[0]->kind == Symbol::Kind::List && args[0]->args.empty())
    return call("empty_axes", {});
  Symbol::Sort sort = Symbol::Sort::Positions;
  if (k == Symbol::Kind::Integer) sort = Symbol::Sort::Integer;
  if (k == Symbol::Kind::Boolean) sort = Symbol::Sort::Boolean;
  if (k == Symbol::Kind::Shape) sort = Symbol::Sort::Extents;
  if (k == Symbol::Kind::DType) sort = Symbol::Sort::DType;
  if (k == Symbol::Kind::AxisGroup) sort = Symbol::Sort::Axes;
  if (k == Symbol::Kind::Call) {
    if (name == "len" || name == "+" || name == "-" || name == "*" ||
        name == "index")
      sort = Symbol::Sort::Integer;
    else if (name == "==" || name == "!=" || name == "&&" || name == "||" ||
             name == "!" || name == "<" || name == "<=" || name == ">" ||
             name == ">=" || name == "axis_partition" ||
             name == "is_valid_axis_list" || name == "same_set" ||
             name == "disjoint" || name == "subset" || name.starts_with("is_"))
      sort = Symbol::Sort::Boolean;
    else if (name == "axis_extents")
      sort = Symbol::Sort::Extents;
    else if (name == "distinct_axes")
      sort = Symbol::Sort::Boolean;
    else if (name == "concat" || name == "gather" ||
             name == "ordered_complement" || name == "filter_axes" ||
             name == "difference_axes" || name == "union_axes" ||
             name == "intersection_axes" || name == "replace_axes" ||
             name == "unique_axes")
      sort = args.empty() ? Symbol::Sort::Axes : args[0]->sort;
    else if (name == "empty_axes")
      sort = Symbol::Sort::Axes;
    else if (name.starts_with("dtype_"))
      sort = Symbol::Sort::DType;
    else if (name == "if")
      sort = args.at(1)->sort;
  }
  Symbol node{k, std::move(name), owner, index, std::move(args), sort};
  node.structural_key = intern(node);
  return std::make_shared<Symbol>(std::move(node));
}
S integer(std::size_t n) {
  return make(Symbol::Kind::Integer, std::to_string(n));
}
S boolean(bool b) { return make(Symbol::Kind::Boolean, b ? "true" : "false"); }
std::string key(const S& s) {
  return s->structural_key.empty() ? intern(*s) : s->structural_key;
}
bool equal(const S& a, const S& b) { return key(a) == key(b); }
bool callIs(const S& s, const std::string& name) {
  return s->kind == Symbol::Kind::Call && s->name == name;
}
bool containsHole(const S& s) {
  if (s->kind == Symbol::Kind::Hole) return true;
  for (const auto& a : s->args)
    if (containsHole(a)) return true;
  return false;
}
S call(std::string name, std::vector<S> args) {
  auto empty = [](const S& s) {
    return (s->kind == Symbol::Kind::List && s->args.empty()) ||
           callIs(s, "empty_axes");
  };
  if (name == "len" && callIs(args[0], "concat")) {
    auto n = integer(0);
    for (const auto& a : args[0]->args) n = call("+", {n, call("len", {a})});
    return n;
  }
  if (name == "unique_axes" && args[0]->kind == Symbol::Kind::AxisGroup)
    return args[0];
  if (name == "unique_axes" && callIs(args[0], "concat")) {
    std::vector<S> xs;
    for (const auto& a : args[0]->args)
      if (std::none_of(xs.begin(), xs.end(),
                       [&](const S& x) { return equal(x, a); }))
        xs.push_back(a);
    if (xs.size() != args[0]->args.size())
      return call("unique_axes", {call("concat", std::move(xs))});
  }
  if (name == "+" && args[0]->kind == Symbol::Kind::Integer &&
      args[0]->name == "0")
    return args[1];
  if (name == "+" && args[1]->kind == Symbol::Kind::Integer &&
      args[1]->name == "0")
    return args[0];
  if (name == "offset_positions" && args[1]->kind == Symbol::Kind::Integer &&
      args[1]->name == "0")
    return args[0];
  if (name == "inverse" && callIs(args[0], "inverse")) return args[0]->args[0];
  if (name == "inverse" && callIs(args[0], "range")) return args[0];
  if (name == "len" && args[0]->kind == Symbol::Kind::AxisGroup)
    return call("len", {args[0]->args[0]});
  if (name == "len" && callIs(args[0], "inverse"))
    return call("len", {args[0]->args[0]});
  if (name == "len" && callIs(args[0], "empty_axes")) return integer(0);
  if ((name == "filter_axes" || name == "intersection_axes") &&
      equal(args[0], args[1]))
    return args[0];
  if ((name == "difference_axes") && equal(args[0], args[1]))
    return call("empty_axes", {});
  if ((name == "filter_axes" || name == "intersection_axes") &&
      (empty(args[0]) || empty(args[1])))
    return call("empty_axes", {});
  if (name == "difference_axes" && (empty(args[0]) || empty(args[1])))
    return args[0];
  if ((name == "same_set" || name == "subset") && equal(args[0], args[1]))
    return boolean(true);
  if (name == "disjoint" && (empty(args[0]) || empty(args[1])))
    return boolean(true);
  if (name == "union_axes") {
    std::vector<S> out;
    for (const auto& a : args) {
      if (empty(a)) continue;
      auto parts = callIs(a, "union_axes") ? a->args : std::vector<S>{a};
      for (const auto& part : parts)
        if (std::none_of(out.begin(), out.end(),
                         [&](const S& x) { return equal(x, part); }))
          out.push_back(part);
    }
    if (out.empty()) return call("empty_axes", {});
    if (out.size() == 1) return out[0];
    std::sort(out.begin(), out.end(),
              [](const S& a, const S& b) { return key(a) < key(b); });
    args = std::move(out);
  }
  if ((name == "is_valid_axis_list" || name == "axis_partition") &&
      callIs(args[0], "range") && equal(args[0]->args[0], args[1]))
    return boolean(true);
  if (name == "concat") {
    std::vector<S> flat;
    for (const auto& a : args) {
      if (callIs(a, "concat"))
        flat.insert(flat.end(), a->args.begin(), a->args.end());
      else if (!empty(a))
        flat.push_back(a);
    }
    if (flat.empty())
      return !args.empty() && args[0]->sort == Symbol::Sort::Axes
                 ? call("empty_axes", {})
                 : make(Symbol::Kind::List);
    if (flat.size() == 1) return flat[0];
    args = std::move(flat);
  }
  if (name == "len" && args[0]->kind == Symbol::Kind::List)
    return integer(args[0]->args.size());
  if (name == "len" && callIs(args[0], "range")) return args[0]->args[0];
  if (name == "len" && callIs(args[0], "gather"))
    return call("len", {args[0]->args[1]});
  if (name == "gather") {
    const auto& s = args[0];
    const auto& p = args[1];
    if (callIs(p, "positions") && equal(s, p->args[0])) return p->args[1];
    if (callIs(s, "inverse") && callIs(p, "gather") &&
        equal(s->args[0], p->args[0]))
      return p->args[1];
    if (callIs(p, "gather") && callIs(p->args[0], "inverse") &&
        equal(s, p->args[0]->args[0]))
      return p->args[1];
    if (empty(p))
      return s->sort == Symbol::Sort::Axes ? call("empty_axes", {})
                                           : make(Symbol::Kind::List);
    if (callIs(s, "gather") && callIs(p, "inverse") &&
        equal(s->args[1], p->args[0]))
      return s->args[0];
    if (callIs(p, "range") && equal(p->args[0], call("len", {s}))) return s;
    if (callIs(s, "range")) return p;
    if (callIs(s, "gather"))
      return call("gather", {s->args[0], call("gather", {s->args[1], p})});
    if (p->kind == Symbol::Kind::List && p->args.empty()) return p;
  }
  if (name == "==" && equal(args[0], args[1])) return boolean(true);
  return make(Symbol::Kind::Call, std::move(name), 0, 0, std::move(args));
}
bool truth(const S& s) {
  return s->kind == Symbol::Kind::Boolean && s->name == "true";
}

void Context::sourceScope(std::size_t id) {
  tick();
  const auto& n = plan.nodes.at(id);
  source_nodes.insert(id);
  if (n.capture) source_captures.insert(n.capture->value);
  if (n.descriptor && n.definition && !n.automatic)
    source_fields[n.descriptor->value] = n.definition->fields.size();
  for (auto child : n.inputs) sourceScope(child);
  if (id == plan.lhs)
    for (const auto& assumption : plan.assumptions) {
      validateSource(assumption);
      learn(assumption);
    }
}
void Context::validateSource(const S& s) {
  tick();
  if (s && validated_scope.contains(s.get())) return;
  if (!s || s->kind == Symbol::Kind::Hole)
    throw std::runtime_error(
        "constructor or guard contains an unresolved expression");
  if (s->kind == Symbol::Kind::Field &&
      (!source_fields.contains(s->owner) ||
       s->index >= source_fields.at(s->owner)))
    throw std::runtime_error(
        "constructor or guard references a field outside the LHS");
  if ((s->kind == Symbol::Kind::Shape || s->kind == Symbol::Kind::DType) &&
      !source_captures.contains(s->owner))
    throw std::runtime_error(
        "constructor or guard references a tensor outside the LHS");
  if (s->kind == Symbol::Kind::AxisGroup && !source_nodes.contains(s->owner))
    throw std::runtime_error("coordinate expression is outside the LHS frame");
  for (const auto& a : s->args) validateSource(a);
  validated_scope[s.get()] = s;
}
void Context::wellFormed(const S& s) {
  tick();
  if (s && validated_expressions.contains(s.get())) return;
  validateSource(s);
  for (const auto& a : s->args) wellFormed(a);
  if (s->kind == Symbol::Kind::AxisGroup) {
    auto found = coordinate_groups.find({s->owner, s->index});
    if (found == coordinate_groups.end() || s->args.size() != 1)
      throw std::runtime_error("unregistered source coordinate family");
    auto domain = found->second->kind == Symbol::Kind::AxisGroup
                      ? found->second->args[0]
                      : make(Symbol::Kind::List);
    require(call("==", {s->args[0], domain}));
  }
  if (s->kind != Symbol::Kind::Call) {
    validated_expressions[s.get()] = s;
    return;
  }
  static const std::set<std::string> unary{
      "len",           "range",        "inverse", "unique_axes",
      "distinct_axes", "axis_extents", "!"};
  static const std::set<std::string> binary{"gather",
                                            "offset_positions",
                                            "ordered_complement",
                                            "filter_axes",
                                            "intersection_axes",
                                            "difference_axes",
                                            "positions",
                                            "same_set",
                                            "disjoint",
                                            "subset",
                                            "axis_partition",
                                            "is_valid_axis_list",
                                            "index",
                                            "+",
                                            "-",
                                            "*",
                                            "==",
                                            "!=",
                                            "<",
                                            "<=",
                                            ">",
                                            ">=",
                                            "&&",
                                            "||"};
  static const std::set<std::string> ternary{"replace_axes", "slice", "if"};
  if ((unary.contains(s->name) && s->args.size() != 1) ||
      (binary.contains(s->name) && s->args.size() != 2) ||
      (ternary.contains(s->name) && s->args.size() != 3))
    throw std::runtime_error("invalid metadata operator arity");
  if (!unary.contains(s->name) && !binary.contains(s->name) &&
      !ternary.contains(s->name) && s->name != "concat" &&
      s->name != "union_axes" && s->name != "empty_axes" &&
      !s->name.starts_with("dtype_") && !builtins::resolveBuiltin(s->name))
    throw std::runtime_error("undefined metadata operator: " + s->name);
  if (s->name == "empty_axes" && !s->args.empty())
    throw std::runtime_error("invalid empty-axis constructor");
  if (s->name == "inverse")
    require(
        call("is_valid_axis_list", {s->args[0], call("len", {s->args[0]})}));
  if (s->name == "gather")
    require(
        call("is_valid_axis_list", {s->args[1], call("len", {s->args[0]})}));
  if (s->name == "positions") {
    require(call("distinct_axes", {s->args[0]}));
    require(call("distinct_axes", {s->args[1]}));
    require(call("subset", {s->args[1], s->args[0]}));
  }
  if (s->name == "replace_axes") {
    require(call("distinct_axes", {s->args[1]}));
    require(call("distinct_axes", {s->args[2]}));
    require(call("==", {call("len", {s->args[1]}), call("len", {s->args[2]})}));
  }
  validated_expressions[s.get()] = s;
}

void Context::tick() {
  if (!budget--) throw std::runtime_error("symbolic proof work limit exceeded");
}
void Context::note(const std::string& law) {
  if (std::find(steps.begin(), steps.end(), "law: " + law) == steps.end())
    steps.push_back("law: " + law);
}
S Context::canonical(const S& s) {
  tick();
  const auto original = key(s);
  auto cached = canonical_cache.find(original);
  if (cached != canonical_cache.end()) return cached->second;
  auto found = representatives.find(key(s));
  if (found != representatives.end()) return canonical(found->second);
  std::vector<S> args;
  for (const auto& a : s->args) args.push_back(canonical(a));
  S out = s;
  if (s->kind == Symbol::Kind::Call) {
    if (s->name == "gather" && callIs(args[1], "range") &&
        equal(args[1]->args[0], canonical(call("len", {args[0]}))))
      out = args[0];
    else
      out = call(s->name, std::move(args));
  } else if (!args.empty())
    out = make(s->kind, s->name, s->owner, s->index, std::move(args));
  found = representatives.find(key(out));
  auto result = found == representatives.end() ? out : canonical(found->second);
  canonical_cache[original] = result;
  return result;
}
void Context::learn(const S& fact) {
  canonical_cache.clear();
  known_cache.clear();
  auto f = canonical(fact);
  facts.push_back(fact);
  if (callIs(f, "axis_partition")) {
    learn(call("is_valid_axis_list", f->args));
    learn(call("==", {call("len", {f->args[0]}), f->args[1]}));
    return;
  }
  if (callIs(f, "same_set") &&
      (callIs(f->args[0], "empty_axes") || callIs(f->args[1], "empty_axes"))) {
    learn(call("==", f->args));
    return;
  }
  if (callIs(f, "==")) {
    auto a = f->args[0], b = f->args[1];
    auto ka = key(a), kb = key(b);
    if (ka == kb) return;
    std::function<std::size_t(const S&)> weight = [&](const S& x) {
      if (x->kind == Symbol::Kind::AxisGroup) return std::size_t(1);
      std::size_t n = 1;
      for (const auto& y : x->args) n += weight(y);
      return n;
    };
    auto category = [](const S& x) {
      if (x->kind == Symbol::Kind::Integer ||
          x->kind == Symbol::Kind::Boolean ||
          (x->kind == Symbol::Kind::List && x->args.empty()))
        return 0;
      return x->args.empty() || x->kind == Symbol::Kind::AxisGroup ? 1 : 2;
    };
    if (std::tuple(category(a), weight(a), ka) <
        std::tuple(category(b), weight(b), kb))
      std::swap(a, b);
    representatives[key(a)] = b;
  }
}
bool Context::known(const S& f) {
  auto c = canonical(f);
  auto k = key(c);
  auto found = known_cache.find(k);
  if (found != known_cache.end()) return found->second;
  if (!active_queries.insert(k).second) return false;
  bool result = prove(c);
  active_queries.erase(k);
  known_cache[k] = result;
  return result;
}
bool Context::prove(const S& f) {
  auto c = canonical(f);
  if (truth(c)) return true;
  for (const auto& a : facts)
    if (equal(c, canonical(a))) return true;
  if (callIs(c, "axis_partition")) {
    auto parts = [](const S& x) {
      std::vector<std::string> keys;
      if (callIs(x, "concat"))
        for (const auto& a : x->args) keys.push_back(key(a));
      else
        keys.push_back(key(x));
      std::sort(keys.begin(), keys.end());
      return keys;
    };
    for (const auto& fact : facts) {
      auto known = canonical(fact);
      if (callIs(known, "axis_partition") &&
          equal(known->args[1], c->args[1]) &&
          parts(known->args[0]) == parts(c->args[0]))
        return true;
    }
  }
  if (callIs(c, "distinct_axes")) {
    auto a = c->args[0];
    if (a->kind == Symbol::Kind::AxisGroup || callIs(a, "empty_axes"))
      return true;
    if (callIs(a, "concat")) {
      for (std::size_t i = 0; i < a->args.size(); ++i) {
        if (!known(call("distinct_axes", {a->args[i]}))) return false;
        for (std::size_t j = 0; j < i; ++j)
          if (!known(call("disjoint", {a->args[i], a->args[j]}))) return false;
      }
      return true;
    }
    if (callIs(a, "gather"))
      return known(call("distinct_axes", {a->args[0]})) &&
             known(call("is_valid_axis_list",
                        {a->args[1], call("len", {a->args[0]})}));
  }
  if (callIs(c, "same_set"))
    return equal(support(c->args[0]), support(c->args[1]));
  if (callIs(c, "disjoint"))
    return callIs(intersect(c->args[0], c->args[1]), "empty_axes");
  if (callIs(c, "subset")) {
    auto a = support(c->args[0]), b = support(c->args[1]);
    if (equal(a, b) || callIs(a, "empty_axes")) return true;
    if (callIs(a, "intersection_axes") || callIs(a, "filter_axes"))
      return known(call("subset", {a->args[0], b})) ||
             known(call("subset", {a->args[1], b}));
    if (callIs(a, "difference_axes"))
      return known(call("subset", {a->args[0], b}));
    if (callIs(a, "concat") || callIs(a, "union_axes")) {
      for (const auto& p : a->args)
        if (!known(call("subset", {p, b}))) return false;
      return true;
    }
    if (callIs(b, "concat") || callIs(b, "union_axes")) {
      for (const auto& p : b->args)
        if (known(call("subset", {a, p}))) return true;
    }
    return callIs(difference(a, b), "empty_axes");
  }
  if (callIs(c, "is_valid_axis_list") && callIs(c->args[0], "inverse")) {
    auto p = c->args[0]->args[0];
    return known(call("is_valid_axis_list", {p, c->args[1]})) &&
           known(call("==", {call("len", {p}), c->args[1]}));
  }
  // Selecting a distinct sequence by distinct in-range positions preserves
  // distinctness and range. This proves legality of composed permutations.
  if (callIs(c, "is_valid_axis_list") && callIs(c->args[0], "gather")) {
    const auto& selection = c->args[0];
    return known(
               call("is_valid_axis_list", {selection->args[0], c->args[1]})) &&
           known(call("is_valid_axis_list",
                      {selection->args[1], call("len", {selection->args[0]})}));
  }
  return false;
}
void Context::require(const S& f) {
  if (containsHole(f))
    throw std::runtime_error(
        "legality obligation contains an unresolved auto field");
  if (known(f)) return;
  if (!allow_guards)
    throw std::runtime_error("supplied guard does not establish obligation: " +
                             format(*f));
  auto c = canonical(f);
  guards.push_back(c);
  learn(c);
}
S Context::resolve(const S& s) {
  tick();
  if (s->kind == Symbol::Kind::Hole) {
    auto it = solutions.find({s->owner, s->index});
    return it == solutions.end() ? s : resolve(it->second);
  }
  std::vector<S> args;
  for (const auto& a : s->args) args.push_back(resolve(a));
  if (s->kind == Symbol::Kind::Call) return call(s->name, std::move(args));
  return make(s->kind, s->name, s->owner, s->index, std::move(args));
}
S Context::field(const Instance& n, std::size_t f) {
  if (!n.descriptor)
    throw std::runtime_error("attribute has no rule descriptor");
  if (n.automatic && f < n.pins.size() && n.pins[f]) {
    const auto& pin = *n.pins[f];
    if (pin.literal) {
      std::vector<S> xs;
      for (auto i : *pin.literal)
        xs.push_back(make(Symbol::Kind::Integer, std::to_string(i)));
      return make(Symbol::Kind::List, {}, 0, 0, std::move(xs));
    }
    std::string name;
    for (const auto& source : plan.nodes)
      if (source.descriptor == pin.descriptor && source.definition)
        name = source.definition->fields.at(pin.field);
    return make(Symbol::Kind::Field, name, pin.descriptor->value, pin.field);
  }
  return resolve(make(n.automatic ? Symbol::Kind::Hole : Symbol::Kind::Field,
                      n.definition->fields.at(f), n.descriptor->value, f));
}

namespace {
bool elementary(const S& x) {
  if (x->kind == Symbol::Kind::AxisGroup || callIs(x, "empty_axes"))
    return true;
  if (callIs(x, "union_axes") || callIs(x, "concat")) {
    for (const auto& a : x->args)
      if (!elementary(a)) return false;
    return true;
  }
  return false;
}
void tokens(const S& s, std::set<std::pair<std::size_t, std::size_t>>& out) {
  if (s->kind == Symbol::Kind::AxisGroup) {
    out.emplace(s->owner, s->index);
    return;
  }
  if (callIs(s, "intersection_axes") || callIs(s, "filter_axes")) {
    std::set<std::pair<std::size_t, std::size_t>> x, y;
    tokens(s->args[0], x);
    tokens(s->args[1], y);
    for (const auto& t : x)
      if (y.contains(t)) out.insert(t);
    return;
  }
  if (callIs(s, "difference_axes") || callIs(s, "gather")) {
    tokens(s->args[0], out);
    return;
  }
  for (const auto& a : s->args)
    if (a->sort == Symbol::Sort::Axes) tokens(a, out);
}
bool separate(const S& a, const S& b) {
  std::set<std::pair<std::size_t, std::size_t>> x, y;
  tokens(a, x);
  tokens(b, y);
  for (const auto& t : x)
    if (y.contains(t)) return false;
  return true;
}
}  // namespace
void Context::bind(const S& interface, const S& group, const S& p) {
  known_cache.clear();
  wiring.push_back({interface, group, p});
}
S Context::positions(const S& a, const S& b) {
  tick();
  auto x = canonical(a), y = canonical(b);
  if (callIs(y, "empty_axes")) return make(Symbol::Kind::List);
  if (equal(x, y)) return canonical(call("range", {call("len", {x})}));
  if (callIs(y, "gather") && equal(x, y->args[0])) return y->args[1];
  for (auto it = wiring.rbegin(); it != wiring.rend(); ++it)
    if (equal(x, canonical(it->interface)) && equal(y, canonical(it->group)))
      return canonical(it->positions);
  if (callIs(x, "gather"))
    return call("gather",
                {call("inverse", {x->args[1]}), positions(x->args[0], y)});
  if (callIs(y, "concat")) {
    std::vector<S> ps;
    for (const auto& part : y->args) ps.push_back(positions(x, part));
    return call("concat", std::move(ps));
  }
  if (callIs(x, "concat")) {
    S offset = integer(0);
    for (const auto& part : x->args) {
      if (equal(part, y))
        return call("offset_positions",
                    {call("range", {call("len", {y})}), offset});
      offset = call("+", {offset, call("len", {part})});
    }
  }
  return call("positions", {x, y});
}
S Context::select(const S& a, const S& p) {
  tick();
  auto x = canonical(a), q = canonical(p);
  for (auto it = wiring.rbegin(); it != wiring.rend(); ++it)
    if (equal(x, canonical(it->interface)) &&
        equal(q, canonical(it->positions)))
      return canonical(it->group);
  if (callIs(q, "positions") && equal(x, q->args[0])) return q->args[1];
  return call("gather", {x, q});
}
S Context::replace(const S& axes, const S& from, const S& to) {
  tick();
  if (equal(axes, from)) return to;
  if (separate(axes, from)) return axes;
  if (callIs(axes, "concat")) {
    std::vector<S> out;
    for (const auto& a : axes->args) out.push_back(replace(a, from, to));
    return call("concat", std::move(out));
  }
  if (callIs(axes, "gather"))
    return select(replace(axes->args[0], from, to), axes->args[1]);
  auto p = positions(from, axes);
  if (!callIs(p, "positions")) return select(to, p);
  return call("replace_axes", {axes, from, to});
}
S Context::support(const S& a) {
  tick();
  auto x = canonical(a);
  auto found = covers.find(key(x));
  if (found != covers.end() && !equal(x, found->second))
    return support(found->second);
  if (callIs(x, "unique_axes")) return support(x->args[0]);
  if (callIs(x, "difference_axes")) return difference(x->args[0], x->args[1]);
  if (callIs(x, "intersection_axes") || callIs(x, "filter_axes"))
    return intersect(x->args[0], x->args[1]);
  if (callIs(x, "gather") &&
      known(call("is_valid_axis_list",
                 {x->args[1], call("len", {x->args[0]})})) &&
      known(call("==", {call("len", {x->args[1]}), call("len", {x->args[0]})})))
    return support(x->args[0]);
  if (callIs(x, "concat") || callIs(x, "union_axes")) {
    std::vector<S> out;
    for (const auto& a : x->args) out.push_back(support(a));
    return call("union_axes", std::move(out));
  }
  return x;
}
S Context::intersect(const S& a, const S& b) {
  tick();
  auto x = support(a), y = support(b);
  if (equal(x, y)) return x;
  if (separate(x, y)) return call("empty_axes", {});
  if (!elementary(x) || !elementary(y)) {
    if (callIs(x, "intersection_axes") &&
        (equal(x->args[0], y) || equal(x->args[1], y)))
      return x;
    if (callIs(y, "intersection_axes") &&
        (equal(y->args[0], x) || equal(y->args[1], x)))
      return y;
    if (key(y) < key(x)) std::swap(x, y);
    return call("intersection_axes", {x, y});
  }
  if (callIs(x, "union_axes") || callIs(x, "concat")) {
    std::vector<S> out;
    for (const auto& p : x->args) out.push_back(intersect(p, y));
    return call("union_axes", std::move(out));
  }
  if (callIs(y, "union_axes") || callIs(y, "concat")) return intersect(y, x);
  return call("intersection_axes", {x, y});
}
S Context::difference(const S& a, const S& b) {
  tick();
  auto x = support(a), y = support(b);
  if (equal(x, y)) return call("empty_axes", {});
  if (separate(x, y)) return x;
  if (!elementary(x) || !elementary(y)) return call("difference_axes", {x, y});
  if (callIs(x, "union_axes") || callIs(x, "concat")) {
    std::vector<S> out;
    for (const auto& p : x->args) out.push_back(difference(p, y));
    return call("union_axes", std::move(out));
  }
  if (callIs(y, "union_axes") || callIs(y, "concat")) {
    auto out = x;
    for (const auto& p : y->args) out = difference(out, p);
    return out;
  }
  return call("difference_axes", {x, y});
}
S Context::filter(const S& order, const S& wanted) {
  tick();
  auto w = support(wanted), o = canonical(order);
  if (equal(o, w)) return o;
  if (callIs(o, "unique_axes"))
    return call("unique_axes", {filter(o->args[0], w)});
  if (callIs(o, "concat")) {
    std::vector<S> out;
    for (const auto& p : o->args) out.push_back(filter(p, w));
    return call("concat", std::move(out));
  }
  if (separate(o, w) || callIs(w, "empty_axes")) return call("empty_axes", {});
  if (equal(intersect(o, w), support(o))) return o;
  auto p = positions(o, w);
  if (callIs(p, "ordered_complement") || callIs(p, "range"))
    return select(o, p);
  return call("filter_axes", {o, w});
}
S Context::ordered(const S& candidates, const S& fallback) {
  return filter(fallback, candidates);
}
S Context::extents(const S& axes) {
  tick();
  if (axes->kind == Symbol::Kind::AxisGroup) return axes->args[0];
  if (callIs(axes, "empty_axes")) return make(Symbol::Kind::List);
  if (callIs(axes, "concat")) {
    std::vector<S> out;
    for (const auto& p : axes->args) out.push_back(extents(p));
    return call("concat", std::move(out));
  }
  if (callIs(axes, "gather"))
    return call("gather", {extents(axes->args[0]), axes->args[1]});
  return call("axis_extents", {axes});
}
}  // namespace tepl::core::index_semantics
