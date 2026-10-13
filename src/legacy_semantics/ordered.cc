#include "src/legacy_semantics/ordered.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace tepl::core::semantics {
namespace {
using Node = inference::SemanticNode;
struct Unsupported {};
struct Limit {};
struct Illegal {};

// Restricted symbolic coordinate routing. This is a checked normal form,
// not an attempt to decide equivalence of arbitrary semantic programs.
struct Term {
  std::string op;
  std::vector<Term> args;
  std::string key;
};
Term make(std::string op, std::vector<Term> args = {}) {
  if (op == "rank" && args.size() == 1)
    return make("len", {make("shape", std::move(args))});
  if (op == "unary:+" && args.size() == 1) return args[0];
  if (op == "+" && args.size() == 2) {
    if (args[0].op == "0") return args[1];
    if (args[1].op == "0") return args[0];
  }
  if (op == "concat") {
    std::vector<Term> flat;
    for (auto& a : args) {
      if (a.op == "concat")
        flat.insert(flat.end(), a.args.begin(), a.args.end());
      else if (a.op != "list" || !a.args.empty())
        flat.push_back(std::move(a));
    }
    if (flat.empty()) return make("list");
    if (flat.size() == 1) return flat[0];
    args = std::move(flat);
  }
  if ((op == "==" || op == "+" || op == "*" || op == "is_disjoint") &&
      args.size() == 2 && args[1].key < args[0].key)
    std::swap(args[0], args[1]);
  if (op == "scatter" && args.size() == 3) {
    auto groups = [](const Term& t) {
      return t.op == "concat" ? t.args : std::vector<Term>{t};
    };
    auto axes = groups(args[1]), values = groups(args[2]);
    if (axes.size() == values.size()) {
      std::vector<Term> routes;
      for (std::size_t i = 0; i < axes.size(); ++i)
        routes.push_back(make("route", {axes[i], values[i]}));
      std::sort(routes.begin(), routes.end(),
                [](auto& a, auto& b) { return a.key < b.key; });
      routes.insert(routes.begin(), args[0]);
      return make("routing", std::move(routes));
    }
  }
  std::string key = std::to_string(op.size()) + ":" + op + "(";
  for (auto& a : args) {
    if (key.size() + a.key.size() > 65536) throw Unsupported{};
    key += a.key + ";";
  }
  key += ")";
  return {std::move(op), std::move(args), std::move(key)};
}
struct Normalizer {
  const Node& node;
  std::string binder;
  std::map<std::size_t, Term> memo;
  std::set<std::size_t> active;
  Term get(std::size_t id) {
    if (auto it = memo.find(id); it != memo.end()) return it->second;
    if (active.size() >= 128 || !active.insert(id).second) throw Unsupported{};
    const auto& e = node.expressions.at(id);
    std::vector<Term> args;
    for (auto a : e.args) args.push_back(get(a));
    auto op = e.op;
    if (op == binder && !binder.empty()) op = "K";
    if (op == "sum:" + binder && !binder.empty()) op = "sum:K";
    auto out = make(op, std::move(args));
    active.erase(id);
    memo.emplace(id, out);
    return out;
  }
};
Term attr(std::size_t field) { return make("a" + std::to_string(field)); }
Term tensor(std::size_t operand) { return make("t" + std::to_string(operand)); }
Term shape(std::size_t operand) { return make("shape", {tensor(operand)}); }
Term len(Term x) { return make("len", {std::move(x)}); }
Term gather(Term x, Term a) {
  return make("gather", {std::move(x), std::move(a)});
}
Term concat(std::vector<Term> xs) { return make("concat", std::move(xs)); }
Term freeAxes(std::size_t operand, Term batch, Term contract) {
  return make("exclude", {make("range", {len(shape(operand))}),
                          concat({std::move(batch), std::move(contract)})});
}
struct Expected {
  Term element, output;
  std::set<std::string> assertions;
  std::set<std::string> metadata_terms;
  void collect(const Term& t) {
    metadata_terms.insert(t.key);
    for (auto& a : t.args) collect(a);
  }
  void assertion(Term t) {
    assertions.insert(t.key);
    collect(t);
  }
};
Expected permutationContract() {
  auto p = attr(0), s = shape(0);
  Expected e{make("read", {tensor(0), make("scatter", {len(s), p, make("o")})}),
             gather(s, p),
             {}};
  e.assertion(make("is_valid_axis_list", {p, len(s)}));
  e.assertion(make("==", {len(p), len(s)}));
  return e;
}
Expected contractionContract(const OrderedContract& c) {
  auto l = shape(c.left), r = shape(c.right);
  auto lc = attr(c.left_contract), rc = attr(c.right_contract);
  auto lb = attr(c.left_batch), rb = attr(c.right_batch);
  auto lf = freeAxes(c.left, lb, lc), rf = freeAxes(c.right, rb, rc);
  auto b = make("take", {make("o"), make("0"), len(lb)});
  auto i = make("take", {make("o"), len(lb), len(lf)});
  auto j = make("take", {make("o"), make("+", {len(lb), len(lf)}), len(rf)});
  auto li = make("scatter",
                 {len(l), concat({lb, lf, lc}), concat({b, i, make("K")})});
  auto ri = make("scatter",
                 {len(r), concat({rb, rf, rc}), concat({b, j, make("K")})});
  Expected e{make("sum:K", {make("index_space", {gather(l, lc)}),
                            make("*", {make("read", {tensor(c.left), li}),
                                       make("read", {tensor(c.right), ri})})}),
             concat({gather(l, lb), gather(l, lf), gather(r, rf)}),
             {}};
  for (auto [axes, dims] :
       std::vector<std::pair<Term, Term>>{{lc, l}, {rc, r}, {lb, l}, {rb, r}})
    e.assertion(make("is_valid_axis_list", {axes, len(dims)}));
  e.assertion(make("is_disjoint", {lb, lc}));
  e.assertion(make("is_disjoint", {rb, rc}));
  e.assertion(make("==", {len(lb), len(rb)}));
  e.assertion(make("==", {len(lc), len(rc)}));
  e.assertion(make("==", {gather(l, lb), gather(r, rb)}));
  e.assertion(make("==", {gather(l, lc), gather(r, rc)}));
  return e;
}

// Concrete port networks and the independent certificate checker. A port is
// an axis of a named, read-once input; union-find records logical coordinates.
struct Network {
  std::vector<std::size_t> parent;
  std::vector<std::uint64_t> dimensions;
  explicit Network(const std::vector<List>& shapes = {}) {
    for (auto& s : shapes)
      for (auto d : s) {
        parent.push_back(parent.size());
        dimensions.push_back(d);
      }
  }
  std::size_t root(std::size_t a) {
    auto r = a;
    while (parent.at(r) != r) r = parent[r];
    while (parent[a] != a) {
      auto next = parent[a];
      parent[a] = r;
      a = next;
    }
    return r;
  }
  void merge(std::size_t a, std::size_t b) {
    a = root(a);
    b = root(b);
    if (dimensions[a] != dimensions[b]) throw Illegal{};
    parent[std::max(a, b)] = std::min(a, b);
  }
};
using Axes = std::vector<std::size_t>;
using AxisSet = std::set<std::size_t>;
struct Context {
  const inference::SemanticPlan& plan;
  const std::vector<List>& shapes;
  const std::vector<List>& known;
  std::vector<std::optional<OrderedContract>> contracts;
  std::vector<Axes> ports;
  Axes final;
  std::map<std::size_t, AxisSet> support;
  std::vector<AxisSet> inputs, boundary;
  Network source;
  std::size_t budget;
  std::chrono::steady_clock::time_point deadline;
  Context(const inference::SemanticPlan& p, const std::vector<List>& s,
          const std::vector<List>& k, std::size_t work)
      : plan(p),
        shapes(s),
        known(k),
        source(),
        budget(work),
        deadline(std::chrono::steady_clock::now() +
                 std::chrono::milliseconds(p.timeout_ms)) {
    if (!p.algebraic) throw Unsupported{};
    if (p.nodes.size() > 10000) throw Limit{};
    std::size_t total = 0;
    for (auto& dims : s) {
      if (dims.size() > 100000 - total) throw Limit{};
      total += dims.size();
      for (auto d : dims)
        if (d < 2) throw Unsupported{};
      Axes axes(dims.size());
      std::iota(axes.begin(), axes.end(), total - dims.size());
      ports.push_back(std::move(axes));
    }
    source = Network(s);
    contracts.resize(p.nodes.size());
    inputs.resize(p.nodes.size());
    boundary.resize(p.nodes.size());
    for (std::size_t id = 0; id < p.nodes.size(); ++id) {
      tick();
      auto& n = p.nodes[id];
      if (n.capture < 0) {
        contracts[id] = extractOrderedContract(n);
        if (!contracts[id]) throw Unsupported{};
      } else if (!n.inputs.empty() || !n.fields.empty() ||
                 std::size_t(n.capture) >= s.size()) {
        throw Unsupported{};
      }
    }
    std::set<std::size_t> outputs;
    AxisSet seen_nodes;
    auto source_inputs = leaves(p.source_root, true, seen_nodes, outputs);
    seen_nodes.clear();
    auto target_inputs = leaves(p.rhs_root, false, seen_nodes, outputs);
    if (source_inputs != target_inputs || source_inputs.size() != s.size())
      throw Unsupported{};
    final = evaluate(p.source_root, source, nullptr);
    for (auto& axis : final) axis = source.root(axis);
    for (std::size_t i = 0; i < ports.size(); ++i)
      for (auto& a : ports[i]) {
        a = source.root(a);
        support[a].insert(i);
      }
    interfaces(p.rhs_root);
  }
  void tick() {
    if (!budget || std::chrono::steady_clock::now() >= deadline) throw Limit{};
    --budget;
  }
  AxisSet leaves(std::size_t id, bool lhs, AxisSet& seen,
                 std::set<std::size_t>& outputs, std::size_t depth = 0) {
    tick();
    if (depth > 256) throw Limit{};
    if (!seen.insert(id).second) throw Unsupported{};
    const auto& n = plan.nodes.at(id);
    if (n.capture >= 0) return {std::size_t(n.capture)};
    if ((lhs && id >= plan.source_end) || (!lhs && id < plan.source_end))
      throw Unsupported{};
    for (auto& f : n.fields) {
      if (lhs) {
        if (f.known < 0) throw Unsupported{};
      } else if (f.known != -1 || !outputs.insert(f.output).second) {
        throw Unsupported{};
      }
    }
    AxisSet all;
    for (auto child : n.inputs) {
      auto child_inputs = leaves(child, lhs, seen, outputs, depth + 1);
      for (auto input : child_inputs)
        if (!all.insert(input).second) throw Unsupported{};
    }
    return all;
  }
  const List& field(const Node& n, std::size_t f,
                    const std::vector<List>* candidate) {
    const auto& ref = n.fields.at(f);
    return candidate ? candidate->at(ref.output) : known.at(ref.known);
  }
  void valid(const List& axes, std::size_t rank) {
    AxisSet seen;
    for (auto a : axes) {
      tick();
      if (a >= rank || !seen.insert(a).second) throw Illegal{};
    }
  }
  Axes evaluate(std::size_t id, Network& network,
                const std::vector<List>* candidate) {
    tick();
    const auto& n = plan.nodes.at(id);
    if (n.capture >= 0) {
      Axes raw(shapes.at(n.capture).size());
      std::size_t offset = 0;
      for (std::size_t i = 0; i < std::size_t(n.capture); ++i)
        offset += shapes[i].size();
      std::iota(raw.begin(), raw.end(), offset);
      return raw;
    }
    const auto& c = *contracts.at(id);
    auto l = evaluate(n.inputs.at(c.left), network, candidate);
    if (c.kind == OrderedContract::Kind::Permutation) {
      const auto& p = field(n, c.permutation, candidate);
      valid(p, l.size());
      if (p.size() != l.size()) throw Illegal{};
      Axes out;
      for (auto i : p) out.push_back(l[i]);
      return out;
    }
    auto r = evaluate(n.inputs.at(c.right), network, candidate);
    const auto& lc = field(n, c.left_contract, candidate);
    const auto& rc = field(n, c.right_contract, candidate);
    const auto& lb = field(n, c.left_batch, candidate);
    const auto& rb = field(n, c.right_batch, candidate);
    List left_used = lc, right_used = rc;
    left_used.insert(left_used.end(), lb.begin(), lb.end());
    right_used.insert(right_used.end(), rb.begin(), rb.end());
    valid(left_used, l.size());
    valid(right_used, r.size());
    if (lc.size() != rc.size() || lb.size() != rb.size()) throw Illegal{};
    for (std::size_t i = 0; i < lc.size(); ++i)
      network.merge(l[lc[i]], r[rc[i]]);
    for (std::size_t i = 0; i < lb.size(); ++i)
      network.merge(l[lb[i]], r[rb[i]]);
    Axes out;
    for (auto i : lb) out.push_back(l[i]);
    AxisSet lu(left_used.begin(), left_used.end());
    AxisSet ru(right_used.begin(), right_used.end());
    for (std::size_t i = 0; i < l.size(); ++i)
      if (!lu.contains(i)) out.push_back(l[i]);
    for (std::size_t i = 0; i < r.size(); ++i)
      if (!ru.contains(i)) out.push_back(r[i]);
    return out;
  }
  void interfaces(std::size_t id) {
    tick();
    const auto& n = plan.nodes.at(id);
    if (n.capture >= 0) {
      inputs[id] = {std::size_t(n.capture)};
      boundary[id] = AxisSet(ports[n.capture].begin(), ports[n.capture].end());
      return;
    }
    for (auto child : n.inputs) {
      interfaces(child);
      inputs[id].insert(inputs[child].begin(), inputs[child].end());
    }
    for (auto& [axis, uses] : support) {
      tick();
      bool inside = false, outside = false;
      for (auto input : uses) {
        inside |= inputs[id].contains(input);
        outside |= !inputs[id].contains(input);
      }
      if (inside && (outside || std::find(final.begin(), final.end(), axis) !=
                                    final.end()))
        boundary[id].insert(axis);
    }
  }
  int role(std::size_t id, std::size_t a) {
    const auto& n = plan.nodes.at(id);
    const auto& c = *contracts[id];
    bool left = boundary[n.inputs[c.left]].contains(a);
    bool right = boundary[n.inputs[c.right]].contains(a);
    if (left && right && boundary[id].contains(a)) return 0;
    if (left && !right) return 1;
    if (!left && right) return 2;
    throw Illegal{};
  }
  Axes requirement(std::size_t target) {
    Axes found;
    bool located = false;
    std::function<void(std::size_t, const Axes&)> visit = [&](std::size_t id,
                                                              const Axes& q) {
      tick();
      if (id == target) {
        if (located) throw Illegal{};
        found = q;
        located = true;
        return;
      }
      const auto& n = plan.nodes.at(id);
      if (n.capture >= 0) return;
      const auto& c = *contracts[id];
      if (c.kind == OrderedContract::Kind::Permutation) {
        visit(n.inputs[0], {});
      } else {
        Axes l, r;
        for (auto a : q) {
          auto which = role(id, a);
          if (which == 1) l.push_back(a);
          if (which == 2) r.push_back(a);
        }
        visit(n.inputs[c.left], l);
        visit(n.inputs[c.right], r);
      }
    };
    visit(plan.rhs_root, final);
    if (!located) throw Illegal{};
    return found;
  }
  bool obstruction(const OrderedObstruction& cert) {
    auto q = requirement(cert.node);
    auto a = std::find(q.begin(), q.end(), cert.first);
    auto b = std::find(q.begin(), q.end(), cert.second);
    if (a == q.end() || b == q.end() || a >= b) return false;
    const auto& n = plan.nodes.at(cert.node);
    if (n.capture >= 0) {
      auto& axes = ports[n.capture];
      return std::find(axes.begin(), axes.end(), cert.first) >
             std::find(axes.begin(), axes.end(), cert.second);
    }
    if (contracts[cert.node]->kind != OrderedContract::Kind::Contraction)
      return false;
    return role(cert.node, cert.first) > role(cert.node, cert.second);
  }
  bool witness(const std::vector<List>& fields) {
    Network rhs(shapes);
    auto out = evaluate(plan.rhs_root, rhs, &fields);
    // Compare exact port partitions, not extent labels or contraction order.
    for (std::size_t p = 0; p < source.parent.size(); ++p) {
      tick();
      if (source.root(p) != rhs.root(p)) return false;
    }
    if (out.size() != final.size()) return false;
    for (std::size_t i = 0; i < out.size(); ++i)
      if (rhs.root(out[i]) != final[i]) return false;
    return true;
  }
};

struct Constructor {
  Context& context;
  std::vector<List> fields;
  std::optional<OrderedObstruction> conflict;
  explicit Constructor(Context& c) : context(c) {
    std::size_t size = 0;
    for (auto& n : c.plan.nodes)
      for (auto& f : n.fields) {
        if (f.output >= 100000) throw Limit{};
        size = std::max(size, f.output + 1);
      }
    fields.resize(size);
    for (auto& n : c.plan.nodes)
      for (auto& f : n.fields)
        if (f.known >= 0) fields[f.output] = c.known.at(f.known);
  }
  List positions(const Axes& layout, const Axes& selected) {
    std::map<std::size_t, std::size_t> index;
    for (std::size_t i = 0; i < layout.size(); ++i) index.emplace(layout[i], i);
    List result;
    for (auto a : selected) {
      context.tick();
      result.push_back(index.at(a));
    }
    return result;
  }
  Axes build(std::size_t id, const Axes& required) {
    context.tick();
    const auto& n = context.plan.nodes.at(id);
    if (n.capture >= 0) {
      auto axes = context.ports[n.capture];
      auto pos = positions(axes, required);
      for (std::size_t i = 1; i < pos.size(); ++i)
        if (pos[i - 1] > pos[i]) {
          conflict = OrderedObstruction{id, required[i - 1], required[i]};
          return {};
        }
      return axes;
    }
    const auto& c = *context.contracts[id];
    auto assign = [&](std::size_t f, List value) {
      fields.at(n.fields.at(f).output) = std::move(value);
    };
    if (c.kind == OrderedContract::Kind::Permutation) {
      auto child = build(n.inputs[0], {});
      if (conflict) return {};
      Axes out = required;
      for (auto a : child)
        if (std::find(required.begin(), required.end(), a) == required.end())
          out.push_back(a);
      assign(c.permutation, positions(child, out));
      return out;
    }
    Axes lq, rq, batch;
    int last = -1;
    for (std::size_t i = 0; i < required.size(); ++i) {
      auto role = context.role(id, required[i]);
      if (role < last) {
        conflict = OrderedObstruction{id, required[i - 1], required[i]};
        return {};
      }
      last = role;
      if (role == 0) batch.push_back(required[i]);
      if (role == 1) lq.push_back(required[i]);
      if (role == 2) rq.push_back(required[i]);
    }
    auto l = build(n.inputs[c.left], lq);
    if (conflict) return {};
    auto r = build(n.inputs[c.right], rq);
    if (conflict) return {};
    const auto& lb = context.boundary[n.inputs[c.left]];
    const auto& rb = context.boundary[n.inputs[c.right]];
    Axes contract;
    for (auto a : lb)
      if (rb.contains(a)) {
        context.tick();
        if (!context.boundary[id].contains(a))
          contract.push_back(a);
        else if (std::find(batch.begin(), batch.end(), a) == batch.end())
          batch.push_back(a);
      }
    assign(c.left_contract, positions(l, contract));
    assign(c.right_contract, positions(r, contract));
    assign(c.left_batch, positions(l, batch));
    assign(c.right_batch, positions(r, batch));
    Axes out = batch;
    for (auto a : l)
      if (!rb.contains(a)) out.push_back(a);
    for (auto a : r)
      if (!lb.contains(a)) out.push_back(a);
    return out;
  }
};
}  // namespace

std::optional<OrderedContract> extractOrderedContract(const Node& n) {
  try {
    if (n.capture >= 0 || n.expressions.size() > 2000) return {};
    auto root = n.expressions.at(n.root).op;
    std::string binder = root.starts_with("sum:") ? root.substr(4) : "";
    Normalizer normalized{n, binder};
    auto computation = normalized.get(n.root);
    auto element = computation.key;
    auto output = normalized.get(n.shape).key;
    std::vector<Term> assertions;
    for (auto a : n.assertions) {
      auto t = normalized.get(a);
      if (t.op != "true") assertions.push_back(std::move(t));
    }
    auto matches = [&](Expected e) {
      if (e.element.key != element || e.output.key != output) return false;
      e.collect(e.output);
      std::set<std::string> required;
      for (auto& t : assertions) {
        // Lowered lets carry self-equalities to preserve partial-expression
        // legality. Ignore one only if standard metadata also evaluates that
        // expression. An unused partial let prevents contract admission.
        if (t.op == "==" && t.args.size() == 2 &&
            t.args[0].key == t.args[1].key &&
            e.metadata_terms.contains(t.args[0].key))
          continue;
        required.insert(t.key);
      }
      return required == e.assertions;
    };
    if (n.inputs.size() == 1 && n.fields.size() == 1 &&
        matches(permutationContract()))
      return OrderedContract{OrderedContract::Kind::Permutation};
    if (n.inputs.size() != 2 || n.fields.size() != 4 || binder.empty())
      return {};
    auto field = [&](const Term& t) -> std::optional<std::size_t> {
      if (!t.args.empty() || t.op.size() < 2 || t.op[0] != 'a' ||
          t.op.find_first_not_of("0123456789", 1) != std::string::npos)
        return {};
      auto id = std::stoul(t.op.substr(1));
      return id < n.fields.size() ? std::optional(id) : std::nullopt;
    };
    // Derive field roles from coordinate routing. The reduction domain names
    // its operand and selector; each scatter route associates an axis list
    // with either the reduction coordinate K or an output-coordinate slice.
    if (computation.op != "sum:K" || computation.args.size() != 2) return {};
    const auto& domain = computation.args[0];
    if (domain.op != "index_space" || domain.args.size() != 1) return {};
    const auto& dimensions = domain.args[0];
    if (dimensions.op != "gather" || dimensions.args.size() != 2) return {};
    OrderedContract c{OrderedContract::Kind::Contraction};
    if (dimensions.args[0].key == shape(0).key)
      c.left = 0;
    else if (dimensions.args[0].key == shape(1).key)
      c.left = 1;
    else
      return {};
    c.right = 1 - c.left;
    auto lc = field(dimensions.args[1]);
    if (!lc) return {};
    c.left_contract = *lc;
    const auto& body = computation.args[1];
    if (body.op != "*" || body.args.size() != 2) return {};
    std::array<const Term*, 2> reads{};
    for (auto& read : body.args) {
      if (read.op != "read" || read.args.size() != 2) return {};
      for (std::size_t i = 0; i < 2; ++i)
        if (read.args[0].key == tensor(i).key) reads[i] = &read;
    }
    if (!reads[0] || !reads[1]) return {};
    auto selector = [&](std::size_t operand,
                        const Term& coordinate) -> std::optional<std::size_t> {
      const auto& routing = reads[operand]->args[1];
      if (routing.op != "routing") return {};
      std::optional<std::size_t> result;
      for (std::size_t i = 1; i < routing.args.size(); ++i) {
        const auto& route = routing.args[i];
        if (route.op != "route" || route.args.size() != 2) return {};
        if (route.args[1].key == coordinate.key) {
          if (result) return {};
          result = field(route.args[0]);
          if (!result) return {};
        }
      }
      return result;
    };
    auto rc = selector(c.right, make("K"));
    if (!rc || selector(c.left, make("K")) != lc) return {};
    c.right_contract = *rc;
    std::optional<std::size_t> lb;
    for (std::size_t f = 0; f < n.fields.size(); ++f) {
      auto batch = make("take", {make("o"), make("0"), len(attr(f))});
      if (selector(c.left, batch) == f) {
        if (lb) return {};
        lb = f;
      }
    }
    if (!lb) return {};
    c.left_batch = *lb;
    auto rb =
        selector(c.right, make("take", {make("o"), make("0"), len(attr(*lb))}));
    if (!rb) return {};
    c.right_batch = *rb;
    if (std::set<std::size_t>{c.left_contract, c.right_contract, c.left_batch,
                              c.right_batch}
            .size() != 4)
      return {};
    if (matches(contractionContract(c))) return c;
  } catch (...) {
    // Unrecognized, cyclic, or oversized IR cannot acquire a theorem contract.
  }
  return {};
}

std::optional<Solution> solveOrdered(const inference::SemanticPlan& p,
                                     const std::vector<List>& shapes,
                                     const std::vector<List>& known,
                                     const std::vector<List>& pinned,
                                     std::size_t budget) {
  if (!pinned.empty()) return {};
  try {
    Context context(p, shapes, known, budget);
    Constructor construct(context);
    construct.build(p.rhs_root, context.final);
    if (construct.conflict) {
      if (!context.obstruction(*construct.conflict))
        return Solution{
            Status::Unknown, {}, "obstruction check failed", "ordered"};
      auto& c = *construct.conflict;
      return Solution{Status::Invalid,
                      {},
                      "two-axis ordering obstruction at RHS node " +
                          std::to_string(c.node) + ": logical axis " +
                          std::to_string(c.first) + " must precede " +
                          std::to_string(c.second),
                      "ordered"};
    }
    if (!context.witness(construct.fields))
      return Solution{
          Status::Unknown, {}, "constructed network check failed", "ordered"};
    return Solution{Status::Found, std::move(construct.fields),
                    "checked ordered-contraction witness", "ordered"};
  } catch (const Unsupported&) {
    return {};
  } catch (const Limit&) {
    return Solution{
        Status::Unknown, {}, "ordered-contraction resource limit", "ordered"};
  } catch (...) {
    return Solution{Status::Unknown,
                    {},
                    "illegal source or incomplete ordered analysis",
                    "ordered"};
  }
}
bool checkOrderedWitness(const inference::SemanticPlan& p,
                         const std::vector<List>& shapes,
                         const std::vector<List>& known,
                         const std::vector<List>& fields) {
  try {
    Context context(p, shapes, known, 1000000);
    return context.witness(fields);
  } catch (...) {
    return false;
  }
}
bool checkOrderedObstruction(const inference::SemanticPlan& p,
                             const std::vector<List>& shapes,
                             const std::vector<List>& known,
                             const OrderedObstruction& cert) {
  try {
    Context context(p, shapes, known, 1000000);
    return context.obstruction(cert);
  } catch (...) {
    return false;
  }
}
}  // namespace tepl::core::semantics
