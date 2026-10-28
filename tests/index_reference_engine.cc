#include <algorithm>
#include <functional>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <variant>

#include "tests/index_constructor_eval.h"
#include "tests/index_reference.h"

namespace tepl::core::semantics {
namespace {
using Axis = std::int64_t;
using Axes = List;
using Set = std::set<Axis>;
struct Failure : std::runtime_error {
  bool illegal;
  Failure(std::string s, bool bad = false)
      : std::runtime_error(s), illegal(bad) {}
};
struct Unbound : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Expr;
using Ptr = std::shared_ptr<const Expr>;
struct Read {
  std::size_t capture, occurrence;
  Axes axes;
};
struct Expr {
  enum class Kind { Read, Add, Product, Reduce, Const } kind;
  Read read{};
  Axes axes;
  std::vector<Ptr> args;
  std::int64_t constant = 0;
};
struct Tensor {
  Axes axes;
  Ptr value;
};
struct Term {
  Axes reductions;
  std::vector<Read> reads;
  std::int64_t coefficient = 1;
};
using Terms = std::vector<Term>;
using Datum = std::variant<std::int64_t, bool, List>;

Set set(const Axes& a) { return {a.begin(), a.end()}; }
std::string list(const List& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  return s + "]";
}
Axes positions(const Axes& layout, const Axes& selected) {
  Axes out;
  for (auto a : selected) {
    auto it = std::find(layout.begin(), layout.end(), a);
    if (it == layout.end())
      throw Failure("logical axis absent from operand interface");
    out.push_back(it - layout.begin());
  }
  return out;
}
std::string key(const Term& t) {
  std::vector<std::size_t> captures;
  for (const auto& r : t.reads) captures.push_back(r.capture);
  std::sort(captures.begin(), captures.end());
  std::string k;
  for (auto c : captures) k += std::to_string(c) + ",";
  return k;
}
std::string expression(const Ptr& e) {
  if (e->kind == Expr::Kind::Const)
    return "Const(" + std::to_string(e->constant) + ")";
  if (e->kind == Expr::Kind::Read)
    return "Input(#" + std::to_string(e->read.capture) + "," +
           list(e->read.axes) + ")";
  auto s = e->kind == Expr::Kind::Add       ? std::string("Add(")
           : e->kind == Expr::Kind::Product ? std::string("Product(")
                                            : "Reduce(";
  if (e->kind == Expr::Kind::Reduce) s += list(e->axes) + ",";
  for (std::size_t i = 0; i < e->args.size(); ++i) {
    if (i) s += ",";
    s += expression(e->args[i]);
  }
  return s + ")";
}
std::string normal(const Terms& ts) {
  std::string s;
  for (const auto& t : ts) {
    if (!s.empty()) s += " + ";
    s += "Reduce(" + list(t.reductions) + ",Product(";
    for (std::size_t i = 0; i < t.reads.size(); ++i) {
      if (i) s += ",";
      s += "Input(#" + std::to_string(t.reads[i].capture) + "," +
           list(t.reads[i].axes) + ")";
    }
    s += "))";
  }
  return s.empty() ? "0" : s;
}
struct Engine {
  const Plan& plan;
  const Metadata& metadata;
  Limits limits;
  Result result;
  std::size_t work = 0;
  std::vector<std::int64_t> domains;
  std::map<std::size_t, std::vector<std::optional<List>>> attrs;
  bool checking_source = true;
  std::map<std::size_t, Axes> leaf_layouts;
  std::map<std::size_t, Set> boundary;
  Tensor source;

  void tick(std::size_t n = 1) {
    if (n > limits.max_work || work > limits.max_work - n)
      throw Failure("index matching work limit exceeded");
    work += n;
  }
  [[noreturn]] void bad(const std::string& s) {
    throw Failure(s, checking_source);
  }
  Axis fresh(std::int64_t extent) {
    if (extent < 0) bad("negative tensor extent");
    tick();
    domains.push_back(extent);
    return domains.size() - 1;
  }
  std::int64_t extent(Axis a) { return domains.at(a); }
  void legal(const List& p, std::size_t rank) {
    Set seen;
    for (auto i : p)
      if (i < 0 || std::size_t(i) >= rank || !seen.insert(i).second)
        bad("axis positions must be distinct and in range");
  }
  Datum eval(const Meta& m, const Instance& node,
             const std::vector<Tensor>& inputs,
             const std::vector<Axes>& groups) {
    tick();
    if (m.kind == Meta::Kind::Integer) return std::stoll(m.name);
    if (m.kind == Meta::Kind::Boolean) return m.name == "true";
    if (m.kind == Meta::Kind::Attribute) {
      if (!node.descriptor || !attrs.contains(node.descriptor->value) ||
          m.index >= attrs.at(node.descriptor->value).size() ||
          !attrs.at(node.descriptor->value)[m.index])
        throw Unbound("unresolved attribute '" + m.name + "'");
      return *attrs.at(node.descriptor->value)[m.index];
    }
    if (m.kind == Meta::Kind::Group) return groups.at(m.index);
    if (m.kind == Meta::Kind::List) {
      List out;
      for (const auto& a : m.args)
        out.push_back(std::get<std::int64_t>(eval(*a, node, inputs, groups)));
      return out;
    }
    if (m.kind != Meta::Kind::Call)
      throw Failure("operand used as metadata value");
    if (m.name == "all_axes" || m.name == "rank" || m.name == "shape") {
      const auto& axes = inputs.at(m.args.at(0)->index).axes;
      if (m.name == "rank") return std::int64_t(axes.size());
      List out;
      for (std::size_t i = 0; i < axes.size(); ++i)
        out.push_back(m.name == "shape" ? extent(axes[i]) : std::int64_t(i));
      return out;
    }
    std::vector<Datum> args;
    for (const auto& a : m.args) args.push_back(eval(*a, node, inputs, groups));
    auto seq = [&](std::size_t i) -> const List& {
      return std::get<List>(args.at(i));
    };
    auto integer = [&](std::size_t i) {
      return std::get<std::int64_t>(args.at(i));
    };
    if (m.name == "concat") {
      List out;
      for (const auto& a : args) {
        const auto& s = std::get<List>(a);
        out.insert(out.end(), s.begin(), s.end());
      }
      return out;
    }
    if (m.name == "len") return std::int64_t(seq(0).size());
    if (m.name == "ordered_complement") {
      List out;
      auto excluded = set(seq(1));
      for (auto i : seq(0))
        if (!excluded.contains(i)) out.push_back(i);
      return out;
    }
    if (m.name == "permute") {
      legal(seq(1), seq(0).size());
      if (seq(1).size() != seq(0).size())
        bad("permutation must cover every axis");
      List out;
      for (auto i : seq(1)) out.push_back(seq(0)[i]);
      return out;
    }
    if (m.name == "is_valid_axis_list") {
      Set seen;
      for (auto i : seq(0))
        if (i < 0 || i >= integer(1) || !seen.insert(i).second) return false;
      return true;
    }
    if (m.name == "==") return args.at(0) == args.at(1);
    if (m.name == "!=") return args.at(0) != args.at(1);
    if (m.name == "&&")
      return std::get<bool>(args[0]) && std::get<bool>(args[1]);
    if (m.name == "||")
      return std::get<bool>(args[0]) || std::get<bool>(args[1]);
    if (m.name == "<") return integer(0) < integer(1);
    if (m.name == "<=") return integer(0) <= integer(1);
    if (m.name == ">") return integer(0) > integer(1);
    if (m.name == ">=") return integer(0) >= integer(1);
    if (m.name == "+" || m.name == "-" || m.name == "*") {
      std::int64_t out;
      bool overflow =
          m.name == "+" ? __builtin_add_overflow(integer(0), integer(1), &out)
          : m.name == "-"
              ? __builtin_sub_overflow(integer(0), integer(1), &out)
              : __builtin_mul_overflow(integer(0), integer(1), &out);
      if (overflow) bad("axis metadata arithmetic overflow");
      return out;
    }
    throw Failure("unsupported axis evaluator builtin");
  }
  Ptr read(std::size_t capture, std::size_t occurrence, Axes axes) {
    return std::make_shared<Expr>(
        Expr{Expr::Kind::Read, {capture, occurrence, std::move(axes)}, {}, {}});
  }
  Ptr value(const Value& v, const std::vector<Tensor>& inputs,
            const std::vector<Axes>& groups) {
    tick();
    if (v.kind == Value::Kind::Const)
      return std::make_shared<Expr>(
          Expr{Expr::Kind::Const, {}, {}, {}, v.constant});
    if (v.kind == Value::Kind::Operand) return inputs.at(v.index).value;
    std::vector<Ptr> args;
    for (const auto& a : v.args) args.push_back(value(*a, inputs, groups));
    auto kind = v.kind == Value::Kind::Add       ? Expr::Kind::Add
                : v.kind == Value::Kind::Product ? Expr::Kind::Product
                                                 : Expr::Kind::Reduce;
    return std::make_shared<Expr>(
        Expr{kind,
             {},
             v.kind == Value::Kind::Reduce ? groups.at(v.index) : Axes{},
             args});
  }
  bool constantValue(const Ptr& e) {
    if (e->kind == Expr::Kind::Const) return true;
    if (e->kind == Expr::Kind::Read) return false;
    for (const auto& a : e->args)
      if (!constantValue(a)) return false;
    return true;
  }
  Set free(const Ptr& e) {
    if (e->kind == Expr::Kind::Const) return {};
    if (e->kind == Expr::Kind::Read) return set(e->read.axes);
    auto out = free(e->args[0]);
    if (e->kind == Expr::Kind::Reduce) {
      for (auto a : e->axes) out.erase(a);
    } else {
      auto r = free(e->args[1]);
      if (e->kind == Expr::Kind::Add && out != r &&
          !constantValue(e->args[0]) && !constantValue(e->args[1]))
        bad("Add coordinate interfaces differ");
      out.insert(r.begin(), r.end());
    }
    return out;
  }
  Ptr rename(const Ptr& e, std::map<Axis, Axis> map) {
    tick();
    if (e->kind == Expr::Kind::Const) return e;
    if (e->kind == Expr::Kind::Read) {
      auto r = e->read;
      for (auto& a : r.axes)
        if (map.contains(a)) a = map.at(a);
      return read(r.capture, r.occurrence, r.axes);
    }
    auto binders = e->axes;
    if (e->kind == Expr::Kind::Reduce) {
      for (auto& a : binders) {
        auto next = fresh(extent(a));
        map[a] = next;
        a = next;
      }
    }
    std::vector<Ptr> args;
    for (const auto& arg : e->args) args.push_back(rename(arg, map));
    return std::make_shared<Expr>(Expr{e->kind, {}, binders, args});
  }
  Tensor finish(const Instance& n, const std::vector<Tensor>& inputs,
                const std::vector<Axes>& groups) {
    for (const auto& a : n.definition->assertions)
      if (!std::get<bool>(eval(*a, n, inputs, groups)))
        bad("axis assertion failed: " + format(*a));
    auto out = std::get<List>(eval(*n.definition->output, n, inputs, groups));
    auto body = value(*n.definition->value, inputs, groups);
    auto dependencies = free(body), interface = set(out);
    if (interface.size() != out.size() ||
        !std::includes(interface.begin(), interface.end(), dependencies.begin(),
                       dependencies.end()))
      bad("output interface must list every free axis exactly once");
    return {out, body};
  }
  Tensor lower(std::size_t id) {
    tick();
    const auto& n = plan.nodes.at(id);
    if (n.capture) {
      if (n.capture->value >= metadata.shapes.size())
        bad("missing input shape");
      Axes axes;
      for (auto size : metadata.shapes[n.capture->value])
        axes.push_back(fresh(size));
      return {axes, read(n.capture->value, id, axes)};
    }
    std::vector<Tensor> inputs;
    for (auto input : n.inputs) inputs.push_back(lower(input));
    std::vector<Axes> groups(n.definition->groups.size());
    std::vector<Axes> layouts;
    for (const auto& in : inputs) layouts.emplace_back(in.axes.size(), -1);
    for (std::size_t g = 0; g < groups.size(); ++g) {
      bool first = true;
      for (const auto& b : n.definition->groups[g].bindings) {
        const auto& in = inputs.at(b.operand);
        auto p = std::get<List>(eval(*b.positions, n, inputs, groups));
        legal(p, in.axes.size());
        if (!first && p.size() != groups[g].size())
          bad("paired binding lengths differ");
        for (std::size_t i = 0; i < p.size(); ++i) {
          auto size = extent(in.axes[p[i]]);
          if (first)
            groups[g].push_back(fresh(size));
          else if (size != extent(groups[g][i]))
            bad("paired binding extents differ");
          if (layouts[b.operand][p[i]] != -1)
            bad("input axis belongs to multiple groups");
          layouts[b.operand][p[i]] = groups[g][i];
        }
        first = false;
      }
    }
    for (std::size_t o = 0; o < inputs.size(); ++o) {
      if (std::find(layouts[o].begin(), layouts[o].end(), -1) !=
          layouts[o].end())
        bad("input axis has no binding");
      std::map<Axis, Axis> renaming;
      for (std::size_t i = 0; i < layouts[o].size(); ++i)
        renaming[inputs[o].axes[i]] = layouts[o][i];
      inputs[o] = {layouts[o], rename(inputs[o].value, renaming)};
    }
    return finish(n, inputs, groups);
  }
  void note(const std::string& law) {
    auto definition =
        std::find_if(laws().begin(), laws().end(),
                     [&](const auto& entry) { return entry.name == law; });
    if (definition == laws().end())
      throw Failure("undefined algebraic law: " + law);
    if (definition->algebraic && !plan.algebraic)
      throw Failure("arithmetic policy does not permit law: " + law);
    if (std::find(result.trace.begin(), result.trace.end(), "law: " + law) ==
        result.trace.end())
      result.trace.push_back("law: " + law);
  }
  Terms normalize(const Ptr& e) {
    tick();
    if (e->kind == Expr::Kind::Read) return {{{}, {e->read}}};
    if (e->kind == Expr::Kind::Const) return {{{}, {}, e->constant}};
    auto l = normalize(e->args[0]);
    if (e->kind == Expr::Kind::Reduce) {
      if (l.size() > 1) note("reduction-linearity");
      if (e->args[0]->kind == Expr::Kind::Reduce) note("reduction-composition");
      for (auto& t : l)
        t.reductions.insert(t.reductions.begin(), e->axes.begin(),
                            e->axes.end());
    } else {
      auto r = normalize(e->args[1]);
      if (e->kind == Expr::Kind::Add) {
        if (l.size() + r.size() > limits.max_terms)
          throw Failure("normalization term limit exceeded");
        if (e->args[0]->kind == Expr::Kind::Add ||
            e->args[1]->kind == Expr::Kind::Add)
          note("add-associativity");
        l.insert(l.end(), r.begin(), r.end());
      } else {
        if (!r.empty() && l.size() > limits.max_terms / r.size())
          throw Failure("normalization term limit exceeded");
        if (l.size() > 1 || r.size() > 1) note("product-distributivity");
        if (e->args[0]->kind == Expr::Kind::Product ||
            e->args[1]->kind == Expr::Kind::Product)
          note("product-associativity");
        Terms out;
        for (const auto& a : l)
          for (const auto& b : r) {
            tick();
            if (!a.reductions.empty() || !b.reductions.empty())
              note("reduction-extraction");
            // This side condition is checked rather than assumed. Lowering
            // freshens binders, so well-scoped composed programs satisfy it.
            auto independent = [&](const Term& bound, const Term& other) {
              auto axes = set(bound.reductions);
              for (const auto& read : other.reads)
                for (auto x : read.axes)
                  if (axes.contains(x))
                    throw Failure(
                        "reduction extraction would capture a free axis");
            };
            independent(a, b);
            independent(b, a);
            Term t = a;
            if (__builtin_mul_overflow(a.coefficient, b.coefficient,
                                       &t.coefficient))
              throw Failure("reference coefficient overflow");
            t.reductions.insert(t.reductions.end(), b.reductions.begin(),
                                b.reductions.end());
            if (set(t.reductions).size() != t.reductions.size())
              throw Failure("reduction binders are not disjoint");
            t.reads.insert(t.reads.end(), b.reads.begin(), b.reads.end());
            out.push_back(std::move(t));
          }
        l = std::move(out);
      }
    }
    if (l.size() > limits.max_terms)
      throw Failure("normalization term limit exceeded");
    return l;
  }
  Ptr skeleton(std::size_t id) {
    const auto& n = plan.nodes.at(id);
    if (n.capture) return read(n.capture->value, id, {});
    std::vector<Tensor> inputs;
    for (auto child : n.inputs) inputs.push_back({{}, skeleton(child)});
    return value(*n.definition->value, inputs,
                 std::vector<Axes>(n.definition->groups.size()));
  }
  std::set<std::size_t> leaves(std::size_t id) {
    tick();
    const auto& n = plan.nodes[id];
    if (n.capture) return {id};
    std::set<std::size_t> out;
    for (auto child : n.inputs) {
      auto part = leaves(child);
      out.insert(part.begin(), part.end());
    }
    return out;
  }
  void correspond(const Terms& lhs, const Terms& rhs) {
    std::map<std::string, const Term*> source_terms;
    for (const auto& t : lhs)
      if (!source_terms.emplace(key(t), &t).second)
        throw Failure(
            "multiple terms with the same capture multidegree are unsupported");
    std::set<std::string> matched;
    std::vector<std::pair<const Term*, const Term*>> pairs;
    for (const auto& t : rhs) {
      auto k = key(t);
      if (!source_terms.contains(k) || !matched.insert(k).second)
        throw Failure("normalized term correspondence is absent or ambiguous");
      const auto& src = *source_terms.at(k);
      std::map<std::size_t, const Read*> reads;
      for (const auto& r : src.reads)
        if (!reads.emplace(r.capture, &r).second)
          throw Failure(
              "repeated capture factors require occurrence matching "
              "(unsupported)");
      std::set<std::size_t> seen;
      for (const auto& r : t.reads) {
        if (!seen.insert(r.capture).second)
          throw Failure("repeated capture factors are unsupported");
        const auto& axes = reads.at(r.capture)->axes;
        if (leaf_layouts.contains(r.occurrence) &&
            leaf_layouts.at(r.occurrence) != axes)
          throw Failure(
              "one RHS input occurrence needs incompatible logical bindings "
              "across terms");
        leaf_layouts[r.occurrence] = axes;
      }
      pairs.push_back({&src, &t});
    }
    if (matched.size() != source_terms.size())
      throw Failure("normalized term counts differ");
    auto final = set(source.axes);
    std::function<void(std::size_t)> visit = [&](std::size_t id) {
      auto local = leaves(id);
      std::optional<Set> required;
      for (const auto& pair : pairs) {
        Set inside, outside;
        bool active = false;
        for (const auto& read : pair.second->reads) {
          auto& target = local.contains(read.occurrence) ? inside : outside;
          active |= local.contains(read.occurrence);
          auto a = leaf_layouts.at(read.occurrence);
          target.insert(a.begin(), a.end());
        }
        if (!active) continue;
        Set b;
        for (auto a : inside)
          if (final.contains(a) || outside.contains(a)) b.insert(a);
        if (required && *required != b)
          throw Failure(
              "RHS subtree requires different interfaces in different additive "
              "terms");
        required = b;
      }
      if (!required)
        throw Failure("RHS operation is absent from semantic computation");
      boundary[id] = *required;
      for (auto child : plan.nodes[id].inputs) visit(child);
    };
    visit(plan.rhs);
  }
  void assign(const Meta& m, const List& value, const Instance& n,
              const std::vector<Tensor>& inputs,
              const std::vector<Axes>& groups) {
    tick();
    if (m.kind == Meta::Kind::Attribute) {
      if (!n.descriptor) throw Failure("attribute has no descriptor");
      auto& slots = attrs.at(n.descriptor->value);
      auto& slot = slots.at(m.index);
      if (slot && *slot != value)
        throw Failure(
            "attribute binding conflicts with pin, known value, or shared use");
      if (!slot) {
        slot = value;
        result.trace.push_back("construct @#" +
                               std::to_string(n.descriptor->value) + "." +
                               m.name + " = " + list(value));
      }
      return;
    }
    try {
      if (std::get<List>(eval(m, n, inputs, groups)) != value)
        throw Failure(
            "binding expression disagrees with required coordinate "
            "positions: " +
            format(m));
    } catch (const Unbound&) {
      throw Failure(
          "inverse binding expression requires unsupported constraint "
          "solving: " +
          format(m));
    }
  }
  Axes preferred(const Set& wanted, const Axes& fallback) {
    Axes out;
    for (auto a : source.axes)
      if (wanted.contains(a)) out.push_back(a);
    for (auto a : fallback)
      if (wanted.contains(a) &&
          std::find(out.begin(), out.end(), a) == out.end())
        out.push_back(a);
    return out;
  }
  void outputHoles(const Meta& m, const Instance& n,
                   const std::vector<Tensor>& inputs,
                   const std::vector<Axes>& groups) {
    if (m.kind != Meta::Kind::Call) return;
    if (m.name == "permute") {
      try {
        eval(*m.args[1], n, inputs, groups);
      } catch (const Unbound&) {
        auto base = std::get<List>(eval(*m.args[0], n, inputs, groups));
        auto out = preferred(set(base), base);
        assign(*m.args[1], positions(base, out), n, inputs, groups);
      }
    }
    for (const auto& a : m.args) outputHoles(*a, n, inputs, groups);
  }
  Tensor construct(std::size_t id) {
    tick();
    const auto& n = plan.nodes[id];
    if (n.capture) {
      if (!leaf_layouts.contains(id))
        throw Failure("capture has no matched binding");
      auto axes = leaf_layouts.at(id);
      return {axes, read(n.capture->value, id, axes)};
    }
    std::vector<Tensor> inputs;
    for (auto child : n.inputs) inputs.push_back(construct(child));
    std::set<std::size_t> reduced;
    std::function<void(const Value&)> reductions = [&](const Value& v) {
      if (v.kind == Value::Kind::Reduce) reduced.insert(v.index);
      for (const auto& a : v.args) reductions(*a);
    };
    reductions(*n.definition->value);
    std::vector<Set> selected(n.definition->groups.size());
    Set all;
    for (const auto& in : inputs) all.insert(in.axes.begin(), in.axes.end());
    for (auto a : all) {
      std::set<std::size_t> support;
      for (std::size_t o = 0; o < inputs.size(); ++o)
        if (set(inputs[o].axes).contains(a)) support.insert(o);
      std::optional<std::size_t> chosen;
      for (std::size_t g = 0; g < selected.size(); ++g) {
        std::set<std::size_t> ports;
        for (const auto& b : n.definition->groups[g].bindings)
          ports.insert(b.operand);
        if (ports != support ||
            reduced.contains(g) == boundary.at(id).contains(a))
          continue;
        if (chosen)
          throw Failure(
              "multiple groups have the same incidence and reduction role");
        chosen = g;
      }
      if (!chosen)
        throw Failure(
            "axis incidence cannot be represented by target bindings");
      selected[*chosen].insert(a);
    }
    std::vector<Axes> groups(selected.size());
    for (std::size_t g = 0; g < groups.size(); ++g) {
      const auto& declaration = n.definition->groups[g];
      const auto& first = declaration.bindings.front();
      const auto& base = inputs[first.operand].axes;
      if (reduced.contains(g))
        groups[g] = Axes(selected[g].begin(), selected[g].end());
      else if (declaration.bindings.size() > 1)
        groups[g] = preferred(selected[g], base);
      else
        for (auto a : base)
          if (selected[g].contains(a)) groups[g].push_back(a);
      // A supplied ordering is authoritative; choosing a canonical order must
      // not reject an otherwise valid pinned or known descriptor.
      std::optional<Axes> prescribed;
      for (const auto& b : declaration.bindings) {
        try {
          const auto& layout = inputs[b.operand].axes;
          auto p = std::get<List>(eval(*b.positions, n, inputs, groups));
          legal(p, layout.size());
          Axes ordered;
          for (auto pos : p) ordered.push_back(layout[pos]);
          if (set(ordered) != selected[g])
            throw Failure("known binding has incompatible logical axes");
          if (prescribed && *prescribed != ordered)
            throw Failure("known bindings prescribe different group orders");
          prescribed = ordered;
        } catch (const Unbound&) {
        }
      }
      if (prescribed) groups[g] = *prescribed;
      result.trace.push_back("match n" + std::to_string(id) + "." +
                             declaration.name + " = " + list(groups[g]));
      for (const auto& b : declaration.bindings)
        assign(*b.positions, positions(inputs[b.operand].axes, groups[g]), n,
               inputs, groups);
    }
    outputHoles(*n.definition->output, n, inputs, groups);
    auto out = finish(n, inputs, groups);
    if (set(out.axes) != boundary.at(id))
      throw Failure(
          "constructed operation boundary differs from matched boundary");
    return out;
  }
  bool equivalent(const Tensor& lhs, const Terms& lt, const Tensor& rhs,
                  const Terms& rt) {
    if (lhs.axes.size() != rhs.axes.size() || lt.size() != rt.size())
      return false;
    std::map<Axis, Axis> outputs;
    Set used;
    for (std::size_t i = 0; i < lhs.axes.size(); ++i) {
      if (extent(lhs.axes[i]) != extent(rhs.axes[i])) return false;
      outputs[rhs.axes[i]] = lhs.axes[i];
      used.insert(lhs.axes[i]);
    }
    std::map<std::string, const Term*> left;
    for (const auto& t : lt)
      if (!left.emplace(key(t), &t).second) return false;
    std::set<std::string> seen;
    for (const auto& t : rt) {
      auto k = key(t);
      if (!left.contains(k) || !seen.insert(k).second) return false;
      const auto& l = *left.at(k);
      if (l.coefficient != t.coefficient ||
          l.reductions.size() != t.reductions.size())
        return false;
      auto map = outputs;
      auto range = used;
      auto lb = set(l.reductions), rb = set(t.reductions);
      std::map<std::size_t, const Read*> reads;
      for (const auto& r : l.reads)
        if (!reads.emplace(r.capture, &r).second) return false;
      std::set<std::size_t> captures;
      for (const auto& r : t.reads) {
        if (!captures.insert(r.capture).second || !reads.contains(r.capture))
          return false;
        const auto& a = reads.at(r.capture)->axes;
        if (a.size() != r.axes.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
          tick();
          auto x = r.axes[i], y = a[i];
          if (extent(x) != extent(y) || rb.contains(x) != lb.contains(y))
            return false;
          if (map.contains(x)) {
            if (map.at(x) != y) return false;
          } else {
            if (!range.insert(y).second) return false;
            map[x] = y;
          }
        }
      }
      for (auto b : t.reductions)
        if (!map.contains(b) || !lb.contains(map.at(b))) return false;
    }
    note("commutative-matching");
    note("axis-alpha-renaming");
    return true;
  }
  void initialize() {
    for (const auto& [id, fields] : metadata.descriptors) {
      std::vector<std::optional<List>> values;
      for (const auto& f : fields) values.push_back(f);
      attrs[id] = values;
    }
    tepl::test::ConstructorEvaluator explicit_values{metadata, plan};
    for (const auto& [id, descriptor] : plan.descriptors) {
      if (descriptor.isCaptured()) continue;
      std::vector<std::optional<List>> values;
      for (std::size_t f = 0; f < descriptor.fields.size(); ++f) {
        if (descriptor.isAuto(f))
          values.emplace_back();
        else
          values.push_back(
              std::get<List>(explicit_values.eval(descriptor.fields[f])));
      }
      attrs[id] = std::move(values);
    }
    for (const auto& n : plan.nodes) {
      if (!n.descriptor || !n.definition) continue;
      auto id = n.descriptor->value;
      if (!attrs.contains(id)) attrs[id].resize(n.definition->fields.size());
      if (attrs.at(id).size() != n.definition->fields.size())
        bad("descriptor field count mismatch");
      if (!plan.needsSynthesis(n)) {
        for (const auto& f : attrs.at(id))
          if (!f) bad("missing known descriptor field");
      }
    }
  }
  Result run() {
    if (!plan.algebraic)
      throw Failure(
          "initial index inference requires --semantic-policy=algebraic");
    initialize();
    source = lower(plan.lhs);
    result.trace.push_back("lhs-graph: " + expression(source.value));
    auto lhs = normalize(source.value);
    result.trace.push_back("lhs-normalized: " + normal(lhs));
    checking_source = false;
    auto rhs = normalize(skeleton(plan.rhs));
    correspond(lhs, rhs);
    auto candidate = construct(plan.rhs);
    if (candidate.axes != source.axes)
      throw Failure(
          "candidate output axis order differs; another ordering may require "
          "search");
    // Re-lower from concrete attributes, without trusting constructed axis
    // identities or synthesized graph annotations, before accepting the
    // witness.
    auto checked = lower(plan.rhs);
    auto normalized = normalize(checked.value);
    result.trace.push_back("rhs-graph: " + expression(checked.value));
    result.trace.push_back("rhs-normalized: " + normal(normalized));
    if (!equivalent(source, lhs, checked, normalized))
      throw Failure(
          "independent normalized equivalence check did not establish "
          "equality");
    for (const auto& n : plan.nodes)
      if (n.descriptor && !plan.isCaptured(n)) {
        std::vector<List> fields;
        for (const auto& value : attrs.at(n.descriptor->value)) {
          if (!value) throw Failure("an auto field remains unresolved");
          fields.push_back(*value);
        }
        result.attributes[n.descriptor->value] = fields;
      }
    result.status = Status::Found;
    result.reason =
        "constructed legal attributes; ordered interfaces and normalized "
        "semantics agree";
    result.trace.push_back(
        "verified: legality + ordered shape + capture access + "
        "reduction-domain bijections");
    return result;
  }
};
}  // namespace
Result infer(const Plan& plan, const Metadata& metadata, Limits limits) {
  Engine engine{plan, metadata, limits};
  try {
    return engine.run();
  } catch (const Failure& e) {
    engine.result.status =
        e.illegal ? Status::IllegalMetadata : Status::Unknown;
    engine.result.reason = e.what();
  } catch (const Unbound& e) {
    engine.result.reason = e.what();
  } catch (const std::out_of_range&) {
    engine.result.reason = "missing metadata or invalid symbolic reference";
  }
  engine.result.attributes.clear();
  return engine.result;
}
}  // namespace tepl::core::semantics
