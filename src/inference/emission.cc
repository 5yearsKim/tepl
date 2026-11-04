#include "src/inference/emission.h"
#include "src/inference/extensions.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tepl::inference {
namespace {
struct Builder {
  EmissionBlock block;
  std::size_t add(Op op, std::vector<std::size_t> args = {},
                  std::size_t cap = 0) {
    block.expressions.push_back({op, std::move(args), cap});
    return block.expressions.size() - 1;
  }
  std::size_t cat(std::initializer_list<std::size_t> xs) {
    return add(Op::Concat, xs);
  }
  std::size_t extend(std::size_t q, std::size_t set, std::size_t order) {
    return cat({q, add(Op::Filter, {order, add(Op::Difference, {set, q})})});
  }
  EmissionBlock run(const Program& p) {
    block.source = p;
    const auto& code = p.replacement;
    std::map<std::size_t, std::size_t> input;
    for (auto c : p.source.back().captures) input[c] = add(Op::Input, {}, c);
    auto output = add(Op::Output), empty = add(Op::Empty);
    struct Node {
      std::size_t m, p, b, k, l, r, q;
    };
    std::vector<Node> ns(code.size());
    for (std::size_t i = 0; i < code.size(); ++i) {
      const auto& n = code[i];
      auto& r = ns[i];
      std::vector<std::size_t> inside, outside{output};
      for (auto [cap, x] : input)
        (std::binary_search(n.captures.begin(), n.captures.end(), cap)
             ? inside
             : outside)
            .push_back(x);
      r.m =
          add(Op::Intersect, {add(Op::Union, inside), add(Op::Union, outside)});
      if (n.kind == Kind::Input)
        r.p = input.at(*n.capture);
      else if (n.kind == Kind::Permutation)
        r.p = ns[n.inputs[0]].p;
      else if (n.kind == Kind::Summation)
        r.p = add(Op::Intersect, {ns[n.inputs[0]].p, r.m});
      else {
        auto l = ns[n.inputs[0]].p, rt = ns[n.inputs[1]].p;
        auto shared = add(Op::Intersect, {l, rt});
        r.b = add(Op::Intersect, {shared, r.m});
        r.k = add(Op::Difference, {shared, r.m});
        r.l = add(Op::Difference, {l, rt});
        r.r = add(Op::Difference, {rt, l});
        r.p = add(Op::Union, {r.b, r.l, r.r});
      }
    }
    block.checks.push_back(add(Op::SameSet, {ns.back().p, output}));
    ns.back().q = output;
    for (std::size_t i = code.size(); i-- > 0;) {
      const auto& n = code[i];
      auto& r = ns[i];
      block.checks.push_back(add(Op::Distinct, {r.q}));
      block.checks.push_back(add(Op::Subset, {r.q, r.p}));
      if (n.kind == Kind::Input)
        block.checks.push_back(add(
            Op::Equal, {r.q, add(Op::Filter, {input.at(*n.capture), r.q})}));
      else if (n.kind == Kind::Permutation)
        ns[n.inputs[0]].q = empty;
      else if (n.kind == Kind::Summation)
        ns[n.inputs[0]].q = r.q;
      else {
        auto qb = add(Op::Filter, {r.q, r.b}), ql = add(Op::Filter, {r.q, r.l}),
             qr = add(Op::Filter, {r.q, r.r});
        block.checks.push_back(add(Op::Equal, {r.q, cat({qb, ql, qr})}));
        ns[n.inputs[0]].q = ql;
        ns[n.inputs[1]].q = qr;
      }
    }
    std::vector<std::size_t> layouts;
    for (std::size_t i = 0; i < code.size(); ++i) {
      const auto& n = code[i];
      const auto& r = ns[i];
      std::size_t layout;
      if (n.kind == Kind::Input)
        layout = input.at(*n.capture);
      else {
        auto left = layouts[n.inputs[0]];
        auto& fields = block.attributes[*n.descriptor];
        fields.resize(*std::max_element(n.fields.begin(), n.fields.end()) + 1);
        auto assign = [&](std::size_t field, std::size_t axes,
                          std::size_t group) {
          fields[field] = add(Op::Positions, {axes, group});
        };
        if (n.kind == Kind::Permutation) {
          layout = extend(r.q, r.p, left);
          assign(n.fields[0], left, layout);
        } else if (n.kind == Kind::Summation) {
          auto removed =
              add(Op::Filter,
                  {left, add(Op::Difference, {ns[n.inputs[0]].p, r.p})});
          assign(n.fields[0], left, removed);
          layout = add(Op::Filter, {left, r.p});
        } else {
          auto right = layouts[n.inputs[1]];
          auto batches = extend(add(Op::Filter, {r.q, r.b}), r.b, left);
          auto contracted = add(Op::Filter, {left, r.k});
          assign(n.fields[0], left, batches);
          assign(n.fields[1], right, batches);
          assign(n.fields[2], left, contracted);
          assign(n.fields[3], right, contracted);
          layout = cat({batches, add(Op::Filter, {left, r.l}),
                        add(Op::Filter, {right, r.r})});
        }
      }
      layouts.push_back(layout);
      block.layouts[n.node] = layout;
    }
    return std::move(block);
  }
};
struct Stop {
  Status status;
  const char* reason;
};
struct Interpreter {
  const EmissionBlock& b;
  const SourceNetwork& source;
  const Limits& limits;
  std::size_t& remaining;
  using Value = std::variant<AxisSequence, List, bool>;
  std::map<std::size_t, Value> values;
  void charge(std::size_t n = 1) {
    if (n > remaining) throw Stop{Status::ResourceLimit, "emission work limit"};
    remaining -= n;
  }
  AxisSequence seq(std::size_t id) { return std::get<AxisSequence>(eval(id)); }
  const Value& eval(std::size_t id) {
    if (auto it = values.find(id); it != values.end()) return it->second;
    charge();
    const auto& e = b.expressions.at(id);
    Value result;
    if (e.op == Op::Input)
      result = source.inputs.at(e.capture);
    else if (e.op == Op::Output)
      result = source.output;
    else if (e.op == Op::Empty)
      result = AxisSequence{};
    else {
      AxisSequence a;
      if (!e.args.empty()) a = seq(e.args[0]);
      AxisSequence x;
      if (e.args.size() > 1) x = seq(e.args[1]);
      auto contains = [&](const AxisSequence& s, AxisId v) {
        charge(s.size());
        return std::find(s.begin(), s.end(), v) != s.end();
      };
      if (e.op == Op::Equal)
        result = a == x;
      else if (e.op == Op::Distinct) {
        AxisSet s;
        for (auto v : a) {
          charge();
          s.insert(v);
        }
        result = s.size() == a.size();
      } else if (e.op == Op::Subset || e.op == Op::SameSet) {
        bool yes = true;
        for (auto v : a) yes = contains(x, v) && yes;
        if (e.op == Op::SameSet)
          for (auto v : x) yes = contains(a, v) && yes;
        result = yes;
      } else if (e.op == Op::Positions) {
        List positions;
        AxisSet seen;
        for (auto v : a) {
          charge();
          if (!seen.insert(v).second)
            throw std::logic_error("nonunique emitted interface");
        }
        seen.clear();
        for (auto v : x) {
          charge(a.size());
          auto it = std::find(a.begin(), a.end(), v);
          if (it == a.end() || !seen.insert(v).second)
            throw std::logic_error("undefined emitted position query");
          if (static_cast<std::size_t>(it - a.begin()) >
              static_cast<std::size_t>(
                  std::numeric_limits<std::int64_t>::max()))
            throw Stop{Status::ResourceLimit, "emission index range"};
          positions.push_back(it - a.begin());
        }
        result = std::move(positions);
      } else {
        AxisSequence s;
        if (e.op == Op::Union || e.op == Op::Concat) {
          for (auto arg : e.args)
            for (auto v : seq(arg)) {
              charge();
              if (e.op == Op::Concat || !contains(s, v)) s.push_back(v);
            }
        } else
          for (auto v : a) {
            charge();
            auto present = contains(x, v);
            if ((e.op == Op::Difference ? !present : present)) s.push_back(v);
          }
        result = std::move(s);
      }
    }
    std::visit(
        [&](const auto& v) {
          if constexpr (!std::is_same_v<std::decay_t<decltype(v)>, bool>)
            if (v.size() > limits.axes)
              throw Stop{Status::ResourceLimit, "emission list limit"};
        },
        result);
    return values.emplace(id, std::move(result)).first->second;
  }
};
}  // namespace
Emission emit(const Program& p) {
  validate(p);
  if (p.mode != Mode::Synthesize)
    throw std::invalid_argument("emission requires synthesis mode");
  Emission out;
  out.original = p;
  if (hasExtensions(p)) {
    out.blocks.push_back(emitExtension(p));
    return out;
  }
  if (p.components.empty())
    out.blocks.push_back(Builder{}.run(p));
  else
    for (const auto& c : p.components)
      out.blocks.push_back(Builder{}.run(
          Program{Mode::Synthesize, c.source, c.replacement, {}}));
  return out;
}
namespace {
Result runEmission(const Emission& e, const Metadata& m, const Limits& limits,
                   bool build, bool allow_degenerate = false) {
  if (hasExtensions(e.original)) return executeExtension(e, m, limits, build);
  Result out;
  std::size_t remaining = limits.work;
  try {
    auto original =
        replaySource(e.original, m, {remaining, limits.axes}, allow_degenerate);
    remaining -= original.work;
    if (original.status != Status::Analyzed) return original;
    std::vector<Result> sources;
    std::vector<Interpreter> engines;
    sources.reserve(e.blocks.size());
    engines.reserve(e.blocks.size());
    // Complete every predicate before evaluating any position-builder node.
    for (const auto& block : e.blocks) {
      sources.push_back(replaySource(block.source, m, {remaining, limits.axes},
                                     allow_degenerate));
      auto& source = sources.back();
      remaining -= source.work;
      if (source.status != Status::Analyzed) {
        out = source;
        out.work = limits.work - remaining;
        return out;
      }
      engines.push_back(
          Interpreter{block, source.source, limits, remaining, {}});
      for (auto q : block.checks)
        if (!std::get<bool>(engines.back().eval(q))) {
          out.status =
              allow_degenerate ? Status::Inapplicable : Status::Impossible;
          out.work = limits.work - remaining;
          return out;
        }
    }
    out.status = build ? Status::Solved : Status::Feasible;
    if (build)
      for (std::size_t i = 0; i < engines.size(); ++i) {
        auto& vm = engines[i];
        const auto& block = e.blocks[i];
        Result part;
        part.status = Status::Solved;
        part.source = sources[i].source;
        for (const auto& [descriptor, fields] : block.attributes)
          for (auto field : fields)
            part.attributes[descriptor].push_back(
                std::get<List>(vm.eval(field)));
        for (auto [node, id] : block.layouts) part.layouts[node] = vm.seq(id);
        out.attributes.insert(part.attributes.begin(), part.attributes.end());
        if (e.blocks.size() == 1) {
          out.source = part.source;
          out.layouts = part.layouts;
        } else
          out.components.push_back(std::move(part));
      }
  } catch (const Stop& s) {
    out = {};
    out.status = s.status;
    out.reason = s.reason;
  }
  out.work = limits.work - remaining;
  return out;
}
}  // namespace
Result predicate(const Emission& e, const Metadata& m, const Limits& limits) {
  return runEmission(e, m, limits, false);
}
Result execute(const Emission& e, const Metadata& m, const Limits& limits) {
  return runEmission(e, m, limits, true);
}
Result executeSound(const Emission& e, const Metadata& m,
                    const Limits& limits) {
  return runEmission(e, m, limits, true, true);
}
Portable portable(const Emission& e) {
  Portable p;
  p.program = e.original;
  for (const auto& b : e.blocks) {
    p.nodes.push_back(b.expressions);
    p.checks.push_back(b.checks);
    p.attributes.emplace_back();
    for (const auto& [id, fields] : b.attributes) {
      std::vector<std::size_t> row{id};
      row.insert(row.end(), fields.begin(), fields.end());
      p.attributes.back().push_back(std::move(row));
    }
  }
  return p;
}
bool validatePortable(const Portable& p) {
  try {
    return portable(emit(p.program)) == p;
  } catch (const std::invalid_argument&) {
    return false;
  }
}
}  // namespace tepl::inference
