#include "src/inference/emission.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "src/inference/extensions.h"
#include "src/inference/insertion.h"

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
      else if (n.kind == Kind::Permutation || n.kind == Kind::Reverse)
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
      else if (n.kind == Kind::Reverse)
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
        if (n.kind == Kind::Reverse) {
          layout = left;
          fields[n.fields[0]] = add(Op::ReversalAxes, {}, n.node);
        } else if (n.kind == Kind::Permutation) {
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
    auto reversed = [](const auto& code) {
      return std::any_of(code.begin(), code.end(),
                         [](const auto& n) { return n.kind == Kind::Reverse; });
    };
    if (reversed(p.source) || reversed(p.replacement))
      block.checks.push_back(add(Op::OrientationCheck));
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
  std::map<std::size_t, List> reversal_axes;
  std::optional<bool> orientation_valid;
  void charge(std::size_t n = 1) {
    if (n > remaining) throw Stop{Status::ResourceLimit, "emission work limit"};
    remaining -= n;
  }
  AxisSequence seq(std::size_t id) { return std::get<AxisSequence>(eval(id)); }
  bool orientations() {
    if (orientation_valid) return *orientation_valid;
    const auto& code = b.source.replacement;
    using Vertex = std::pair<std::size_t, AxisId>;
    std::map<Vertex, std::size_t> vertices;
    std::vector<std::optional<bool>> pins;
    std::vector<AxisSequence> layouts;
    std::map<std::size_t, std::size_t> inputs;
    for (const auto& n : code) {
      layouts.push_back(seq(b.layouts.at(n.node)));
      if (n.capture) inputs[*n.capture] = n.node;
      for (auto axis : layouts.back()) {
        charge();
        vertices.emplace(Vertex{n.node, axis}, vertices.size());
        pins.emplace_back();
      }
    }
    std::vector<std::size_t> parent(vertices.size()), rank(vertices.size());
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    std::function<std::size_t(std::size_t)> root = [&](std::size_t v) {
      charge();
      if (parent[v] != v) parent[v] = root(parent[v]);
      return parent[v];
    };
    auto join = [&](std::size_t first, std::size_t second) {
      first = root(first);
      second = root(second);
      if (first == second) return;
      if (rank[first] < rank[second]) std::swap(first, second);
      parent[second] = first;
      if (rank[first] == rank[second]) ++rank[first];
    };
    for (std::size_t i = 0; i < code.size(); ++i) {
      const auto& n = code[i];
      if (n.kind == Kind::Input || n.kind == Kind::Reverse) continue;
      for (auto child : n.inputs)
        for (auto axis : layouts[child]) {
          charge();
          if (vertices.contains({n.node, axis}))
            join(vertices.at({n.node, axis}),
                 vertices.at({code[child].node, axis}));
        }
      if (n.kind == Kind::Contraction)
        for (auto axis : layouts[n.inputs[0]]) {
          charge();
          auto other = vertices.find({code[n.inputs[1]].node, axis});
          if (other != vertices.end())
            join(vertices.at({code[n.inputs[0]].node, axis}), other->second);
        }
    }
    for (std::size_t i = 0; i < source.ports.size(); ++i) {
      charge();
      pins[vertices.at({inputs.at(source.ports[i].capture),
                        source.classes[i]})] = source.port_orientations.at(i);
    }
    for (std::size_t i = 0; i < source.output.size(); ++i) {
      charge();
      auto v = vertices.at({code.back().node, source.output[i]});
      auto bit = source.output_orientations.at(i);
      if (pins[v] && *pins[v] != bit) {
        orientation_valid = false;
        return false;
      }
      pins[v] = bit;
    }
    std::vector<std::optional<bool>> fixed(vertices.size());
    for (std::size_t i = 0; i < pins.size(); ++i) {
      if (!pins[i]) continue;
      auto r = root(i);
      if (fixed[r] && fixed[r] != pins[i]) {
        orientation_valid = false;
        return false;
      }
      fixed[r] = pins[i];
    }
    for (std::size_t i = 0; i < code.size(); ++i)
      if (code[i].kind == Kind::Reverse) {
        auto child = code[i].inputs[0];
        auto& axes = reversal_axes[code[i].node];
        for (std::size_t j = 0; j < layouts[child].size(); ++j) {
          auto axis = layouts[child][j];
          auto before = root(vertices.at({code[child].node, axis}));
          auto after = root(vertices.at({code[i].node, axis}));
          if (fixed[before].value_or(false) != fixed[after].value_or(false))
            axes.push_back(j);
        }
      }
    orientation_valid = true;
    return true;
  }
  const Value& eval(std::size_t id) {
    if (auto it = values.find(id); it != values.end()) return it->second;
    charge();
    const auto& e = b.expressions.at(id);
    Value result;
    if (e.op == Op::OrientationCheck)
      result = orientations();
    else if (e.op == Op::ReversalAxes)
      result = reversal_axes.at(e.capture);
    else if (e.op == Op::Input)
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
EmissionBlock emitCoreBlock(const Program& p) { return Builder{}.run(p); }
Emission emit(const Program& p) {
  validate(p);
  if (p.mode != Mode::Synthesize)
    throw std::invalid_argument("emission requires synthesis mode");
  Emission out;
  out.original = p;
  if (p.restricted_insertion) return emitInsertion(p);
  if (usesSpecializedSchema(p)) {
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
Result runCore(const Program& program, std::span<const EmissionBlock> blocks,
               const Metadata& m, const Limits& limits, bool build,
               bool allow_degenerate, ExecutionTiming* timing) {
  Result out;
  std::size_t remaining = limits.work;
  try {
    auto original =
        replaySource(program, m, {remaining, limits.axes}, allow_degenerate);
    remaining -= original.work;
    if (original.status != Status::Analyzed) return original;
    std::vector<Result> sources;
    std::vector<Interpreter> engines;
    sources.reserve(blocks.size());
    engines.reserve(blocks.size());
    // Complete every predicate before evaluating any position-builder node.
    for (const auto& block : blocks) {
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
          Interpreter{block, source.source, limits, remaining, {}, {}, {}});
      detail::PhaseTimer phase(timing ? &timing->feasibility_us : nullptr);
      for (auto q : block.checks)
        if (!std::get<bool>(engines.back().eval(q))) {
          out.status =
              allow_degenerate ? Status::Inapplicable : Status::Impossible;
          out.work = limits.work - remaining;
          return out;
        }
    }
    out.status = build ? Status::Solved : Status::Feasible;
    if (build) {
      detail::PhaseTimer phase(timing ? &timing->construction_us : nullptr);
      for (std::size_t i = 0; i < engines.size(); ++i) {
        auto& vm = engines[i];
        const auto& block = blocks[i];
        Result part;
        part.status = Status::Solved;
        part.source = sources[i].source;
        for (const auto& [descriptor, fields] : block.attributes)
          for (auto field : fields)
            part.attributes[descriptor].push_back(
                std::get<List>(vm.eval(field)));
        for (auto [node, id] : block.layouts) part.layouts[node] = vm.seq(id);
        out.attributes.insert(part.attributes.begin(), part.attributes.end());
        if (blocks.size() == 1) {
          out.source = part.source;
          out.layouts = part.layouts;
        } else
          out.components.push_back(std::move(part));
      }
    }
  } catch (const Stop& s) {
    out = {};
    out.status = s.status;
    out.reason = s.reason;
  }
  out.work = limits.work - remaining;
  return out;
}
Result runEmission(const Emission& e, const Metadata& m, const Limits& limits,
                   bool build, bool allow_degenerate = false,
                   ExecutionTiming* timing = nullptr) {
  if (timing) *timing = {};
  if (e.insertion)
    return executeInsertion(e, m, limits, build, allow_degenerate, timing);
  if (usesSpecializedSchema(e.original))
    return executeExtension(e, m, limits, build, timing);
  return runCore(e.original, e.blocks, m, limits, build, allow_degenerate,
                 timing);
}
}  // namespace
Result executeCoreBlock(const EmissionBlock& block, const Metadata& m,
                        const Limits& limits, bool build,
                        ExecutionTiming* timing) {
  return runCore(block.source, std::span(&block, 1), m, limits, build, false,
                 timing);
}
Result predicate(const Emission& e, const Metadata& m, const Limits& limits) {
  return runEmission(e, m, limits, false);
}
Result execute(const Emission& e, const Metadata& m, const Limits& limits) {
  return runEmission(e, m, limits, true);
}
Result executeSound(const Emission& e, const Metadata& m, const Limits& limits,
                    ExecutionTiming* timing) {
  return runEmission(e, m, limits, true, true, timing);
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
