#include "src/inference/evaluate.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "src/inference/extensions.h"

namespace tepl::inference {
namespace {
struct Failure {
  Status status;
  std::string reason;
};
struct Work {
  std::size_t remaining, initial, max_axes;
  explicit Work(const Limits& limits)
      : remaining(limits.work), initial(limits.work), max_axes(limits.axes) {}
  void size(std::size_t n) {
    if (n > max_axes)
      throw Failure{Status::ResourceLimit, "ordered axis limit"};
  }
  void charge(std::size_t count) {
    if (count > remaining)
      throw Failure{Status::ResourceLimit, "ordered work limit"};
    remaining -= count;
  }
  void tick() { charge(1); }
};

AxisSet set(const AxisSequence& a, Work& work) {
  work.charge(a.size());
  return {a.begin(), a.end()};
}
AxisSet intersect(const AxisSet& a, const AxisSet& b, Work& work) {
  AxisSet out;
  for (auto x : a) {
    work.tick();
    if (b.contains(x)) out.insert(x);
  }
  return out;
}
AxisSet difference(const AxisSet& a, const AxisSet& b, Work& work) {
  AxisSet out;
  for (auto x : a) {
    work.tick();
    if (!b.contains(x)) out.insert(x);
  }
  return out;
}
AxisSequence filter(const AxisSequence& a, const AxisSet& b, Work& work) {
  AxisSequence out;
  for (auto x : a) {
    work.tick();
    if (b.contains(x)) out.push_back(x);
  }
  return out;
}
void append(AxisSequence& a, const AxisSequence& b, Work& work) {
  work.charge(b.size());
  a.insert(a.end(), b.begin(), b.end());
}
AxisSequence extend(const AxisSequence& q, const AxisSet& axes,
                    const AxisSequence& order, Work& work) {
  work.charge(q.size());
  auto out = q;
  append(out, filter(order, difference(axes, set(q, work), work), work), work);
  return out;
}
List positions(const AxisSequence& layout, const AxisSequence& axes,
               Work& work) {
  std::map<AxisId, std::int64_t> index;
  for (std::size_t i = 0; i < layout.size(); ++i) {
    work.tick();
    index.emplace(layout[i], i);
  }
  List out;
  for (auto a : axes) {
    work.tick();
    auto it = index.find(a);
    if (it == index.end())
      throw std::logic_error("ordered construction lost a live axis");
    out.push_back(it->second);
  }
  return out;
}

// Pass 1 (and independent RHS validation): interpret actual positional fields.
struct Replay {
  const Attributes& attributes;
  Work& work;
  Status invalid;
  SourceNetwork network;
  AxisSequence parent;
  std::vector<bool> parity;
  std::map<std::size_t, AxisSequence> outputs;
  std::map<std::size_t, std::vector<AxisSequence>> groups;

  Replay(const Program& p, const Metadata& m, const Attributes& a, Work& w,
         Status error = Status::InvalidSource,
         std::uint64_t minimum_extent = DomainRequirements::minimum_extent)
      : attributes(a), work(w), invalid(error) {
    for (auto capture : p.source.back().captures) {
      work.tick();
      auto it = m.shapes.find(capture);
      if (it == m.shapes.end())
        throw Failure{Status::MissingMetadata, "missing capture shape"};
      work.size(it->second.size());
      auto& layout = network.inputs[capture];
      for (std::size_t i = 0; i < it->second.size(); ++i) {
        work.tick();
        work.size(network.ports.size() + 1);
        if (it->second[i] < minimum_extent)
          throw Failure{Status::OutsideDomain, "ordered extents must be >= 2"};
        layout.push_back(network.ports.size());
        network.ports.push_back({capture, i, it->second[i]});
      }
    }
    parent.resize(network.ports.size());
    parity.resize(parent.size());
    std::iota(parent.begin(), parent.end(), AxisId{0});
  }
  std::pair<AxisId, bool> signedRoot(AxisId a) {
    auto r = a;
    bool bit = false;
    while (parent[r] != r) {
      work.tick();
      bit = bit != parity[r];
      r = parent[r];
    }
    const bool total = bit;
    while (parent[a] != a) {
      work.tick();
      auto next = parent[a];
      bool edge = parity[a];
      parent[a] = r;
      parity[a] = bit;
      bit = bit != edge;
      a = next;
    }
    return {r, total};
  }
  AxisId root(AxisId a) { return signedRoot(a).first; }
  void join(AxisId a, AxisId b, bool left_bit, bool right_bit) {
    auto [ar, ap] = signedRoot(a);
    auto [br, bp] = signedRoot(b);
    a = ar;
    b = br;
    if (network.ports[a].extent != network.ports[b].extent)
      throw Failure{invalid, "paired extents differ"};
    if (a == b) throw Failure{invalid, "source contains a reflection cycle"};
    auto high = std::max(a, b);
    parent[high] = std::min(a, b);
    parity[high] = (left_bit != right_bit) != (ap != bp);
  }
  void selection(const List& a, std::size_t rank, AxisSet& used) {
    work.size(a.size());
    for (auto x : a) {
      work.tick();
      if (x < 0 || static_cast<std::uint64_t>(x) >= rank ||
          !used.insert(x).second)
        throw Failure{invalid, "invalid/overlapping axis list"};
    }
  }
  SourceNetwork run(const std::vector<Instruction>& code) {
    std::vector<AxisSequence> layouts(code.size());
    std::vector<std::vector<bool>> bits(code.size());
    for (std::size_t id = 0; id < code.size(); ++id) {
      work.tick();
      const auto& n = code[id];
      auto& out = layouts[id];
      auto& output_bits = bits[id];
      if (n.kind == Kind::Input) {
        out = network.inputs.at(*n.capture);
        output_bits.resize(out.size());
      } else if (n.kind == Kind::Addition) {
        const auto& left = layouts[n.inputs[0]];
        const auto& right = layouts[n.inputs[1]];
        if (left.size() != right.size())
          throw Failure{invalid, "addition operand ranks differ"};
        for (std::size_t i = 0; i < left.size(); ++i) {
          work.tick();
          if (network.ports[left[i]].extent != network.ports[right[i]].extent)
            throw Failure{invalid, "addition operand shapes differ"};
        }
        // This schedule checks source legality only. Additive equivalence is
        // checked in separate component networks; never join their ports here.
        out = left;
        output_bits = bits[n.inputs[0]];
        groups[n.node].resize(5);
        groups[n.node][4] = out;
      } else {
        auto it = attributes.find(*n.descriptor);
        if (it == attributes.end())
          throw Failure{Status::MissingMetadata,
                        "missing operation descriptor"};
        const auto& fields = it->second;
        if (fields.size() != n.fields.size())
          throw Failure{invalid, "descriptor field count differs"};
        const auto& left = layouts[n.inputs[0]];
        const auto& left_bits = bits[n.inputs[0]];
        auto& g = groups[n.node];
        g.resize(5);
        AxisSet left_used, right_used;
        if (n.kind == Kind::Permutation) {
          g[4] = left;
          const auto& permutation = fields[n.fields[0]];
          selection(permutation, left.size(), left_used);
          if (permutation.size() != left.size())
            throw Failure{invalid, "permutation is not full rank"};
          for (auto i : permutation) {
            out.push_back(left[i]);
            output_bits.push_back(left_bits[i]);
          }
        } else if (n.kind == Kind::Reverse) {
          selection(fields[n.fields[0]], left.size(), left_used);
          out = left;
          output_bits = left_bits;
          for (auto i : left_used) output_bits[i] = !output_bits[i];
        } else if (n.kind == Kind::Summation) {
          const auto& dimensions = fields[n.fields[0]];
          selection(dimensions, left.size(), left_used);
          for (auto i : dimensions) g[1].push_back(left[i]);
          for (std::size_t i = 0; i < left.size(); ++i) {
            work.tick();
            if (!left_used.contains(i)) {
              out.push_back(left[i]);
              output_bits.push_back(left_bits[i]);
              g[2].push_back(left[i]);
            }
          }
        } else {
          const auto& right = layouts[n.inputs[1]];
          const auto& right_bits = bits[n.inputs[1]];
          const auto& lb = fields[n.fields[0]];
          const auto& rb = fields[n.fields[1]];
          const auto& lk = fields[n.fields[2]];
          const auto& rk = fields[n.fields[3]];
          selection(lb, left.size(), left_used);
          selection(lk, left.size(), left_used);
          selection(rb, right.size(), right_used);
          selection(rk, right.size(), right_used);
          if (lb.size() != rb.size() || lk.size() != rk.size())
            throw Failure{invalid, "paired lists have unequal lengths"};
          for (std::size_t i = 0; i < lb.size(); ++i)
            join(left[lb[i]], right[rb[i]], left_bits[lb[i]],
                 right_bits[rb[i]]);
          for (std::size_t i = 0; i < lk.size(); ++i)
            join(left[lk[i]], right[rk[i]], left_bits[lk[i]],
                 right_bits[rk[i]]);
          for (auto i : lb) {
            out.push_back(left[i]);
            output_bits.push_back(left_bits[i]);
            g[0].push_back(left[i]);
          }
          for (auto i : lk) g[1].push_back(left[i]);
          for (std::size_t i = 0; i < left.size(); ++i) {
            work.tick();
            if (!left_used.contains(i)) {
              out.push_back(left[i]);
              output_bits.push_back(left_bits[i]);
              g[2].push_back(left[i]);
            }
          }
          for (std::size_t i = 0; i < right.size(); ++i) {
            work.tick();
            if (!right_used.contains(i)) {
              out.push_back(right[i]);
              output_bits.push_back(right_bits[i]);
              g[3].push_back(right[i]);
            }
          }
        }
      }
      outputs[n.node] = out;
    }
    network.output = layouts.back();
    for (std::size_t i = 0; i < parent.size(); ++i) {
      auto [r, bit] = signedRoot(i);
      network.classes.push_back(r);
      network.port_orientations.push_back(bit);
    }
    for (std::size_t i = 0; i < network.output.size(); ++i)
      network.output_orientations.push_back(
          bits.back()[i] != signedRoot(network.output[i]).second);
    auto normalize = [&](AxisSequence& a) {
      for (auto& x : a) {
        work.tick();
        x = root(x);
      }
    };
    normalize(network.output);
    for (auto& [_, a] : network.inputs) normalize(a);
    for (auto& [_, a] : outputs) normalize(a);
    for (auto& [_, values] : groups)
      for (auto& a : values) normalize(a);
    return std::move(network);
  }
};

struct Analysis {
  const Program& program;
  Work& work;
  Result result;
  std::map<AxisId, AxisSet> support;
  AxisSet final;
  Analysis(const Program& p, Work& w, SourceNetwork source)
      : program(p), work(w) {
    result.source = std::move(source);
    final = set(result.source.output, work);
    for (std::size_t i = 0; i < result.source.ports.size(); ++i) {
      work.tick();
      support[result.source.classes[i]].insert(result.source.ports[i].capture);
    }
  }
  // Pass 2: all capture sets were computed at compilation, not at each match.
  void boundaries() {
    for (const auto& n : program.replacement) {
      work.tick();
      auto& r = result.requirements[n.node];
      work.charge(n.captures.size());
      AxisSet captures(n.captures.begin(), n.captures.end());
      for (const auto& [a, s] : support) {
        work.tick();
        if (!intersect(s, captures, work).empty() &&
            (final.contains(a) || !difference(s, captures, work).empty()))
          r.mandatory_axes.insert(a);
      }
      if (n.kind == Kind::Contraction) {
        const auto& left =
            result.requirements.at(program.replacement[n.inputs[0]].node)
                .retained_axes;
        const auto& right =
            result.requirements.at(program.replacement[n.inputs[1]].node)
                .retained_axes;
        auto shared = intersect(left, right, work);
        r.batch = intersect(shared, r.mandatory_axes, work);
        r.contract = difference(shared, r.mandatory_axes, work);
        r.left_free = difference(left, right, work);
        r.right_free = difference(right, left, work);
        r.retained_axes = r.batch;
        r.retained_axes.insert(r.left_free.begin(), r.left_free.end());
        r.retained_axes.insert(r.right_free.begin(), r.right_free.end());
      } else if (n.kind == Kind::Input) {
        r.retained_axes = set(result.source.inputs.at(*n.capture), work);
      } else {
        const auto& child =
            result.requirements.at(program.replacement[n.inputs[0]].node)
                .retained_axes;
        r.retained_axes = n.kind == Kind::Summation
                              ? intersect(child, r.mandatory_axes, work)
                              : child;
        if (n.kind == Kind::Summation) {
          r.contract = difference(child, r.retained_axes, work);
          r.left_free = r.retained_axes;
        }
      }
    }
  }
  // Pass 3: reverse postorder is a parent-before-child schedule for a tree.
  void ordering() {
    result.requirements[program.replacement.back().node].required_order =
        result.source.output;
    for (auto it = program.replacement.rbegin();
         it != program.replacement.rend(); ++it) {
      work.tick();
      const auto& n = *it;
      const auto& r = result.requirements.at(n.node);
      if (n.kind == Kind::Permutation)
        result.requirements[program.replacement[n.inputs[0]].node]
            .required_order = {};
      if (n.kind == Kind::Summation)
        result.requirements[program.replacement[n.inputs[0]].node]
            .required_order = r.required_order;
      if (n.kind == Kind::Contraction) {
        result.requirements[program.replacement[n.inputs[0]].node]
            .required_order = filter(r.required_order, r.left_free, work);
        result.requirements[program.replacement[n.inputs[1]].node]
            .required_order = filter(r.required_order, r.right_free, work);
      }
    }
  }
  // Pass 4: a local descending pair is an exact negative witness.
  std::optional<Obstruction> obstruction() {
    std::map<std::size_t, const Instruction*> instructions;
    for (const auto& n : program.replacement) instructions[n.node] = &n;
    const auto root = program.replacement.back().node;
    const auto& retained = result.requirements.at(root).retained_axes;
    if (retained != final) {
      auto extra = difference(retained, final, work);
      auto missing = difference(final, retained, work);
      auto axis = extra.empty() ? *missing.begin() : *extra.begin();
      return Obstruction{root, axis, axis, Obstruction::Kind::RetainedSet};
    }
    for (const auto& [id, r] : result.requirements) {
      work.tick();
      auto q = set(r.required_order, work);
      if (q.size() != r.required_order.size() ||
          !difference(q, r.retained_axes, work).empty()) {
        auto extra = difference(q, r.retained_axes, work);
        auto axis = extra.empty() ? r.required_order.front() : *extra.begin();
        return Obstruction{id, axis, axis, Obstruction::Kind::Requirement};
      }
      const auto& n = *instructions.at(id);
      if (n.kind == Kind::Permutation || n.kind == Kind::Summation ||
          r.required_order.size() < 2)
        continue;
      std::map<AxisId, std::size_t> input_positions;
      if (n.capture) {
        const auto& order = result.source.inputs.at(*n.capture);
        for (std::size_t i = 0; i < order.size(); ++i) {
          work.tick();
          input_positions.emplace(order[i], i);
        }
      }
      auto role = [&](AxisId a) -> std::size_t {
        if (n.capture) return input_positions.at(a);
        return r.batch.contains(a) ? 0 : r.left_free.contains(a) ? 1 : 2;
      };
      for (std::size_t i = 1; i < r.required_order.size(); ++i) {
        work.tick();
        if (role(r.required_order[i - 1]) > role(r.required_order[i]))
          return Obstruction{id, r.required_order[i - 1], r.required_order[i]};
      }
    }
    return {};
  }
  // Pass 5: runs only after pass 4 succeeds and reuses its complete analysis.
  void construct() {
    for (const auto& n : program.replacement) {
      work.tick();
      auto& output = result.layouts[n.node];
      if (n.kind == Kind::Input) {
        output = result.source.inputs.at(*n.capture);
        continue;
      }
      const auto& r = result.requirements.at(n.node);
      const auto& left =
          result.layouts.at(program.replacement[n.inputs[0]].node);
      auto& fields = result.attributes[*n.descriptor];
      fields.resize(n.fields.size());
      if (n.kind == Kind::Permutation) {
        output = extend(r.required_order, r.retained_axes, left, work);
        fields[n.fields[0]] = positions(left, output, work);
      } else if (n.kind == Kind::Summation) {
        auto removed = difference(set(left, work), r.retained_axes, work);
        fields[n.fields[0]] =
            positions(left, filter(left, removed, work), work);
        output = filter(left, r.retained_axes, work);
      } else {
        const auto& right =
            result.layouts.at(program.replacement[n.inputs[1]].node);
        auto b = extend(filter(r.required_order, r.batch, work), r.batch,
                        AxisSequence(r.batch.begin(), r.batch.end()), work);
        AxisSequence k(r.contract.begin(), r.contract.end());
        fields[n.fields[0]] = positions(left, b, work);
        fields[n.fields[1]] = positions(right, b, work);
        fields[n.fields[2]] = positions(left, k, work);
        fields[n.fields[3]] = positions(right, k, work);
        output = b;
        append(output, filter(left, r.left_free, work), work);
        append(output, filter(right, r.right_free, work), work);
      }
    }
  }
};

Attributes explicitAttributes(const Program& p, const Metadata& m) {
  Attributes out;
  for (const auto& n : p.replacement)
    if (n.descriptor) {
      auto it = m.descriptors.find(*n.descriptor);
      if (it == m.descriptors.end())
        throw Failure{Status::MissingMetadata, "missing RHS descriptor"};
      out.emplace(*n.descriptor, it->second);
    }
  return out;
}

Result run(const Program& p, const Metadata& m, const Limits& limits,
           bool construct, bool analyze_only = false) {
  if (hasExtensions(p)) {
    Result r;
    r.status = Status::Unsupported;
    r.reason = "schema extensions require schema emission";
    return r;
  }
  Work work(limits);
  Result out;
  try {
    validate(p);
    Replay replay(p, m, m.descriptors, work);
    auto source = replay.run(p.source);
    if (!p.components.empty()) {
      // The original schedule establishes legality, not a shared additive
      // axis network. Only the component results expose equivalence classes.
      out.status = analyze_only ? Status::Analyzed
                   : construct  ? Status::Solved
                                : Status::Feasible;
      if (!analyze_only && p.mode == Mode::Validate) {
        auto attrs = explicitAttributes(p, m);
        Replay rhs(p, m, attrs, work, Status::InvalidReplacement);
        rhs.run(p.replacement);  // Check every actual addition's shape.
        out.status = Status::Solved;
      }
      for (const auto& component : p.components) {
        Program single{p.mode, component.source, component.replacement, {}};
        auto r = run(single, m, {work.remaining, work.max_axes}, construct,
                     analyze_only);
        work.charge(r.work);
        if (r.status != out.status) {
          out.status = r.status;
          out.reason = r.reason;
        }
        out.components.push_back(std::move(r));
        if (out.status != Status::Analyzed && out.status != Status::Solved &&
            out.status != Status::Feasible)
          break;
      }
      if (out.status == Status::Solved)
        for (const auto& r : out.components)
          out.attributes.insert(r.attributes.begin(), r.attributes.end());
    } else if (analyze_only) {
      Analysis a(p, work, std::move(source));
      if (p.mode == Mode::Synthesize) {
        a.boundaries();
        a.ordering();
      }
      a.result.replay_layouts = std::move(replay.outputs);
      a.result.replay_groups = std::move(replay.groups);
      a.result.status = Status::Analyzed;
      out = std::move(a.result);
    } else if (p.mode == Mode::Validate) {
      auto attrs = explicitAttributes(p, m);
      Replay rhs(p, m, attrs, work, Status::InvalidReplacement);
      auto target = rhs.run(p.replacement);
      out.source = std::move(source);
      if (out.source.classes != target.classes ||
          out.source.output != target.output) {
        out.status = Status::NotEquivalent;
        out.reason =
            "supplied RHS attributes change the axis network or output order";
      } else {
        out.status = Status::Solved;
        out.attributes = std::move(attrs);
        out.layouts = std::move(rhs.outputs);
      }
    } else {
      Analysis a(p, work, std::move(source));
      a.boundaries();
      a.ordering();
      if (auto witness = a.obstruction()) {
        a.result.status = Status::Impossible;
        a.result.reason =
            "retained axes or required relative order are incompatible";
        a.result.obstruction = witness;
      } else if (construct) {
        a.construct();
        a.result.status = Status::Solved;
      } else
        a.result.status = Status::Feasible;
      out = std::move(a.result);
    }
  } catch (const Failure& f) {
    out = {};
    out.status = f.status;
    out.reason = f.reason;
  } catch (const std::invalid_argument& e) {
    out = {};
    out.reason = e.what();
  }
  out.work = work.initial - work.remaining;
  return out;
}
}  // namespace

Result replaySource(const Program& p, const Metadata& m, const Limits& limits,
                    bool allow_degenerate) {
  auto unsupported = [](const auto& code) {
    return std::any_of(code.begin(), code.end(), [](const auto& n) {
      return n.kind != Kind::Input && n.kind != Kind::Permutation &&
             n.kind != Kind::Summation && n.kind != Kind::Contraction &&
             n.kind != Kind::Addition && n.kind != Kind::Reverse;
    });
  };
  if (unsupported(p.source) || unsupported(p.replacement)) {
    Result r;
    r.status = Status::Unsupported;
    r.reason = "schema extensions require schema emission";
    return r;
  }
  Work work(limits);
  Result out;
  try {
    validate(p);
    Replay replay(p, m, m.descriptors, work, Status::InvalidSource,
                  allow_degenerate ? 0 : DomainRequirements::minimum_extent);
    out.source = replay.run(p.source);
    out.replay_layouts = std::move(replay.outputs);
    out.replay_groups = std::move(replay.groups);
    out.status = Status::Analyzed;
  } catch (const Failure& f) {
    out.status = f.status;
    out.reason = f.reason;
  } catch (const std::invalid_argument& e) {
    out.reason = e.what();
  }
  out.work = work.initial - work.remaining;
  return out;
}

Result evaluate(const Program& p, const Metadata& m, const Limits& limits) {
  return run(p, m, limits, true);
}
Result analyze(const Program& p, const Metadata& m, const Limits& limits) {
  return run(p, m, limits, false, true);
}
Result construct(const Program& p, const Result& analysis,
                 const Limits& limits) {
  if (hasExtensions(p)) {
    Result r;
    r.status = Status::Unsupported;
    r.reason = "schema extensions require schema emission";
    return r;
  }
  Work work(limits);
  Result out;
  try {
    if (p.mode != Mode::Synthesize || analysis.status != Status::Analyzed)
      throw std::invalid_argument(
          "constructor requires analyzed synthesis metadata");
    validate(p);
    if (!p.components.empty()) {
      if (analysis.components.size() != p.components.size())
        throw std::invalid_argument("missing component analysis");
      out.status = Status::Solved;
      for (std::size_t i = 0; i < p.components.size(); ++i) {
        const auto& component = p.components[i];
        Program single{p.mode, component.source, component.replacement, {}};
        auto r = inference::construct(single, analysis.components[i],
                                      {work.remaining, work.max_axes});
        work.charge(r.work);
        out.components.push_back(std::move(r));
        if (out.components.back().status != Status::Solved) {
          out.status = out.components.back().status;
          out.reason = out.components.back().reason;
          break;
        }
      }
      if (out.status == Status::Solved)
        for (const auto& r : out.components)
          out.attributes.insert(r.attributes.begin(), r.attributes.end());
      out.work = work.initial - work.remaining;
      return out;
    }
    Analysis a(p, work, analysis.source);
    a.boundaries();
    a.ordering();
    if (a.result.requirements != analysis.requirements)
      throw std::invalid_argument("inconsistent analyzed requirements");
    if (auto witness = a.obstruction()) {
      a.result.status = Status::Impossible;
      a.result.reason =
          "retained axes or required relative order are incompatible";
      a.result.obstruction = witness;
      a.result.work = work.initial - work.remaining;
      return std::move(a.result);
    }
    a.construct();
    a.result.status = Status::Solved;
    out = std::move(a.result);
  } catch (const Failure& f) {
    out = {};
    out.status = f.status;
    out.reason = f.reason;
  } catch (const std::invalid_argument& e) {
    out = {};
    out.reason = e.what();
  }
  out.work = work.initial - work.remaining;
  return out;
}
Result replayReplacement(const Program& p, const Metadata& m,
                         const Limits& limits) {
  if (hasExtensions(p)) {
    Result r;
    r.status = Status::Unsupported;
    r.reason = "schema extensions require schema emission";
    return r;
  }
  Work work(limits);
  Result out;
  try {
    validate(p);
    auto attrs = explicitAttributes(p, m);
    Replay replay(p, m, attrs, work, Status::InvalidReplacement);
    auto network = replay.run(p.replacement);
    if (p.components.empty()) {
      out.source = std::move(network);
      out.replay_layouts = replay.outputs;
      out.replay_groups = std::move(replay.groups);
      out.layouts = std::move(replay.outputs);
    } else {
      for (const auto& component : p.components) {
        Program single{p.mode, component.source, component.replacement, {}};
        auto r = replayReplacement(single, m, {work.remaining, work.max_axes});
        work.charge(r.work);
        if (r.status != Status::Analyzed) throw Failure{r.status, r.reason};
        out.components.push_back(std::move(r));
      }
    }
    out.attributes = std::move(attrs);
    out.status = Status::Analyzed;
  } catch (const Failure& f) {
    out = {};
    out.status = f.status;
    out.reason = f.reason;
  } catch (const std::invalid_argument& e) {
    out = {};
    out.reason = e.what();
  }
  out.work = work.initial - work.remaining;
  return out;
}
Result feasibility(const Program& p, const Metadata& m, const Limits& limits) {
  return run(p, m, limits, false);
}
bool check(const Program& p, const Metadata& m, const Result& r) {
  Work work({});
  try {
    validate(p);
    Replay replay(p, m, m.descriptors, work);
    auto source = replay.run(p.source);
    if (!p.components.empty()) {
      if (r.source != SourceNetwork{}) return false;
      if (!r.requirements.empty() || !r.layouts.empty() || r.obstruction ||
          r.components.empty() || r.components.size() > p.components.size())
        return false;
      Attributes attributes;
      bool impossible = false;
      for (std::size_t i = 0; i < r.components.size(); ++i) {
        const auto& component = p.components[i];
        const auto& certificate = r.components[i];
        Program single{p.mode, component.source, component.replacement, {}};
        if (!check(single, m, certificate)) return false;
        impossible |= certificate.status == Status::Impossible;
        if (r.status != Status::Impossible && certificate.status != r.status)
          return false;
        attributes.insert(certificate.attributes.begin(),
                          certificate.attributes.end());
      }
      if (r.status == Status::Impossible)
        return impossible && r.attributes.empty();
      if (r.components.size() != p.components.size() ||
          r.attributes != attributes)
        return false;
      if (r.status == Status::Solved) {
        Replay rhs(p, m, r.attributes, work, Status::InvalidReplacement);
        rhs.run(p.replacement);
        return true;
      }
      return r.status == Status::Feasible;
    }
    if (source != r.source) return false;
    if (p.mode == Mode::Validate) {
      if (r.status != Status::Solved || !r.requirements.empty() ||
          r.attributes != explicitAttributes(p, m))
        return false;
    } else {
      Analysis a(p, work, source);
      a.boundaries();
      a.ordering();
      if (a.result.requirements != r.requirements) return false;
      if (r.status == Status::Impossible)
        return r.attributes.empty() && r.layouts.empty() && r.obstruction &&
               a.obstruction() == r.obstruction;
      if (r.status == Status::Feasible)
        return r.attributes.empty() && r.layouts.empty() && !r.obstruction &&
               !a.obstruction();
      if (r.status != Status::Solved) return false;
    }
    if (r.obstruction) return false;
    Replay rhs(p, m, r.attributes, work, Status::InvalidReplacement);
    auto target = rhs.run(p.replacement);
    return source.classes == target.classes && source.output == target.output &&
           rhs.outputs == r.layouts;
  } catch (const Failure&) {
    return false;
  } catch (const std::invalid_argument&) {
    return false;
  } catch (const std::out_of_range&) {
    return false;
  }
}
}  // namespace tepl::inference
