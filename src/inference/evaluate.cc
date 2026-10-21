#include "src/inference/evaluate.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

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
  std::map<std::size_t, AxisSequence> outputs;

  Replay(const Program& p, const Metadata& m, const Attributes& a, Work& w,
         Status error = Status::InvalidSource)
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
        if (it->second[i] < 2)
          throw Failure{Status::OutsideDomain, "ordered extents must be >= 2"};
        layout.push_back(network.ports.size());
        network.ports.push_back({capture, i, it->second[i]});
      }
    }
    parent.resize(network.ports.size());
    std::iota(parent.begin(), parent.end(), AxisId{0});
  }
  AxisId root(AxisId a) {
    auto r = a;
    while (parent[r] != r) {
      work.tick();
      r = parent[r];
    }
    while (parent[a] != a) {
      work.tick();
      auto next = parent[a];
      parent[a] = r;
      a = next;
    }
    return r;
  }
  void join(AxisId a, AxisId b) {
    a = root(a);
    b = root(b);
    if (network.ports[a].extent != network.ports[b].extent)
      throw Failure{invalid, "paired extents differ"};
    parent[std::max(a, b)] = std::min(a, b);
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
    for (std::size_t id = 0; id < code.size(); ++id) {
      work.tick();
      const auto& n = code[id];
      auto& out = layouts[id];
      if (n.kind == Kind::Input) {
        out = network.inputs.at(*n.capture);
      } else {
        auto it = attributes.find(*n.descriptor);
        if (it == attributes.end())
          throw Failure{Status::MissingMetadata,
                        "missing operation descriptor"};
        const auto& fields = it->second;
        if (fields.size() != n.fields.size())
          throw Failure{invalid, "descriptor field count differs"};
        const auto& left = layouts[n.inputs[0]];
        AxisSet left_used, right_used;
        if (n.kind == Kind::Permutation) {
          const auto& permutation = fields[n.fields[0]];
          selection(permutation, left.size(), left_used);
          if (permutation.size() != left.size())
            throw Failure{invalid, "permutation is not full rank"};
          for (auto i : permutation) out.push_back(left[i]);
        } else {
          const auto& right = layouts[n.inputs[1]];
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
            join(left[lb[i]], right[rb[i]]);
          for (std::size_t i = 0; i < lk.size(); ++i)
            join(left[lk[i]], right[rk[i]]);
          for (auto i : lb) out.push_back(left[i]);
          for (std::size_t i = 0; i < left.size(); ++i) {
            work.tick();
            if (!left_used.contains(i)) out.push_back(left[i]);
          }
          for (std::size_t i = 0; i < right.size(); ++i) {
            work.tick();
            if (!right_used.contains(i)) out.push_back(right[i]);
          }
        }
      }
      outputs[n.node] = out;
    }
    network.output = layouts.back();
    for (std::size_t i = 0; i < parent.size(); ++i)
      network.classes.push_back(root(i));
    auto normalize = [&](AxisSequence& a) {
      for (auto& x : a) {
        work.tick();
        x = root(x);
      }
    };
    normalize(network.output);
    for (auto& [_, a] : network.inputs) normalize(a);
    for (auto& [_, a] : outputs) normalize(a);
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
          r.live.insert(a);
      }
      if (n.kind == Kind::Contraction) {
        const auto& left =
            result.requirements.at(program.replacement[n.inputs[0]].node).live;
        const auto& right =
            result.requirements.at(program.replacement[n.inputs[1]].node).live;
        auto shared = intersect(left, right, work);
        r.batch = intersect(shared, r.live, work);
        r.contract = difference(shared, r.live, work);
        r.left_free = difference(left, right, work);
        r.right_free = difference(right, left, work);
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
    for (const auto& [id, r] : result.requirements) {
      work.tick();
      const auto& n = *instructions.at(id);
      if (n.kind == Kind::Permutation || r.required_order.size() < 2) continue;
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
        output = extend(r.required_order, r.live, left, work);
        fields[n.fields[0]] = positions(left, output, work);
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
           bool construct) {
  Work work(limits);
  Result out;
  try {
    validate(p);
    Replay replay(p, m, m.descriptors, work);
    auto source = replay.run(p.source);
    if (p.mode == Mode::Validate) {
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
        a.result.reason = "required axes have incompatible relative order";
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

Result evaluate(const Program& p, const Metadata& m, const Limits& limits) {
  return run(p, m, limits, true);
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
