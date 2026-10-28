#include "src/core/semantics/symbolic/metadata_sequences.h"

#include <algorithm>
#include <functional>
#include <limits>
namespace tepl::core::semantics {
std::optional<std::vector<SequenceSelection>> projectSelection(
    Context& c, const S& base, const S& positions) {
  if (!callIs(base, "concat")) return {};
  if (callIs(positions, "ordered_complement") &&
      callIs(positions->args[0], "range") &&
      positions->args[1]->kind == Symbol::Kind::List &&
      c.equal(c.canonical(positions->args[0]->args[0]),
              c.canonical(call("len", {base})))) {
    // Locate the finitely many excluded axis positions, not every position
    // in the universe. Once they are consumed, suffix blocks stay symbolic.
    std::vector<MetadataInteger> excluded;
    for (const auto& value : positions->args[1]->args) {
      auto index = metadataInteger(value);
      if (!index || *index < 0) return {};
      excluded.push_back(*index);
    }
    std::sort(excluded.begin(), excluded.end());
    excluded.erase(std::unique(excluded.begin(), excluded.end()),
                   excluded.end());
    std::vector<SequenceSelection> result;
    MetadataInteger offset = 0;
    std::size_t next = 0;
    for (const auto& part : base->args) {
      c.tick();
      auto length = c.canonical(call("len", {part}));
      if (next == excluded.size()) {
        result.push_back({part, call("range", {length})});
        continue;
      }
      auto size = metadataInteger(length);
      MetadataInteger end;
      if (!size || *size < 0 || __builtin_add_overflow(offset, *size, &end))
        return {};
      std::vector<S> local;
      while (next < excluded.size() && excluded[next] < end) {
        auto index = excluded[next++] - offset;
        if (index < 0 || index > std::numeric_limits<std::size_t>::max())
          return {};
        local.push_back(integer(static_cast<std::size_t>(index)));
      }
      result.push_back(
          {part, call("ordered_complement",
                      {call("range", {length}),
                       make(Symbol::Kind::List, {}, 0, 0, std::move(local))})});
      offset = end;
    }
    if (next != excluded.size()) return {};
    return result;
  }
  if (positions->kind != Symbol::Kind::List) return {};
  std::vector<SequenceSelection> result;
  std::vector<S> local_positions;
  std::size_t previous = base->args.size();
  auto flush = [&] {
    if (!local_positions.empty()) {
      result.push_back(
          {base->args[previous],
           make(Symbol::Kind::List, {}, 0, 0, std::move(local_positions))});
      local_positions.clear();
    }
  };
  for (const auto& position : positions->args) {
    auto local = metadataInteger(position);
    if (!local || *local < 0) return {};
    std::size_t block = 0;
    for (; block < base->args.size(); ++block) {
      c.tick();
      auto size =
          metadataInteger(c.canonical(call("len", {base->args[block]})));
      if (!size || *size < 0) return {};
      if (*local < *size) break;
      *local -= *size;
    }
    if (block == base->args.size() ||
        *local > std::numeric_limits<std::size_t>::max())
      return {};
    if (block != previous) {
      flush();
      previous = block;
    }
    local_positions.push_back(integer(static_cast<std::size_t>(*local)));
  }
  flush();
  return result;
}
S substituteMetadata(const S& e, const S& variable, const S& value) {
  if (equal(e, variable)) return value;
  if (e->args.empty()) return e;
  std::vector<S> args;
  for (const auto& a : e->args)
    args.push_back(substituteMetadata(a, variable, value));
  // Keep the evaluation tree intact; definedness is checked before folding.
  return make(e->kind, e->name, e->owner, e->index, std::move(args));
}
S liftMetadataMap(Context& c, const S& seq, const S& var, const S& body) {
  auto rawCall = [](std::string name, std::vector<S> args) {
    return make(Symbol::Kind::Call, std::move(name), 0, 0, std::move(args));
  };
  std::function<S(const S&)> lift = [&](const S& x) -> S {
    c.tick();
    if (equal(x, var)) return seq;
    if (x->kind == Symbol::Kind::Integer)
      return rawCall("offset_extents", {rawCall("extent_sub", {seq, seq}), x});
    if (callIs(x, "index") && equal(x->args[1], var))
      return rawCall("gather", {x->args[0], seq});
    if ((callIs(x, "+") || callIs(x, "-") || callIs(x, "*")) &&
        x->args.size() == 2) {
      auto a = lift(x->args[0]), b = lift(x->args[1]);
      if (a && b)
        return rawCall(callIs(x, "+")   ? "extent_add"
                       : callIs(x, "-") ? "extent_sub"
                                        : "extent_mul",
                       {a, b});
    }
    return {};
  };
  return lift(body);
}
S simplifyMetadataSequence(Context& c, const S& e) {
  if (e->kind == Symbol::Kind::List && !e->args.empty()) {
    auto first = e->args[0];
    if (callIs(first, "index")) {
      std::vector<S> indices;
      bool same = true;
      for (const auto& x : e->args) {
        if (!callIs(x, "index") || !equal(x->args[0], first->args[0])) {
          same = false;
          break;
        }
        indices.push_back(x->args[1]);
      }
      if (same)
        return call("gather", {first->args[0], make(Symbol::Kind::List, {}, 0,
                                                    0, std::move(indices))});
    }
    if (callIs(first, "+") || callIs(first, "-") || callIs(first, "*")) {
      std::vector<S> a, b;
      bool same = true;
      for (const auto& x : e->args) {
        if (!callIs(x, first->name) || x->args.size() != 2) {
          same = false;
          break;
        }
        a.push_back(x->args[0]);
        b.push_back(x->args[1]);
      }
      if (same)
        return call(first->name == "+"   ? "extent_add"
                    : first->name == "-" ? "extent_sub"
                                         : "extent_mul",
                    {make(Symbol::Kind::List, {}, 0, 0, a),
                     make(Symbol::Kind::List, {}, 0, 0, b)});
    }
  }
  if (callIs(e, "sequence_map")) {
    auto seq = e->args[0], var = e->args[1], body = e->args[2];
    if (seq->kind == Symbol::Kind::List && seq->args.size() <= 64) {
      std::vector<S> out;
      for (const auto& x : seq->args) {
        c.tick();
        out.push_back(c.canonical(substituteMetadata(body, var, x)));
      }
      return make(Symbol::Kind::List, {}, 0, 0, std::move(out));
    }
    if (auto lifted = liftMetadataMap(c, seq, var, body))
      return c.canonical(lifted);
  }
  if ((callIs(e, "all") || callIs(e, "any")) &&
      e->args[0]->kind == Symbol::Kind::List) {
    bool all = callIs(e, "all");
    for (const auto& x : e->args[0]->args) {
      if (x->kind != Symbol::Kind::Boolean) return e;
      if (truth(x) != all) return boolean(!all);
    }
    return boolean(all);
  }
  if (callIs(e, "all") && callIs(e->args[0], "sequence_map")) {
    auto m = e->args[0], body = m->args[2], var = m->args[1];
    if ((callIs(body, ">") || callIs(body, ">=")) &&
        equal(body->args[0], var) && equal(body->args[1], integer(0)))
      return call(
          callIs(body, ">") ? "is_positive_extents" : "is_valid_extents",
          {m->args[0]});
    // An explicit pointwise bound in a where condition is also a bound on
    // the whole supplied metadata sequence. This preserves empty-list truth.
    if ((callIs(body, "<=") || callIs(body, ">=")) &&
        equal(body->args[0], var) &&
        body->args[1]->kind == Symbol::Kind::Integer) {
      auto repeated =
          call("offset_extents",
               {call("extent_sub", {m->args[0], m->args[0]}), body->args[1]});
      return call("extent_le", callIs(body, "<=")
                                   ? std::vector<S>{m->args[0], repeated}
                                   : std::vector<S>{repeated, m->args[0]});
    }
    if (callIs(body, "==") && callIs(body->args[0], "index") &&
        equal(body->args[0]->args[1], var) &&
        body->args[1]->kind == Symbol::Kind::Integer) {
      auto selected = call("gather", {body->args[0]->args[0], m->args[0]});
      auto repeated =
          call("offset_extents",
               {call("extent_sub", {selected, selected}), body->args[1]});
      return call("==", {selected, repeated});
    }
  }
  return e;
}
}  // namespace tepl::core::semantics
