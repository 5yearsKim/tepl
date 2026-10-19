#include "src/core/semantics/symbolic/metadata_sequences.h"

#include <functional>
namespace tepl::core::semantics {
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
