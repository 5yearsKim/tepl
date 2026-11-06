#include "src/inference/extensions.h"

#include <algorithm>
#include <charconv>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "src/inference/metadata_facts.h"

namespace tepl::inference {
MetadataTerm termField(std::size_t d, std::size_t f) {
  return {"field:" + std::to_string(d) + ":" + std::to_string(f), {}};
}
MetadataTerm termShape(std::size_t c) {
  return {"shape:" + std::to_string(c), {}};
}
MetadataTerm termNumber(std::int64_t n) {
  return {"#" + std::to_string(n), {}};
}
MetadataTerm termCall(std::string op, std::vector<MetadataTerm> a) {
  return {std::move(op), std::move(a)};
}
std::string termKey(const MetadataTerm& t) {
  std::string out = t.op + "(";
  for (const auto& a : t.args) out += termKey(a) + ",";
  return out + ")";
}
namespace {
using T = MetadataTerm;
bool number(const T& t, std::int64_t& n) {
  if (t.op.empty() || t.op[0] != '#' || !t.args.empty()) return false;
  auto r = std::from_chars(t.op.data() + 1, t.op.data() + t.op.size(), n);
  return r.ec == std::errc{} && r.ptr == t.op.data() + t.op.size();
}
std::optional<List> literal(const T& t) {
  if (t.op != "list") return {};
  List r;
  for (const auto& a : t.args) {
    std::int64_t n;
    if (!number(a, n)) return {};
    r.push_back(n);
  }
  return r;
}
T listTerm(const List& xs) {
  T t{"list", {}};
  for (auto n : xs) t.args.push_back(termNumber(n));
  return t;
}
List range(std::size_t n) {
  List xs(n);
  std::iota(xs.begin(), xs.end(), 0);
  return xs;
}
bool contains(const List& a, std::int64_t x) {
  return std::find(a.begin(), a.end(), x) != a.end();
}
// Both compilation and loading use exactly these conservative simplifications.
// In particular, no index or extent check is erased by algebraic cancellation.
T simplify(T t, const MetadataFacts& f) {
  for (auto& a : t.args) a = simplify(std::move(a), f);
  if (t.op == "broadcast_transpose_extents" && t.args.size() == 4) {
    auto shape = t.args[0], d = t.args[1], e = t.args[2], p = t.args[3];
    if (d == T{"list", {}}) return simplify(T{"gather", {e, p}}, f);
    auto n = T{"len", {p}};
    auto oldnew = T{"exclude", {T{"range", {n}}, d}};
    auto allpos = T{"concat", {d, oldnew}}, allshape = T{"concat", {shape, e}};
    auto oldshape = T{"scatter_n", {allpos, allshape, n}};
    auto mapped = T{"gather", {T{"inverse", {p}}, d}};
    auto newpos = T{"exclude", {T{"range", {n}}, mapped}};
    return simplify(T{"gather", {T{"gather", {oldshape, p}}, newpos}}, f);
  }
  if (t.op.starts_with("shape:") && t.args.empty()) {
    auto it = f.ranks.find(std::stoull(t.op.substr(6)));
    if (it != f.ranks.end() && it->second == 0) return {"list", {}};
  }
  if (t.op.starts_with("field:")) {
    auto p = t.op.find(':', 6);
    auto d = std::stoull(t.op.substr(6, p - 6)),
         i = std::stoull(t.op.substr(p + 1));
    if (auto it = f.fields.find({d, i}); it != f.fields.end())
      return listTerm(it->second);
  }
  if (t.op == "list" && !t.args.empty()) {
    const auto& first = t.args[0];
    if (first.op == "index" && first.args.size() == 2 &&
        first.args[0].op.starts_with("shape:")) {
      auto shape = first.args[0];
      auto it = f.ranks.find(std::stoull(shape.op.substr(6)));
      if (it != f.ranks.end() && it->second == t.args.size()) {
        bool all = true;
        for (std::size_t i = 0; i < t.args.size(); ++i)
          all &= t.args[i] == T{"index", {shape, termNumber(i)}};
        if (all) return shape;
      }
    }
  }
  if (t.op == "len" && t.args.size() == 1) {
    auto a = t.args[0];
    if (a.op == "list") return termNumber(a.args.size());
    if (a.op.starts_with("shape:")) {
      if (auto it = f.ranks.find(std::stoull(a.op.substr(6)));
          it != f.ranks.end())
        return termNumber(it->second);
    }
    if (a.op.starts_with("field:")) {
      auto p = a.op.find(':', 6);
      auto d = std::stoull(a.op.substr(6, p - 6)),
           i = std::stoull(a.op.substr(p + 1));
      if (auto it = f.lengths.find({d, i}); it != f.lengths.end())
        return termNumber(it->second);
    }
  }
  if (t.op == "+" && t.args.size() == 2) {
    std::int64_t a, b;
    if (number(t.args[0], a) && number(t.args[1], b) && a >= 0 && b >= 0 &&
        a <= std::numeric_limits<std::int64_t>::max() - b)
      return termNumber(a + b);
  }
  if (t.op == "range" && t.args.size() == 1) {
    std::int64_t n;
    if (number(t.args[0], n) && n >= 0 && n <= 4096) return listTerm(range(n));
  }
  if (t.op == "concat") {
    T out{"list", {}};
    for (const auto& a : t.args) {
      if (a.op != "list") return t;
      out.args.insert(out.args.end(), a.args.begin(), a.args.end());
    }
    return out;
  }
  // These identities are total on the typed, legal-source domain.
  if ((t.op == "pullback" || t.op == "gather") && t.args.size() == 2 &&
      t.args[t.op == "pullback" ? 0 : 1] == T{"list", {}})
    return {"list", {}};
  if (t.op == "xor" && t.args.size() == 2) {
    if (t.args[0] == t.args[1]) return {"list", {}};
    if (t.args[0] == T{"list", {}}) return t.args[1];
    if (t.args[1] == T{"list", {}}) return t.args[0];
  }
  if (t.op == "gather" && t.args.size() == 2 && literal(t.args[1]) &&
      t.args[0].op.starts_with("shape:")) {
    auto shape = t.args[0];
    auto it = f.ranks.find(std::stoull(shape.op.substr(6)));
    if (it != f.ranks.end()) {
      T out{"list", {}};
      const auto positions = literal(t.args[1]);
      for (auto i : *positions) {
        if (i < 0 || static_cast<std::size_t>(i) >= it->second) return t;
        out.args.push_back(T{"index", {shape, termNumber(i)}});
      }
      return out;
    }
  }
  if (t.args.size() == 2 && (t.op == "exclude" || t.op == "positions" ||
                             t.op == "pullback" || t.op == "remap")) {
    auto a = literal(t.args[0]), b = literal(t.args[1]);
    if (a && b) {
      List out;
      if (t.op == "positions") {
        for (auto x : *b) {
          auto it = std::find(a->begin(), a->end(), x);
          if (it == a->end()) return t;
          out.push_back(it - a->begin());
        }
      } else
        for (std::size_t i = 0; i < a->size(); ++i) {
          auto x = a->at(i);
          bool present = contains(*b, x);
          if (t.op == "pullback") {
            if (present) out.push_back(i);
          } else if (!present)
            out.push_back(t.op == "remap"
                              ? x - std::count_if(b->begin(), b->end(),
                                                  [&](auto y) { return y < x; })
                              : x);
        }
      return listTerm(out);
    }
  }
  if (t.op == "gather" && t.args.size() == 2) {
    auto a = literal(t.args[0]), b = literal(t.args[1]);
    if (a && b) {
      List out;
      for (auto i : *b) {
        if (i < 0 || static_cast<std::size_t>(i) >= a->size()) return t;
        out.push_back(a->at(i));
      }
      return listTerm(out);
    }
  }
  return t;
}
struct Builder {
  const Program& p;
  EmissionBlock b;
  using Code = std::vector<Instruction>;
  std::size_t add(Op op, std::vector<std::size_t> args = {},
                  std::size_t c = 0) {
    b.expressions.push_back({op, std::move(args), c});
    return b.expressions.size() - 1;
  }
  std::size_t field(std::size_t i) { return add(Op::SourceField, {}, i); }
  std::size_t ext(std::size_t i) { return add(Op::SourceExtents, {}, i); }
  std::size_t role(std::size_t i, std::size_t f) {
    return add(f == 0   ? Op::SourceField
               : f == 1 ? Op::SourceExtents
               : f == 2 ? Op::SourceField2
                        : Op::SourceField3,
               {}, i);
  }
  void copyDot(std::size_t si, std::size_t ri) {
    set(ri, {role(si, 0), role(si, 1), role(si, 2), role(si, 3)});
  }
  std::size_t shape(std::size_t i) { return add(Op::SourceShape, {}, i); }
  std::size_t empty() { return add(Op::Empty); }
  std::size_t singleton(std::size_t n) { return add(Op::Singleton, {}, n); }
  std::size_t shapeOf(std::size_t i) {
    const auto& n = p.source.at(i);
    if (n.kind == Kind::Input) return shape(i);
    if (n.kind == Kind::Multiplication || n.kind == Kind::Addition) {
      for (auto input : n.inputs)
        if (p.source[input].kind == Kind::Input) return shapeOf(input);
    }
    auto left = shapeOf(n.inputs[0]);
    if (n.kind == Kind::Multiplication || n.kind == Kind::Addition ||
        n.kind == Kind::Reverse || n.kind == Kind::Negation)
      return left;
    if (n.kind == Kind::Contraction) {
      auto right = shapeOf(n.inputs[1]);
      auto free = [&](std::size_t shape, std::size_t batch,
                      std::size_t contract) {
        return binary(
            Op::Gather, shape,
            binary(Op::Complement, unary(Op::Range, unary(Op::Length, shape)),
                   binary(Op::Concat, batch, contract)));
      };
      return binary(Op::Concat, binary(Op::Gather, left, role(i, 0)),
                    binary(Op::Concat, free(left, role(i, 0), role(i, 2)),
                           free(right, role(i, 1), role(i, 3))));
    }
    throw std::invalid_argument("unsupported shape expression in schema");
  }

  std::size_t unary(Op op, std::size_t a) { return add(op, {a}); }
  std::size_t binary(Op op, std::size_t a, std::size_t c) {
    return add(op, {a, c});
  }
  void check(std::size_t a, std::size_t c) {
    b.checks.push_back(binary(Op::Equal, a, c));
  }
  void set(std::size_t ri, std::vector<std::size_t> values) {
    const auto& n = p.replacement.at(ri);
    if (!n.descriptor || values.size() != n.fields.size())
      throw std::invalid_argument("extension descriptor mismatch");
    auto& out = b.attributes[*n.descriptor];
    out.resize(n.fields.size());
    for (std::size_t i = 0; i < values.size(); ++i)
      out.at(n.fields[i]) = values[i];
  }
  bool input(const Code& a, std::size_t i, const Code& c, std::size_t j) {
    return a[i].kind == Kind::Input && c[j].kind == Kind::Input &&
           a[i].capture == c[j].capture;
  }
  // Equal operation trees can copy every descriptor, including supplied fields.
  bool same(std::size_t si, std::size_t ri) {
    const auto& s = p.source[si];
    const auto& r = p.replacement[ri];
    if (s.kind != r.kind || s.capture != r.capture ||
        s.operation != r.operation || s.inputs.size() != r.inputs.size())
      return false;
    for (std::size_t i = 0; i < s.inputs.size(); ++i)
      if (!same(s.inputs[i], r.inputs[i])) return false;
    if (s.descriptor) {
      if (s.kind == Kind::Broadcast)
        set(ri, {field(si), ext(si)});
      else if (s.kind == Kind::Reverse || s.kind == Kind::Permutation ||
               s.kind == Kind::Summation)
        set(ri, {field(si)});
      else
        return false;
    }
    return true;
  }
  EmissionBlock run() {
    b.source = p;
    const auto& s = p.source;
    const auto& r = p.replacement;
    auto si = s.size() - 1, ri = r.size() - 1;
    const auto& a = s.back();
    const auto& z = r.back();
    if (same(si, ri)) return std::move(b);
    b.attributes.clear();
    b.expressions.clear();
    // A chain of numeric negations depends only on its parity.
    auto negbase = si;
    std::size_t parity = 0;
    while (s[negbase].kind == Kind::Negation) {
      parity ^= 1;
      negbase = s[negbase].inputs[0];
    }
    if (negbase != si && ((parity == 0 && input(s, negbase, r, ri)) ||
                          (parity == 1 && z.kind == Kind::Negation &&
                           input(s, negbase, r, z.inputs[0]))))
      return std::move(b);
    // Negation is linear over an empty or nonempty finite sum.
    if (a.kind == Kind::Summation && z.kind == Kind::Negation &&
        s[a.inputs[0]].kind == Kind::Negation &&
        r[z.inputs[0]].kind == Kind::Summation) {
      auto neg = a.inputs[0], red = z.inputs[0];
      if (s[neg].operation == z.operation &&
          input(s, s[neg].inputs[0], r, r[red].inputs[0])) {
        set(red, {field(si)});
        return std::move(b);
      }
    }
    // Reindex only contracted coordinates, leaving all free/batch axes fixed.
    if (a.kind == Kind::Contraction && z.kind == Kind::Contraction) {
      for (std::size_t side = 0; side < 2; ++side) {
        auto rev = a.inputs[side], other = a.inputs[1 - side],
             target = z.inputs[1 - side];
        if (s[rev].kind == Kind::Reverse && r[target].kind == Kind::Reverse &&
            input(s, s[rev].inputs[0], r, z.inputs[side]) &&
            input(s, other, r, r[target].inputs[0])) {
          b.checks.push_back(
              binary(Op::Subset, field(rev), role(si, 2 + side)));
          auto indices = binary(Op::Pullback, role(si, 2 + side), field(rev));
          set(target, {binary(Op::Gather, role(si, 3 - side), indices)});
          copyDot(si, ri);
          return std::move(b);
        }
      }
    }
    // A full aligned contraction is a pointwise product followed by a sum.
    if (a.kind == Kind::Contraction && z.kind == Kind::Summation &&
        r[z.inputs[0]].kind == Kind::Multiplication) {
      const auto& product = r[z.inputs[0]];
      if (input(s, a.inputs[0], r, product.inputs[0]) &&
          input(s, a.inputs[1], r, product.inputs[1])) {
        check(role(si, 0), empty());
        check(role(si, 1), empty());
        check(role(si, 2),
              unary(Op::Range, unary(Op::Length, shape(a.inputs[0]))));
        check(role(si, 3),
              unary(Op::Range, unary(Op::Length, shape(a.inputs[1]))));
        set(ri, {role(si, 2)});
        return std::move(b);
      }
    }
    // An outer product is multiplication of two complementary broadcasts.
    if (a.kind == Kind::Contraction && z.kind == Kind::Multiplication) {
      auto l = z.inputs[0], v = z.inputs[1];
      if (r[l].kind == Kind::Broadcast && r[v].kind == Kind::Broadcast &&
          input(s, a.inputs[0], r, r[l].inputs[0]) &&
          input(s, a.inputs[1], r, r[v].inputs[0])) {
        for (std::size_t f = 0; f < 4; ++f) check(role(si, f), empty());
        auto left = shape(a.inputs[0]), right = shape(a.inputs[1]);
        auto ld = unary(Op::Range, unary(Op::Length, left));
        auto all = unary(Op::Range, binary(Op::Plus, unary(Op::Length, left),
                                           unary(Op::Length, right)));
        set(l, {ld, right});
        set(v, {binary(Op::Complement, all, ld), left});
        return std::move(b);
      }
    }
    // Pull a scalar factor through any generalized dot; supplied output
    // extents are separately proved equal to this source output shape.
    if (a.kind == Kind::Contraction && z.kind == Kind::Multiplication) {
      for (std::size_t side = 0; side < 2; ++side) {
        auto product = a.inputs[side], other = a.inputs[1 - side];
        if (s[product].kind != Kind::Multiplication ||
            s[product].operation != z.operation)
          continue;
        for (std::size_t scalarSide = 0; scalarSide < 2; ++scalarSide) {
          auto bc = s[product].inputs[scalarSide],
               base = s[product].inputs[1 - scalarSide];
          if (s[bc].kind != Kind::Broadcast) continue;
          for (std::size_t outSide = 0; outSide < 2; ++outSide) {
            auto out = z.inputs[outSide], dot = z.inputs[1 - outSide];
            if (r[out].kind != Kind::Broadcast ||
                r[dot].kind != Kind::Contraction ||
                !input(s, s[bc].inputs[0], r, r[out].inputs[0]) ||
                !input(s, base, r, r[dot].inputs[side]) ||
                !input(s, other, r, r[dot].inputs[1 - side]))
              continue;
            check(field(bc), empty());
            copyDot(si, dot);
            set(out, {empty(), shapeOf(si)});
            return std::move(b);
          }
        }
      }
    }
    // A broadcast-created contraction coordinate reads no coordinate of X.
    // Sum its paired coordinates from Y before contracting original axes.
    if (a.kind == Kind::Contraction && z.kind == Kind::Contraction) {
      auto bc = a.inputs[0], y = a.inputs[1], sum = z.inputs[1];
      if (s[bc].kind == Kind::Broadcast && r[sum].kind == Kind::Summation &&
          input(s, s[bc].inputs[0], r, z.inputs[0]) &&
          input(s, y, r, r[sum].inputs[0])) {
        auto d = field(bc), xshape = shapeOf(s[bc].inputs[0]);
        auto outRank = binary(Op::Plus, unary(Op::Length, xshape),
                              unary(Op::Length, ext(bc)));
        auto all = unary(Op::Range, outRank),
             inserted = binary(Op::Complement, all, d);
        b.checks.push_back(binary(Op::Subset, inserted, role(si, 2)));
        b.checks.push_back(binary(Op::Subset, role(si, 0), d));
        auto removedPairs = binary(Op::Pullback, role(si, 2), inserted);
        auto removedY = binary(Op::Gather, role(si, 3), removedPairs);
        auto oldXContract = binary(Op::Complement, role(si, 2), inserted);
        auto keptPairs = binary(Op::Pullback, role(si, 2), d);
        auto lb = binary(Op::Positions, d, role(si, 0));
        auto lc = binary(Op::Positions, d, oldXContract);
        auto xfree =
            binary(Op::Complement, unary(Op::Range, unary(Op::Length, xshape)),
                   binary(Op::Concat, lb, lc));
        check(binary(Op::Gather, d, xfree),
              binary(Op::Complement, all,
                     binary(Op::Concat, role(si, 0), role(si, 2))));
        set(sum, {removedY});
        set(ri, {lb, binary(Op::Remap, role(si, 1), removedY), lc,
                 binary(Op::Remap, binary(Op::Gather, role(si, 3), keptPairs),
                        removedY)});
        return std::move(b);
      }
    }
    // Move a scalar between two nested products and broadcast embeddings.
    if (a.kind == Kind::Multiplication && z.kind == Kind::Multiplication &&
        a.operation == z.operation) {
      auto inner = a.inputs[0], yb = a.inputs[1], out = z.inputs[1];
      if (s[inner].kind == Kind::Multiplication &&
          s[inner].operation == a.operation && s[yb].kind == Kind::Broadcast &&
          r[out].kind == Kind::Broadcast) {
        auto bc = s[inner].inputs[1], product = r[out].inputs[0];
        if (s[bc].kind == Kind::Broadcast &&
            r[product].kind == Kind::Multiplication &&
            r[product].operation == a.operation) {
          auto small = r[product].inputs[1];
          if (r[small].kind == Kind::Broadcast &&
              input(s, s[inner].inputs[0], r, z.inputs[0]) &&
              input(s, s[yb].inputs[0], r, r[product].inputs[0]) &&
              input(s, s[bc].inputs[0], r, r[small].inputs[0])) {
            check(field(bc), empty());
            set(small, {empty(), shapeOf(s[yb].inputs[0])});
            set(out, {field(yb), ext(yb)});
            return std::move(b);
          }
        }
      }
    }
    // Eliminate a leading unit input coordinate and reinsert it as a fresh
    // leading coordinate. Existing broadcast insertions must be appended.
    if (a.kind == Kind::Broadcast && z.kind == Kind::Broadcast &&
        r[z.inputs[0]].kind == Kind::UnitRemoval) {
      auto remove = z.inputs[0];
      if (input(s, a.inputs[0], r, r[remove].inputs[0])) {
        auto zero = singleton(0), one = singleton(1),
             inputShape = shapeOf(a.inputs[0]);
        check(field(si), unary(Op::Range, unary(Op::Length, inputShape)));
        b.checks.push_back(binary(Op::Subset, zero,
            unary(Op::Range, unary(Op::Length, inputShape))));
        check(binary(Op::Gather, inputShape, zero), one);
        set(remove, {zero});
        set(ri, {binary(Op::Complement, field(si), zero),
                 binary(Op::Concat, one, ext(si))});
        return std::move(b);
      }
    }
    // Arbitrary-length reverse chains, including guarded cancellation.
    auto sbase = si;
    std::vector<std::size_t> reverses;
    while (s[sbase].kind == Kind::Reverse) {
      reverses.push_back(sbase);
      sbase = s[sbase].inputs[0];
    }
    if (!reverses.empty() && s[sbase].kind == Kind::Input &&
        ((z.kind == Kind::Reverse && input(s, sbase, r, z.inputs[0])) ||
         input(s, sbase, r, ri))) {
      auto dims = empty();
      for (auto i : reverses)
        dims = binary(Op::SymmetricDifference, dims, field(i));
      if (z.kind == Kind::Reverse)
        set(ri, {dims});
      else
        check(dims, empty());
      return std::move(b);
    }
    // Insertion-only broadcast composition; the outer insertion appends axes.
    if (a.kind == Kind::Broadcast && z.kind == Kind::Broadcast &&
        s[a.inputs[0]].kind == Kind::Broadcast) {
      auto bi = a.inputs[0];
      const auto& inner = s[bi];
      if (input(s, inner.inputs[0], r, z.inputs[0])) {
        auto rank = binary(Op::Plus, unary(Op::Length, shape(inner.inputs[0])),
                           unary(Op::Length, ext(bi)));
        check(field(si), unary(Op::Range, rank));
        set(ri, {binary(Op::Gather, field(si), field(bi)),
                 binary(Op::Concat, ext(bi), ext(si))});
        return std::move(b);
      }
    }
    // Pointwise addition/product commute with identical broadcast embeddings.
    if ((a.kind == Kind::Addition || a.kind == Kind::Multiplication) &&
        z.kind == Kind::Broadcast && r[z.inputs[0]].kind == a.kind &&
        r[z.inputs[0]].operation == a.operation) {
      auto l = a.inputs[0], v = a.inputs[1];
      const auto& inner = r[z.inputs[0]];
      if (s[l].kind == Kind::Broadcast && s[v].kind == Kind::Broadcast &&
          input(s, s[l].inputs[0], r, inner.inputs[0]) &&
          input(s, s[v].inputs[0], r, inner.inputs[1])) {
        check(field(l), field(v));
        check(ext(l), ext(v));
        set(ri, {field(l), ext(l)});
        return std::move(b);
      }
    }
    // Push reversal through a broadcast; inserted dimensions are constant.
    if (a.kind == Kind::Reverse && z.kind == Kind::Broadcast &&
        s[a.inputs[0]].kind == Kind::Broadcast) {
      auto bi = a.inputs[0], base = s[bi].inputs[0], target = z.inputs[0];
      auto pulled = binary(Op::Pullback, field(bi), field(si));
      if (input(s, base, r, target))
        check(pulled, empty());
      else if (r[target].kind == Kind::Reverse &&
               input(s, base, r, r[target].inputs[0]))
        set(target, {pulled});
      else
        throw std::invalid_argument("unrecognized reverse/broadcast pair");
      set(ri, {field(bi), ext(bi)});
      return std::move(b);
    }
    // Any insertion-only broadcast absorbs transpose by composing the
    // embedding and permuting only the complement's inserted extents.
    if (a.kind == Kind::Permutation && z.kind == Kind::Broadcast &&
        s[a.inputs[0]].kind == Kind::Broadcast) {
      auto bc = a.inputs[0];
      if (input(s, s[bc].inputs[0], r, z.inputs[0])) {
        auto d = binary(Op::Gather, unary(Op::Inverse, field(si)), field(bc));
        auto extents =
            add(Op::BroadcastTransposeExtents,
                {shapeOf(s[bc].inputs[0]), field(bc), ext(bc), field(si)});
        set(ri, {d, extents});
        return std::move(b);
      }
    }
    // A broadcast that inserts no axes is a permutation (inverse embedding).
    if (a.kind == Kind::Broadcast && z.kind == Kind::Permutation &&
        input(s, a.inputs[0], r, z.inputs[0])) {
      check(ext(si), empty());
      set(ri, {unary(Op::Inverse, field(si))});
      return std::move(b);
    }
    // Reversal distributes over addition with a scalar broadcast summand.
    if (a.kind == Kind::Reverse && z.kind == Kind::Addition &&
        s[a.inputs[0]].kind == Kind::Addition) {
      const auto& sum = s[a.inputs[0]];
      if (sum.operation != z.operation)
        throw std::invalid_argument("pointwise declaration mismatch");
      auto x = sum.inputs[0], bi = sum.inputs[1], rev = z.inputs[0],
           bc = z.inputs[1];
      if (s[bi].kind == Kind::Broadcast && r[rev].kind == Kind::Reverse &&
          r[bc].kind == Kind::Broadcast && input(s, x, r, r[rev].inputs[0]) &&
          input(s, s[bi].inputs[0], r, r[bc].inputs[0])) {
        check(field(bi), empty());
        set(rev, {field(si)});
        set(bc, {field(bi), ext(bi)});
        return std::move(b);
      }
    }
    // Reduction of original axes commutes with insertion-only broadcast.
    if (a.kind == Kind::Summation && z.kind == Kind::Broadcast &&
        s[a.inputs[0]].kind == Kind::Broadcast &&
        r[z.inputs[0]].kind == Kind::Summation) {
      auto bi = a.inputs[0], red = z.inputs[0];
      if (input(s, s[bi].inputs[0], r, r[red].inputs[0])) {
        b.checks.push_back(binary(Op::Subset, field(si), field(bi)));
        set(red, {binary(Op::Pullback, field(bi), field(si))});
        set(ri, {binary(Op::Remap, field(bi), field(si)), ext(bi)});
        return std::move(b);
      }
    }
    throw std::invalid_argument("outside supported view/reindexing schemas");
  }
};
}  // namespace
bool hasExtensions(const Program& p) {
  auto test = [](const auto& code) {
    return std::any_of(code.begin(), code.end(), [](const auto& n) {
      return n.kind == Kind::Reverse || n.kind == Kind::Broadcast ||
             n.kind == Kind::Multiplication || n.kind == Kind::Negation ||
             n.kind == Kind::UnitRemoval;
    });
  };
  return test(p.source) || test(p.replacement);
}
EmissionBlock emitExtension(const Program& p) { return Builder{p, {}}.run(); }
MetadataTerm emissionTerm(const EmissionBlock& b, std::size_t id) {
  const auto& e = b.expressions.at(id);
  const auto& s = b.source.source;
  if (e.op == Op::SourceShape) return termShape(*s.at(e.capture).capture);
  if (e.op == Op::Singleton)
    return listTerm(List{static_cast<std::int64_t>(e.capture)});
  if (e.op == Op::BroadcastTransposeExtents) {
    std::vector<T> args;
    for (auto id : e.args) args.push_back(emissionTerm(b, id));
    return {"broadcast_transpose_extents", std::move(args)};
  }
  if (e.op == Op::SourceField || e.op == Op::SourceExtents ||
      e.op == Op::SourceField2 || e.op == Op::SourceField3) {
    const auto& n = s.at(e.capture);
    return termField(*n.descriptor,
                     n.fields.at(e.op == Op::SourceExtents  ? 1
                                 : e.op == Op::SourceField2 ? 2
                                 : e.op == Op::SourceField3 ? 3
                                                            : 0));
  }
  std::string name;
  switch (e.op) {
    case Op::Empty:
      name = "list";
      break;
    case Op::Length:
      name = "len";
      break;
    case Op::Range:
      name = "range";
      break;
    case Op::Plus:
      name = "+";
      break;
    case Op::Concat:
      name = "concat";
      break;
    case Op::Gather:
      name = "gather";
      break;
    case Op::Positions:
      name = "positions";
      break;
    case Op::Inverse:
      name = "inverse";
      break;
    case Op::SymmetricDifference:
      name = "xor";
      break;
    case Op::Complement:
      name = "exclude";
      break;
    case Op::Pullback:
      name = "pullback";
      break;
    case Op::Remap:
      name = "remap";
      break;
    case Op::Equal:
      name = "==";
      break;
    case Op::Subset:
      name = "subset";
      break;
    default:
      throw std::invalid_argument("unknown schema metadata expression");
  }
  std::vector<T> args;
  for (auto i : e.args) args.push_back(emissionTerm(b, i));
  return {name, std::move(args)};
}
MetadataTerm normalizeMetadata(MetadataTerm t, const MetadataFacts& f) {
  return simplify(std::move(t), f);
}
bool metadataEqual(const T& a, const T& b, const MetadataFacts& f) {
  // Finite congruence closure of the exact positive equalities. Each term is
  // simplified only using literal source facts or total identities above.
  std::vector<T> terms;
  std::map<std::string, std::size_t> ids;
  std::vector<std::size_t> parent;
  std::function<std::size_t(T)> add = [&](T t) {
    t = simplify(std::move(t), f);
    for (const auto& x : t.args) add(x);
    auto k = termKey(t);
    if (auto it = ids.find(k); it != ids.end()) return it->second;
    auto id = terms.size();
    ids[k] = id;
    terms.push_back(std::move(t));
    parent.push_back(id);
    return id;
  };
  std::function<std::size_t(std::size_t)> root = [&](auto i) {
    return parent[i] == i ? i : parent[i] = root(parent[i]);
  };
  auto join = [&](auto x, auto y) {
    x = root(x);
    y = root(y);
    if (x == y) return false;
    parent[x] = y;
    return true;
  };
  auto ai = add(a), bi = add(b);
  const auto empty_id = add(T{"list", {}});
  for (const auto& [x, y] : f.equalities) join(add(x), add(y));
  // Resource-bounded, fail closed rather than guessing a guard implication.
  if (terms.size() > 4096) return false;
  for (std::size_t round = 0; round < terms.size(); ++round) {
    bool changed = false;
    std::map<std::string, std::size_t> signatures;
    for (std::size_t i = 0; i < terms.size(); ++i) {
      auto sig = terms[i].op + ":";
      if (terms[i].op == "xor" && terms[i].args.size() == 2 &&
          root(ids.at(termKey(simplify(terms[i].args[0], f)))) ==
              root(ids.at(termKey(simplify(terms[i].args[1], f)))))
        changed |= join(i, empty_id);
      for (const auto& t : terms[i].args)
        sig += std::to_string(root(ids.at(termKey(simplify(t, f))))) + ",";
      auto [it, ok] = signatures.emplace(sig, i);
      if (!ok) changed |= join(i, it->second);
    }
    if (!changed) break;
  }
  return root(ai) == root(bi);
}
namespace {
struct ExtensionStop {
  Status status;
  const char* reason;
};
struct ExtensionVM {
  const EmissionBlock& b;
  const Metadata& m;
  const Limits& lim;
  std::size_t remaining;
  std::map<std::size_t, List> values;
  void charge(std::size_t n = 1) {
    if (n > remaining)
      throw ExtensionStop{Status::ResourceLimit, "schema metadata work limit"};
    remaining -= n;
  }
  void bound(std::size_t n) {
    if (n > lim.axes)
      throw ExtensionStop{Status::ResourceLimit, "schema metadata list limit"};
  }
  void require(bool yes) {
    if (!yes)
      throw ExtensionStop{Status::InvalidSource,
                          "invalid source view metadata"};
  }
  const List& field(std::size_t i, std::size_t role) {
    const auto& n = b.source.source.at(i);
    auto it = m.descriptors.find(*n.descriptor);
    if (it == m.descriptors.end() || n.fields.at(role) >= it->second.size())
      throw ExtensionStop{Status::MissingMetadata,
                          "missing source view descriptor"};
    const auto& xs = it->second[n.fields[role]];
    bound(xs.size());
    charge(xs.size());
    return xs;
  }
  bool has(const List& a, std::int64_t x) {
    charge(a.size());
    return contains(a, x);
  }
  const List& eval(std::size_t id) {
    if (auto it = values.find(id); it != values.end()) return it->second;
    charge();
    const auto& e = b.expressions.at(id);
    List out;
    if (e.op == Op::SourceField || e.op == Op::SourceExtents ||
        e.op == Op::SourceField2 || e.op == Op::SourceField3)
      out = field(e.capture, e.op == Op::SourceExtents  ? 1
                             : e.op == Op::SourceField2 ? 2
                             : e.op == Op::SourceField3 ? 3
                                                        : 0);
    else if (e.op == Op::SourceShape) {
      auto cap = *b.source.source.at(e.capture).capture;
      auto it = m.shapes.find(cap);
      if (it == m.shapes.end())
        throw ExtensionStop{Status::MissingMetadata,
                            "missing source view shape"};
      bound(it->second.size());
      charge(it->second.size());
      for (auto x : it->second) {
        if (x > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()))
          throw ExtensionStop{Status::ResourceLimit,
                              "schema extent representation limit"};
        out.push_back(x);
      }
    } else if (e.op == Op::Singleton) {
      if (e.capture >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw ExtensionStop{Status::ResourceLimit, "schema constant range"};
      out = {static_cast<std::int64_t>(e.capture)};
    } else if (e.op == Op::BroadcastTransposeExtents) {
      const auto& shape = eval(e.args.at(0));
      const auto& dims = eval(e.args.at(1));
      const auto& ext = eval(e.args.at(2));
      const auto& perm = eval(e.args.at(3));
      auto n = shape.size() + ext.size();
      bound(n);
      charge(n * 3 + shape.size());
      require(dims.size() == shape.size() && perm.size() == n);
      List old(n, -1), inverse(n, -1);
      std::vector<bool> embedded(n);
      for (std::size_t i = 0; i < dims.size(); ++i) {
        require(dims[i] >= 0 && static_cast<std::size_t>(dims[i]) < n &&
                !embedded[dims[i]]);
        old[dims[i]] = shape[i];
        embedded[dims[i]] = true;
      }
      std::size_t j = 0;
      for (std::size_t i = 0; i < n; ++i)
        if (!embedded[i]) old[i] = ext[j++];
      for (std::size_t i = 0; i < n; ++i) {
        require(perm[i] >= 0 && static_cast<std::size_t>(perm[i]) < n &&
                inverse[perm[i]] < 0);
        inverse[perm[i]] = i;
      }
      std::vector<bool> retained(n);
      for (auto d : dims) retained[inverse[d]] = true;
      for (std::size_t i = 0; i < n; ++i)
        if (!retained[i]) out.push_back(old[perm[i]]);
    } else if (e.op != Op::Empty) {
      const auto& a = eval(e.args.at(0));
      const List empty;
      const auto& c = e.args.size() > 1 ? eval(e.args[1]) : empty;
      switch (e.op) {
        case Op::Length:
          out = {static_cast<std::int64_t>(a.size())};
          break;
        case Op::Range:
          require(a.size() == 1 && a[0] >= 0);
          bound(a[0]);
          charge(a[0]);
          out = range(a[0]);
          break;
        case Op::Plus:
          require(a.size() == 1 && c.size() == 1 && a[0] >= 0 && c[0] >= 0);
          if (a[0] > std::numeric_limits<std::int64_t>::max() - c[0])
            throw ExtensionStop{Status::ResourceLimit,
                                "schema length overflow"};
          out = {a[0] + c[0]};
          break;
        case Op::Concat:
          bound(a.size() + c.size());
          charge(a.size() + c.size());
          out = a;
          out.insert(out.end(), c.begin(), c.end());
          break;
        case Op::Gather:
          bound(c.size());
          charge(c.size());
          for (auto i : c) {
            require(i >= 0 && static_cast<std::size_t>(i) < a.size());
            out.push_back(a[i]);
          }
          break;
        case Op::Positions:
          for (auto x : c) {
            charge(a.size());
            auto it = std::find(a.begin(), a.end(), x);
            require(it != a.end());
            out.push_back(it - a.begin());
          }
          break;
        case Op::Inverse:
          bound(a.size());
          charge(a.size());
          out.assign(a.size(), -1);
          for (std::size_t i = 0; i < a.size(); ++i) {
            require(a[i] >= 0 && static_cast<std::size_t>(a[i]) < a.size() &&
                    out[a[i]] < 0);
            out[a[i]] = i;
          }
          break;
        case Op::SymmetricDifference:
          for (auto x : a)
            if (!has(c, x)) out.push_back(x);
          for (auto x : c)
            if (!has(a, x)) out.push_back(x);
          break;
        case Op::Complement:
          for (auto x : a)
            if (!has(c, x)) out.push_back(x);
          break;
        case Op::Pullback:
          for (std::size_t i = 0; i < a.size(); ++i)
            if (has(c, a[i])) out.push_back(i);
          break;
        case Op::Remap:
          for (auto x : a)
            if (!has(c, x)) {
              charge(c.size());
              out.push_back(x - std::count_if(c.begin(), c.end(),
                                              [&](auto y) { return y < x; }));
            }
          break;
        case Op::Equal:
          charge(a.size() + c.size());
          out = {a == c ? 1 : 0};
          break;
        case Op::Subset: {
          bool yes = true;
          for (auto x : a) yes = has(c, x) && yes;
          out = {yes ? 1 : 0};
          break;
        }
        default:
          throw std::invalid_argument("invalid schema DAG opcode");
      }
    }
    bound(out.size());
    return values.emplace(id, std::move(out)).first->second;
  }
  void sourceLegality() {
    std::vector<List> shapes;
    for (std::size_t i = 0; i < b.source.source.size(); ++i) {
      charge();
      const auto& n = b.source.source[i];
      List shape;
      if (n.kind == Kind::Input) {
        auto it = m.shapes.find(*n.capture);
        if (it == m.shapes.end())
          throw ExtensionStop{Status::MissingMetadata,
                              "missing source view shape"};
        bound(it->second.size());
        charge(it->second.size());
        for (auto x : it->second) {
          if (x > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max()))
            throw ExtensionStop{Status::ResourceLimit,
                                "schema extent representation limit"};
          shape.push_back(x);
        }
      } else {
        const auto& input = shapes.at(n.inputs[0]);
        if (n.kind == Kind::Addition || n.kind == Kind::Multiplication) {
          charge(input.size());
          require(input == shapes.at(n.inputs[1]));
          shape = input;
        } else if (n.kind == Kind::Negation)
          shape = input;
        else if (n.kind == Kind::Contraction) {
          const auto& right = shapes.at(n.inputs[1]);
          const auto& lb = field(i, 0);
          const auto& rb = field(i, 1);
          const auto& lc = field(i, 2);
          const auto& rc = field(i, 3);
          require(lb.size() == rb.size() && lc.size() == rc.size());
          auto validatePair = [&](const List& batch, const List& contract,
                                  std::size_t rank) {
            std::set<std::int64_t> seen;
            for (const auto* xs : {&batch, &contract})
              for (auto x : *xs) {
                charge();
                require(x >= 0 && static_cast<std::size_t>(x) < rank &&
                        seen.insert(x).second);
              }
          };
          validatePair(lb, lc, input.size());
          validatePair(rb, rc, right.size());
          for (std::size_t j = 0; j < lb.size(); ++j) {
            charge();
            require(input[lb[j]] == right[rb[j]]);
            shape.push_back(input[lb[j]]);
          }
          for (std::size_t j = 0; j < lc.size(); ++j) {
            charge();
            require(input[lc[j]] == right[rc[j]]);
          }
          for (std::size_t j = 0; j < input.size(); ++j)
            if (!has(lb, j) && !has(lc, j)) shape.push_back(input[j]);
          for (std::size_t j = 0; j < right.size(); ++j)
            if (!has(rb, j) && !has(rc, j)) shape.push_back(right[j]);
        } else {
          const auto& dims = field(i, 0);
          std::set<std::int64_t> seen;
          auto rank = input.size();
          if (n.kind == Kind::Broadcast) rank += field(i, 1).size();
          bound(rank);
          for (auto x : dims) {
            charge();
            require(x >= 0 && static_cast<std::size_t>(x) < rank &&
                    seen.insert(x).second);
          }
          if (n.kind == Kind::Broadcast) {
            require(dims.size() == input.size());
            const auto& extents = field(i, 1);
            shape.assign(rank, -1);
            for (std::size_t j = 0; j < dims.size(); ++j)
              shape[dims[j]] = input[j];
            std::size_t j = 0;
            for (auto& x : shape)
              if (x < 0) {
                require(extents[j] >= 0);
                x = extents[j++];
              }
          } else if (n.kind == Kind::Permutation) {
            require(dims.size() == input.size());
            for (auto x : dims) shape.push_back(input[x]);
          } else if (n.kind == Kind::Reverse)
            shape = input;
          else if (n.kind == Kind::Summation || n.kind == Kind::UnitRemoval) {
            if (n.kind == Kind::UnitRemoval)
              for (auto d : dims) {
                charge();
                require(input[d] == 1);
              }
            for (std::size_t j = 0; j < input.size(); ++j)
              if (!has(dims, j)) shape.push_back(input[j]);
          } else
            throw std::invalid_argument("unsupported source schema operation");
        }
      }
      bound(shape.size());
      charge(shape.size());
      shapes.push_back(std::move(shape));
    }
  }
};
}  // namespace
Result executeExtension(const Emission& e, const Metadata& m,
                        const Limits& limits, bool build,
                        ExecutionTiming* timing) {
  if (timing) *timing = {};
  Result r;
  ExtensionVM vm{e.blocks.at(0), m, limits, limits.work, {}};
  try {
    vm.sourceLegality();
    {
      detail::PhaseTimer phase(timing ? &timing->feasibility_us : nullptr);
      for (auto id : vm.b.checks)
        if (vm.eval(id) != List{1}) {
          r.status = Status::Inapplicable;
          r.reason = "schema feasibility predicate is false";
          r.work = limits.work - vm.remaining;
          return r;
        }
    }
    if (build) {
      detail::PhaseTimer phase(timing ? &timing->construction_us : nullptr);
      for (const auto& [id, fields] : vm.b.attributes)
        for (auto x : fields) r.attributes[id].push_back(vm.eval(x));
    }
    r.status = build ? Status::Solved : Status::Feasible;
  } catch (const ExtensionStop& stop) {
    r.status = stop.status;
    r.reason = stop.reason;
  }
  r.work = limits.work - vm.remaining;
  return r;
}
}  // namespace tepl::inference
