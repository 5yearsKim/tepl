#include "src/inference/certificate.h"

#include <stdexcept>
namespace tepl::inference {
bool sumMovement(const Program& p) {
  if (!p.components.empty()) return false;
  const auto& s = p.source;
  const auto& r = p.replacement;
  if (s.size() != 4 || r.size() != 4 || s.back().kind != Kind::Contraction ||
      r.back().kind != Kind::Summation)
    return false;
  const auto& a = s.at(s.back().inputs[0]);
  const auto& b = s.at(s.back().inputs[1]);
  const auto& d = r.at(r.back().inputs[0]);
  return a.kind == Kind::Summation && b.kind == Kind::Input &&
         s.at(a.inputs[0]).kind == Kind::Input && d.kind == Kind::Contraction &&
         r.at(d.inputs[0]).kind == Kind::Input &&
         r.at(d.inputs[1]).kind == Kind::Input &&
         s.at(a.inputs[0]).capture == r.at(d.inputs[0]).capture &&
         b.capture == r.at(d.inputs[1]).capture;
}
namespace {
bool sameTree(const std::vector<Instruction>& a, std::size_t ai,
              const std::vector<Instruction>& b, std::size_t bi) {
  const auto& x = a[ai];
  const auto& y = b[bi];
  if (x.kind != y.kind || x.inputs.size() != y.inputs.size() ||
      x.capture != y.capture)
    return false;
  for (std::size_t i = 0; i < x.inputs.size(); ++i)
    if (!sameTree(a, x.inputs[i], b, y.inputs[i])) return false;
  return true;
}
bool foldTranspose(const Program& p, const LayoutFacts& f) {
  if (!p.components.empty() || p.source.size() != 4 ||
      p.replacement.size() != 3)
    return false;
  const auto& s = p.source;
  const auto& r = p.replacement;
  const auto& dot = s.back();
  const auto& target = r.back();
  if (dot.kind != Kind::Contraction || target.kind != Kind::Contraction)
    return false;
  for (std::size_t side = 0; side < 2; ++side) {
    const auto& perm = s[dot.inputs[side]];
    const auto& other = s[dot.inputs[1 - side]];
    if (perm.kind != Kind::Permutation || other.kind != Kind::Input ||
        s[perm.inputs[0]].kind != Kind::Input ||
        r[target.inputs[side]].kind != Kind::Input ||
        r[target.inputs[1 - side]].kind != Kind::Input)
      continue;
    const auto& base = s[perm.inputs[0]];
    if (base.capture != r[target.inputs[side]].capture ||
        other.capture != r[target.inputs[1 - side]].capture)
      continue;
    auto length = f.lengths.find({*dot.descriptor, dot.fields[side + 2]});
    if (length != f.lengths.end() && length->second == 1 &&
        f.batch_plus_two.contains(
            {*base.capture, *dot.descriptor, dot.fields[side]}))
      return true;
  }
  return false;
}
bool reassociation(const Program& p, const LayoutFacts& f) {
  if (!p.components.empty() || p.source.size() != 5 ||
      p.replacement.size() != 5)
    return false;
  const auto& s = p.source;
  const auto& r = p.replacement;
  const auto& outer = s.back();
  const auto& target = r.back();
  if (outer.kind != Kind::Contraction || target.kind != Kind::Contraction)
    return false;
  const auto& inner = s[outer.inputs[0]];
  const auto& c = s[outer.inputs[1]];
  const auto& a = r[target.inputs[0]];
  const auto& nested = r[target.inputs[1]];
  return inner.kind == Kind::Contraction && nested.kind == Kind::Contraction &&
         a.kind == Kind::Input && c.kind == Kind::Input &&
         s[inner.inputs[0]].kind == Kind::Input &&
         s[inner.inputs[1]].kind == Kind::Input &&
         r[nested.inputs[0]].kind == Kind::Input &&
         r[nested.inputs[1]].kind == Kind::Input &&
         s[inner.inputs[0]].capture == a.capture &&
         s[inner.inputs[1]].capture == r[nested.inputs[0]].capture &&
         c.capture == r[nested.inputs[1]].capture &&
         f.batch_prefixes.contains({*outer.descriptor, outer.fields[0],
                                    *inner.descriptor, inner.fields[0]});
}
}  // namespace
bool certify(const Program& p, const LayoutFacts& f, const Limits& limits) {
  try {
    validate(p);
    if (p.mode != Mode::Synthesize || !limits.work || f.inconsistent)
      return false;
    if (hasExtensions(p)) return certifyExtension(p, f);
    if (sumMovement(p)) return true;
    if (foldTranspose(p, f)) return true;
    if (reassociation(p, f)) return true;
    bool identical = true;
    if (p.components.empty())
      identical = sameTree(p.source, p.source.size() - 1, p.replacement,
                           p.replacement.size() - 1);
    else
      for (const auto& c : p.components)
        identical &= sameTree(c.source, c.source.size() - 1, c.replacement,
                              c.replacement.size() - 1);
    if (identical) return true;
    if (p.components.empty()) {
      bool unary = true, source_sum = false, source_perm = false,
           target_sum = false, target_perm = false;
      for (const auto& n : p.source) {
        unary &= n.kind == Kind::Input || n.kind == Kind::Summation ||
                 n.kind == Kind::Permutation;
        source_sum |= n.kind == Kind::Summation;
        source_perm |= n.kind == Kind::Permutation;
      }
      for (const auto& n : p.replacement) {
        unary &= n.kind == Kind::Input || n.kind == Kind::Summation ||
                 n.kind == Kind::Permutation;
        target_sum |= n.kind == Kind::Summation;
        target_perm |= n.kind == Kind::Permutation;
      }
      if (unary && (!source_sum || target_sum) && (!source_perm || target_perm))
        return true;
    }
    Metadata m;
    std::size_t ports = 0;
    for (auto cap : p.source.back().captures) {
      auto it = f.ranks.find(cap);
      if (it == f.ranks.end() || it->second > 256) return false;
      ports += it->second;
      if (ports > 4096) return false;
      m.shapes[cap] = std::vector<std::uint64_t>(it->second, 2);
    }
    for (const auto& n : p.source)
      if (n.descriptor) {
        auto& fields = m.descriptors[*n.descriptor];
        fields.resize(n.fields.size());
        for (auto field : n.fields) {
          auto it = f.fields.find({*n.descriptor, field});
          if (it == f.fields.end()) return false;
          fields[field] = it->second;
        }
      }
    return predicate(emit(p), m, limits).status == Status::Feasible;
  } catch (const std::invalid_argument&) {
    return false;
  }
}
}  // namespace tepl::inference
