#pragma once
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <variant>

#include "tests/index_reference.h"

namespace tepl::test {
using namespace core::semantics;
using Coordinate = std::tuple<std::size_t, std::size_t, std::size_t>;
using Coordinates = std::vector<Coordinate>;
using Datum = std::variant<std::int64_t, bool, List, Coordinates, std::string>;

// Independent interpreter for the synthesized pure metadata program. It does
// not call compiler lowering, matching, normalization, or concrete inference.
struct ConstructorEvaluator {
  const Metadata& metadata;
  const RuleProof& proof;
  std::vector<SymbolPtr> assumptions;
  std::map<const Symbol*, Datum> cache;
  // Keep queried nodes alive: synthesized temporary expressions can otherwise
  // reuse an address and collide with the pointer-keyed cache.
  std::map<const Symbol*, SymbolPtr> retained;
  std::map<std::pair<std::size_t, std::size_t>, SymbolPtr> groups;
  ConstructorEvaluator(const Metadata& m, const RuleProof& p)
      : metadata(m), proof(p) {
    for (const auto& g : p.axis_definitions)
      if (g->kind == Symbol::Kind::AxisGroup) groups[{g->owner, g->index}] = g;
  }
  ConstructorEvaluator(const Metadata& m, const Plan& p)
      : ConstructorEvaluator(m, p.proof) {
    assumptions = p.assumptions;
  }
  template <class T>
  static bool distinct(const std::vector<T>& xs) {
    return std::set<T>(xs.begin(), xs.end()).size() == xs.size();
  }
  static bool valid(const List& p, std::int64_t n) {
    if (!distinct(p)) return false;
    for (auto x : p)
      if (x < 0 || x >= n) return false;
    return true;
  }
  std::int64_t length(const Datum& d) {
    if (const auto* n = std::get_if<List>(&d)) return n->size();
    if (const auto* n = std::get_if<Coordinates>(&d)) return n->size();
    throw std::runtime_error("len of nonsequence");
  }
  Datum eval(const SymbolPtr& s) {
    auto found = cache.find(s.get());
    if (found != cache.end()) return found->second;
    auto value = compute(*s);
    cache[s.get()] = value;
    retained[s.get()] = s;
    return value;
  }
  Datum compute(const Symbol& s) {
    if (s.kind == Symbol::Kind::Integer) return std::stoll(s.name);
    if (s.kind == Symbol::Kind::Boolean) return s.name == "true";
    if (s.kind == Symbol::Kind::Field)
      return metadata.descriptors.at(s.owner).at(s.index);
    if (s.kind == Symbol::Kind::Shape) return metadata.shapes.at(s.owner);
    if (s.kind == Symbol::Kind::DType) return std::string("f64");
    if (s.kind == Symbol::Kind::AxisGroup) {
      auto definition = groups.at({s.owner, s.index});
      auto domain = std::get<List>(eval(definition->args.at(0)));
      Coordinates out;
      for (std::size_t i = 0; i < domain.size(); ++i)
        out.emplace_back(s.owner, s.index, i);
      return out;
    }
    std::vector<Datum> args;
    for (const auto& a : s.args) args.push_back(eval(a));
    if (s.kind == Symbol::Kind::List) {
      List out;
      for (const auto& a : args) out.push_back(std::get<std::int64_t>(a));
      return out;
    }
    if (s.kind != Symbol::Kind::Call)
      throw std::runtime_error("unresolved constructor");
    const auto& n = s.name;
    if (n == "extent_add" || n == "extent_sub" || n == "extent_le" ||
        n == "extent_mul" || n == "extent_floor_div") {
      auto a = std::get<List>(args[0]), b = std::get<List>(args[1]);
      if (a.size() != b.size())
        throw std::runtime_error("extent tuple lengths differ");
      List out;
      bool less = true;
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (n == "extent_floor_div") {
          if (b[i] <= 0) throw std::runtime_error("nonpositive floor divisor");
          // C++ integer division truncates; metadata floor division does not.
          out.push_back(a[i] / b[i] - (a[i] % b[i] < 0 ? 1 : 0));
        } else
          out.push_back(n == "extent_mul"   ? a[i] * b[i]
                        : n == "extent_sub" ? a[i] - b[i]
                                            : a[i] + b[i]);
        less &= a[i] <= b[i];
      }
      return n == "extent_le" ? Datum{less} : Datum{out};
    }
    if (n == "scale_extents") {
      auto out = std::get<List>(args[0]);
      for (auto& x : out) x *= std::get<std::int64_t>(args[1]);
      return out;
    }
    if (n == "offset_extents") {
      auto out = std::get<List>(args[0]);
      for (auto& x : out) x += std::get<std::int64_t>(args[1]);
      return out;
    }
    if (n == "is_positive_extents") {
      auto a = std::get<List>(args[0]);
      return std::all_of(a.begin(), a.end(), [](auto x) { return x > 0; });
    }
    if (n == "empty_axes") return Coordinates{};
    if (n == "place_axes") {
      auto left = std::get<Coordinates>(args[0]),
           right = std::get<Coordinates>(args[2]);
      auto positions = std::get<List>(args[1]);
      if (!valid(positions, left.size() + right.size()) ||
          positions.size() != left.size())
        throw std::runtime_error("invalid axis placement");
      Coordinates result(left.size() + right.size());
      std::vector<bool> occupied(result.size());
      for (std::size_t i = 0; i < left.size(); ++i) {
        result[positions[i]] = left[i];
        occupied[positions[i]] = true;
      }
      std::size_t next = 0;
      for (std::size_t i = 0; i < result.size(); ++i)
        if (!occupied[i]) result[i] = right[next++];
      return result;
    }
    if (n == "is_valid_extents") {
      auto domain = std::get<List>(args[0]);
      return std::all_of(domain.begin(), domain.end(),
                         [](auto n) { return n >= 0; });
    }
    if (n == "len") return length(args[0]);
    if (n == "index")
      return std::get<List>(args[0]).at(std::get<std::int64_t>(args[1]));
    if (n == "slice") {
      auto xs = std::get<List>(args[0]);
      auto start = std::get<std::int64_t>(args[1]),
           end = std::get<std::int64_t>(args[2]);
      if (start < 0 || end < start || end > std::int64_t(xs.size()))
        throw std::runtime_error("invalid slice");
      return List(xs.begin() + start, xs.begin() + end);
    }
    if (n == "==") return args[0] == args[1];
    if (n == "!=") return args[0] != args[1];
    if (n == "&&") return std::get<bool>(args[0]) && std::get<bool>(args[1]);
    if (n == "||") return std::get<bool>(args[0]) || std::get<bool>(args[1]);
    if (n == "!") return !std::get<bool>(args[0]);
    if (n == "+" || n == "-" || n == "*" || n == "<" || n == "<=" || n == ">" ||
        n == ">=") {
      auto a = std::get<std::int64_t>(args[0]),
           b = std::get<std::int64_t>(args[1]);
      if (n == "+") return a + b;
      if (n == "-") return a - b;
      if (n == "*") return a * b;
      if (n == "<") return a < b;
      if (n == "<=") return a <= b;
      if (n == ">") return a > b;
      return a >= b;
    }
    if (n == "is_numeric" || n == "is_float")
      return std::get<std::string>(args[0]) == "f64";
    if (n == "range") {
      auto size = std::get<std::int64_t>(args[0]);
      if (size < 0) throw std::runtime_error("negative range");
      List out(size);
      std::iota(out.begin(), out.end(), 0);
      return out;
    }
    if (n == "inverse") {
      auto p = std::get<List>(args[0]);
      if (!valid(p, p.size()))
        throw std::runtime_error("nonpermutation inverse");
      List out(p.size());
      for (std::size_t i = 0; i < p.size(); ++i) out[p[i]] = i;
      return out;
    }
    if (n == "order_positions") {
      auto positions = std::get<List>(args[0]);
      List order(positions.size());
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return positions[a] < positions[b];
      });
      return order;
    }
    if (n == "offset_positions") {
      auto p = std::get<List>(args[0]);
      for (auto& x : p) x += std::get<std::int64_t>(args[1]);
      return p;
    }
    if (n == "ordered_complement") {
      auto p = std::get<List>(args[0]), excluded = std::get<List>(args[1]);
      List out;
      for (auto x : p)
        if (std::find(excluded.begin(), excluded.end(), x) == excluded.end())
          out.push_back(x);
      return out;
    }
    if (n == "membership_positions" || n == "matched_positions") {
      const auto& p = std::get<List>(args[0]);
      const auto& q = std::get<List>(args[1]);
      List out;
      for (std::size_t i = 0; i < p.size(); ++i)
        if (auto at = std::find(q.begin(), q.end(), p[i]); at != q.end())
          out.push_back(n == "membership_positions" ? i : at - q.begin());
      return out;
    }
    if (n == "concat") {
      if (s.sort == Symbol::Sort::Axes) {
        Coordinates out;
        for (const auto& a : args) {
          auto xs = std::get<Coordinates>(a);
          out.insert(out.end(), xs.begin(), xs.end());
        }
        return out;
      }
      List out;
      for (const auto& a : args) {
        auto xs = std::get<List>(a);
        out.insert(out.end(), xs.begin(), xs.end());
      }
      return out;
    }
    if (n == "gather") {
      auto p = std::get<List>(args[1]);
      if (const auto* source = std::get_if<Coordinates>(&args[0])) {
        Coordinates out;
        for (auto i : p) out.push_back(source->at(i));
        return out;
      }
      auto source = std::get<List>(args[0]);
      List out;
      for (auto i : p) out.push_back(source.at(i));
      return out;
    }
    if (n == "is_valid_axis_list")
      return valid(std::get<List>(args[0]), std::get<std::int64_t>(args[1]));
    if (n == "axis_partition") {
      auto p = std::get<List>(args[0]);
      auto rank = std::get<std::int64_t>(args[1]);
      return valid(p, rank) && p.size() == std::size_t(rank);
    }
    if (n == "distinct_axes") return distinct(std::get<Coordinates>(args[0]));
    if (n == "union_axes" || n == "unique_axes") {
      Coordinates out;
      for (const auto& a : args)
        for (const auto& x : std::get<Coordinates>(a))
          if (std::find(out.begin(), out.end(), x) == out.end())
            out.push_back(x);
      return out;
    }
    if (n == "filter_axes" || n == "intersection_axes" ||
        n == "difference_axes" || n == "same_set" || n == "disjoint" ||
        n == "subset") {
      auto a = std::get<Coordinates>(args[0]),
           b = std::get<Coordinates>(args[1]);
      Coordinates out;
      for (const auto& x : a) {
        bool present = std::find(b.begin(), b.end(), x) != b.end();
        if ((n == "difference_axes") ? !present : present) out.push_back(x);
      }
      if (n == "same_set")
        return std::set<Coordinate>(a.begin(), a.end()) ==
               std::set<Coordinate>(b.begin(), b.end());
      if (n == "disjoint") return out.empty();
      if (n == "subset") return out.size() == a.size();
      return out;
    }
    if (n == "positions") {
      auto source = std::get<Coordinates>(args[0]),
           wanted = std::get<Coordinates>(args[1]);
      if (!distinct(source) || !distinct(wanted))
        throw std::runtime_error("noninjective position lookup");
      List out;
      for (const auto& x : wanted) {
        auto it = std::find(source.begin(), source.end(), x);
        if (it == source.end()) throw std::runtime_error("absent coordinate");
        out.push_back(it - source.begin());
      }
      return out;
    }
    if (n == "replace_axes") {
      auto source = std::get<Coordinates>(args[0]),
           from = std::get<Coordinates>(args[1]),
           to = std::get<Coordinates>(args[2]);
      if (from.size() != to.size() || !distinct(from) || !distinct(to))
        throw std::runtime_error("invalid coordinate transport");
      for (auto& x : source) {
        auto it = std::find(from.begin(), from.end(), x);
        if (it != from.end()) x = to[it - from.begin()];
      }
      return source;
    }
    if (n == "axis_extents") {
      List out;
      for (const auto& x : std::get<Coordinates>(args[0])) {
        auto [owner, group, member] = x;
        auto domain = std::get<List>(eval(groups.at({owner, group})->args[0]));
        out.push_back(domain.at(member));
      }
      return out;
    }
    throw std::runtime_error("unsupported metadata instruction: " + n);
  }
  bool applicable() {
    try {
      for (const auto& g : assumptions)
        if (!std::get<bool>(eval(g))) return false;
      for (const auto& g : proof.guard)
        if (!std::get<bool>(eval(g))) return false;
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  std::map<std::size_t, std::vector<List>> construct() {
    std::map<std::size_t, std::vector<List>> result;
    for (const auto& [id, fields] : proof.constructors)
      for (const auto& f : fields)
        result[id].push_back(std::get<List>(eval(f)));
    return result;
  }
};
}  // namespace tepl::test
