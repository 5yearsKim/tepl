#include <algorithm>
#include <array>
#include <charconv>
#include <limits>

#include "src/core/semantics/context.h"

namespace tepl::core::semantics {
namespace {
using Range = IntegerInterval;
constexpr auto low = std::numeric_limits<MetadataInteger>::min();
constexpr auto high = std::numeric_limits<MetadataInteger>::max();
constexpr MetadataInteger fieldLow = std::numeric_limits<std::int64_t>::min();
constexpr MetadataInteger fieldHigh = std::numeric_limits<std::uint64_t>::max();
}  // namespace
std::optional<MetadataInteger> metadataInteger(const S& x) {
  if (x->kind != Symbol::Kind::Integer || x->name.empty()) return {};
  std::size_t i = 0;
  bool negative = false;
  if (x->name[0] == '-' || x->name[0] == '+') negative = x->name[i++] == '-';
  if (i == x->name.size()) return {};
  MetadataInteger value = 0;
  for (; i < x->name.size(); ++i) {
    auto digit = x->name[i] - '0';
    if (digit < 0 || digit > 9 ||
        __builtin_mul_overflow(value, MetadataInteger(10), &value))
      return {};
    if (negative
            ? __builtin_sub_overflow(value, MetadataInteger(digit), &value)
            : __builtin_add_overflow(value, MetadataInteger(digit), &value))
      return {};
  }
  return value;
}
std::vector<S> Context::arithmeticFacts() const {
  std::vector<S> result;
  for (const auto& f : facts_.raw)
    if (callIs(f, "<") || callIs(f, "<=") || callIs(f, ">") ||
        callIs(f, ">=") || callIs(f, "=="))
      result.push_back(f);
  return result;
}
S Context::uniformElement(const S& seq) {
  tick();
  auto x = canonical(seq);
  auto id = key(x);
  if (auto found = queries_.uniform_elements.find(id);
      found != queries_.uniform_elements.end())
    return found->second;
  if (!queries_.active_uniform_elements.insert(id).second) return {};
  ActiveKey active{queries_.active_uniform_elements, id};
  if (queries_.active_uniform_elements.size() > ProofLimits::query_depth)
    return {};
  S result;
  auto constant = [&](const S& a) -> S {
    if (callIs(a, "sequence_map") && a->args[2]->kind == Symbol::Kind::Integer)
      return a->args[2];
    if (a->kind == Symbol::Kind::List && !a->args.empty() &&
        a->args[0]->kind == Symbol::Kind::Integer &&
        std::all_of(a->args.begin(), a->args.end(),
                    [&](const S& b) { return equal(a->args[0], b); }))
      return a->args[0];
    return {};
  };
  result = constant(x);
  if (!result && callIs(x, "gather") &&
      x->args[1]->kind == Symbol::Kind::List && !x->args[1]->args.empty() &&
      evidence(call("is_valid_axis_list",
                    {x->args[1], call("len", {x->args[0]})}))) {
    for (const auto& position : x->args[1]->args) {
      auto value = canonical(call("index", {x->args[0], position}));
      if (!metadataInteger(value) || (result && !equal(result, value))) {
        result = {};
        break;
      }
      result = value;
    }
  }
  // Track an element value independently of the symbolic tuple length. These
  // are value facts only: constructor definedness still checks every original
  // operand, length, selection and arithmetic intermediate.
  if (!result && callIs(x, "extent_sub") && equal(x->args[0], x->args[1]))
    result = integer(0);
  if (!result && (callIs(x, "offset_extents") || callIs(x, "scale_extents"))) {
    if (auto value = uniformElement(x->args[0]))
      result =
          call(callIs(x, "offset_extents") ? "+" : "*", {value, x->args[1]});
  }
  if (!result &&
      (callIs(x, "extent_add") || callIs(x, "extent_sub") ||
       callIs(x, "extent_mul")) &&
      evidence(
          call("==", {call("len", {x->args[0]}), call("len", {x->args[1]})}))) {
    auto a = uniformElement(x->args[0]), b = uniformElement(x->args[1]);
    if (a && b)
      result = call(callIs(x, "extent_add")   ? "+"
                    : callIs(x, "extent_sub") ? "-"
                                              : "*",
                    {a, b});
  }
  if (!result && callIs(x, "gather") &&
      evidence(
          call("is_valid_axis_list", {x->args[1], call("len", {x->args[0]})})))
    result = uniformElement(x->args[0]);
  if (!result && callIs(x, "concat")) {
    for (const auto& part : x->args) {
      if (equal(canonical(call("len", {part})), integer(0))) continue;
      auto value = uniformElement(part);
      if (!value || (result && !equal(result, value))) {
        result = {};
        break;
      }
      result = value;
    }
  }
  // Checked scalar folding leaves an expression on overflow. Such a value
  // must not become a literal coefficient in extent polynomial reasoning.
  if (result && !metadataInteger(result)) result = {};
  if (!result)
    for (const auto& f : facts_.raw) {
      if (!callIs(f, "==")) continue;
      for (int i = 0; i < 2; ++i) {
        auto value = constant(f->args[1 - i]);
        if (!value || !equal(canonical(f->args[i]), x)) continue;
        result = value;
        break;
      }
      if (result) break;
    }
  queries_.uniform_elements[id] = result;
  return result;
}
IntegerInterval Context::fieldIntegerInterval(const S& x) const {
  if (auto found = plan.metadata_types.find({x->owner, x->index});
      found != plan.metadata_types.end()) {
    auto type = found->second;
    while (auto element = elementType(type)) type = *element;
    if (type.kind == TypeKind::kU64) return {0, fieldHigh};
    if (type.kind == TypeKind::kI64)
      return {fieldLow, std::numeric_limits<std::int64_t>::max()};
  }
  return {fieldLow, fieldHigh};
}
std::optional<Range> Context::integerInterval(const S& raw) {
  tick();
  auto id = key(raw);
  if (auto found = queries_.intervals.find(id);
      found != queries_.intervals.end())
    return found->second;
  if (!queries_.active_intervals.insert(id).second) return {};
  ActiveKey active{queries_.active_intervals, id};
  if (queries_.active_intervals.size() > ProofLimits::query_depth) return {};
  auto x = canonical(raw);
  if (auto n = metadataInteger(x)) return Range{*n, *n};
  Range out{low, high};
  bool defined = false;
  if (x->kind == Symbol::Kind::Field) {
    out = fieldIntegerInterval(x);
    defined = true;
  }
  if (callIs(x, "len")) {
    out = {0, fieldHigh};
    defined = true;
  }
  if (callIs(x, "index")) {
    auto i = integerInterval(x->args[1]);
    if (i && i->first >= 0 &&
        entails(call("<", {x->args[1], call("len", {x->args[0]})}))) {
      defined = true;
      if (x->args[0]->kind == Symbol::Kind::Shape)
        out = {0, fieldHigh};
      else if (x->args[0]->kind == Symbol::Kind::Field)
        out = fieldIntegerInterval(x->args[0]);
      if (entails(call("is_valid_extents", {x->args[0]}))) out.first = 0;
      if (entails(call("is_positive_extents", {x->args[0]}))) out.first = 1;
      if (auto value = uniformElement(x->args[0]))
        if (auto n = metadataInteger(value)) out = {*n, *n};
    }
  }
  if (callIs(x, "+") || callIs(x, "-") || callIs(x, "*")) {
    if (x->args.size() == 1) {
      auto a = integerInterval(x->args[0]);
      if (a && (callIs(x, "+") || a->first > low))
        return callIs(x, "+") ? a
                              : std::optional<Range>{{-a->second, -a->first}};
      return {};
    }
    auto a = integerInterval(x->args[0]), b = integerInterval(x->args[1]);
    if (a && b) {
      MetadataInteger l = low, h = high;
      bool bounded = true;
      if (callIs(x, "+"))
        bounded = !__builtin_add_overflow(a->first, b->first, &l) &&
                  !__builtin_add_overflow(a->second, b->second, &h);
      else if (callIs(x, "-"))
        bounded = !__builtin_sub_overflow(a->first, b->second, &l) &&
                  !__builtin_sub_overflow(a->second, b->first, &h);
      else {
        std::array<MetadataInteger, 4> products;
        bounded = !__builtin_mul_overflow(a->first, b->first, &products[0]) &&
                  !__builtin_mul_overflow(a->first, b->second, &products[1]) &&
                  !__builtin_mul_overflow(a->second, b->first, &products[2]) &&
                  !__builtin_mul_overflow(a->second, b->second, &products[3]);
        if (bounded) {
          l = *std::min_element(products.begin(), products.end());
          h = *std::max_element(products.begin(), products.end());
        }
      }
      if (bounded) {
        out = {l, h};
        defined = true;
      }
    }
  }
  // True source conditions establish both evaluation success and bounds.
  // Never refine intervals from an unproved target obligation.
  for (const auto& f : arithmeticFacts()) {
    auto a = canonical(f->args[0]), b = canonical(f->args[1]);
    auto n = metadataInteger(b);
    std::string op = f->name;
    if (!equal(a, x)) {
      if (!equal(b, x) || !(n = metadataInteger(a))) continue;
      if (op == "<")
        op = ">";
      else if (op == ">")
        op = "<";
      else if (op == "<=")
        op = ">=";
      else if (op == ">=")
        op = "<=";
    }
    if (!n) continue;
    defined = true;
    if (op == ">=" || op == "==") out.first = std::max(out.first, *n);
    if (op == "<=" || op == "==") out.second = std::min(out.second, *n);
    if (op == ">" && *n < high) out.first = std::max(out.first, *n + 1);
    if (op == "<" && *n > low) out.second = std::min(out.second, *n - 1);
  }
  if (!defined || out.first > out.second) return {};
  queries_.intervals[id] = out;
  return out;
}
bool Context::integerComparison(const S& f, QueryMode mode) {
  if (mode == QueryMode::Evidence) return false;
  auto a = integerInterval(f->args[0]), b = integerInterval(f->args[1]);
  if (!a || !b) return false;
  if (f->name == "<") return a->second < b->first;
  if (f->name == "<=")
    return a->second <= b->first ||
           equal(canonical(f->args[0]), canonical(f->args[1]));
  if (f->name == ">") return a->first > b->second;
  if (f->name == ">=")
    return a->first >= b->second ||
           equal(canonical(f->args[0]), canonical(f->args[1]));
  return false;
}
}  // namespace tepl::core::semantics
