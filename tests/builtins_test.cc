#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "src/core/pipeline.h"
#include "src/core/syntax/parse.h"
#include "src/rewrite/builtins/dispatch.h"
#include "src/rewrite/intrinsics.h"
#include "src/shared/builtins/axes.h"
#include "src/shared/builtins/intrinsics/catalog.h"

namespace {
using namespace tepl;
using namespace tepl::rewrite;
using B = core::builtins::Builtin;
using I = core::intrinsics::Intrinsic;
Value ints(std::initializer_list<Integer> values) {
  List xs;
  for (auto value : values) xs.emplace_back(value);
  return xs;
}
Value axes(std::initializer_list<std::size_t> values) {
  List out;
  for (auto id : values) out.emplace_back(Axis{0, 0, id, 2});
  return out;
}
Value bools(std::initializer_list<bool> values) {
  List xs;
  for (auto value : values) xs.emplace_back(value);
  return xs;
}
template <class F>
void fails(F run) {
  bool rejected = false;
  try {
    run();
  } catch (const EvaluationError&) {
    rejected = true;
  }
  assert(rejected);
}
void nativeContracts() {
  struct Case {
    B id;
    std::vector<Value> args;
    Value result;
  };
  const std::vector<Case> cases = {
      {B::kAxisDifference, {axes({2, 0, 1}), axes({0, 2})}, axes({1})},
      {B::kAxisFilter, {axes({2, 0, 1}), axes({0, 2})}, axes({2, 0})},
      {B::kAxisSubsequence, {axes({2, 1}), axes({2, 0, 1})}, true},
      {B::kLen, {ints({4, 5})}, Integer(2)},
      {B::kRange, {Integer(3)}, ints({0, 1, 2})},
      {B::kConcat, {ints({1}), ints({2}), ints({3})}, ints({1, 2, 3})},
      {B::kGather, {ints({4, 5}), ints({1, 0, 1})}, ints({5, 4, 5})},
      {B::kExclude, {ints({2, 3, 2, 4}), ints({2})}, ints({3, 4})},
      {B::kSlice, {ints({2, 3, 4}), Integer(1), Integer(3)}, ints({3, 4})},
      {B::kReplace, {ints({2, 3}), Integer(1), Integer(8)}, ints({2, 8})},
      {B::kToI64, {Integer(-7)}, Integer(-7)},
      {B::kToU64, {Integer(7)}, Integer(7)},
      {B::kSum, {ints({2, 3})}, Integer(5)},
      {B::kProduct, {ints({2, 3})}, Integer(6)},
      {B::kAll, {bools({true, false})}, false},
      {B::kAny, {bools({false, true})}, true},
      {B::kContains, {ints({2, 3}), Integer(3)}, true},
      {B::kIsDisjoint, {ints({2}), ints({3})}, true},
      {B::kIsValidAxisList, {ints({2, 0}), Integer(3)}, true},
      {B::kBroadcastShape, {ints({2, 1, 4}), ints({3, 4})}, ints({2, 3, 4})},
      {B::kMin, {Integer(-7), Integer(3)}, Integer(-7)},
      {B::kMax, {Integer(-7), Integer(3)}, Integer(3)},
      {B::kFloorDiv, {Integer(-7), Integer(3)}, Integer(-3)},
      {B::kCeilDiv, {Integer(-7), Integer(3)}, Integer(-2)},
      {B::kIsFloat, {std::string("bf16")}, true},
      {B::kIsInteger, {std::string("u8")}, true},
      {B::kIsSignedInteger, {std::string("i64")}, true},
      {B::kIsUnsignedInteger, {std::string("u64")}, true},
      {B::kIsNumeric, {std::string("bool")}, false},
  };
  std::set<B> covered;
  Budget budget;
  for (const auto& c : cases) {
    assert(covered.insert(c.id).second);
    assert(evaluateBuiltin(c.id, c.args, budget) == c.result);
    auto too_few = c.args;
    // Variadic concat permits two; remove enough to cross its minimum.
    too_few.resize(core::builtins::signature(c.id).arguments.size() - 1);
    fails([&] { evaluateBuiltin(c.id, too_few, budget); });
  }
  std::size_t contextual = 0;
  for (const auto& entry : core::builtins::catalog())
    if (core::builtins::isAxisQuery(entry.builtin)) {
      assert(!covered.contains(entry.builtin));
      std::vector<Value> args(entry.arguments.size(), Integer(0));
      fails([&] { evaluateBuiltin(entry.builtin, args, budget); });
      ++contextual;
    }
  assert(covered.size() + contextual == core::builtins::catalog().size());
  assert(!boolean(evaluateBuiltin(B::kAxisSubsequence,
                                  {axes({1, 2}), axes({2, 0, 1})}, budget)));
  assert(boolean(evaluateBuiltin(B::kAxisSubsequence,
                                 {axes({}), axes({2, 0, 1})}, budget)));
  fails(
      [&] { evaluateBuiltin(B::kAxisFilter, {ints({0}), ints({0})}, budget); });
  assert(boolean(evaluateBuiltin(B::kAll, {bools({})}, budget)));
  assert(!boolean(evaluateBuiltin(B::kAny, {bools({})}, budget)));
  assert(integer(evaluateBuiltin(B::kSum, {ints({})}, budget)) == 0);
  assert(integer(evaluateBuiltin(B::kProduct, {ints({})}, budget)) == 1);
  auto max = std::numeric_limits<Integer>::max();
  fails([&] { evaluateBuiltin(B::kToU64, {Integer(-1)}, budget); });
  fails([&] {
    evaluateBuiltin(B::kToI64,
                    {Integer(std::numeric_limits<std::int64_t>::max()) + 1},
                    budget);
  });
  fails([&] { evaluateBuiltin(B::kToU64, {max}, budget); });
  fails([&] { evaluateBuiltin(B::kToI64, {true}, budget); });
  assert(integer(evaluateBuiltin(B::kProduct, {ints({max, 2, 0})}, budget)) ==
         0);
  fails([&] { evaluateBuiltin(B::kProduct, {ints({max, 2})}, budget); });
  fails([&] { evaluateBuiltin(B::kSum, {ints({max, 1})}, budget); });
  fails([&] { evaluateBuiltin(B::kRange, {Integer(-1)}, budget); });
  fails([&] { evaluateBuiltin(B::kGather, {ints({2}), ints({-1})}, budget); });
  fails([&] { evaluateBuiltin(B::kGather, {ints({2}), ints({1})}, budget); });
  fails([&] {
    evaluateBuiltin(B::kSlice, {ints({2}), Integer(1), Integer(0)}, budget);
  });
  fails(
      [&] { evaluateBuiltin(B::kCeilDiv, {Integer(1), Integer(0)}, budget); });
  fails([&] {
    evaluateBuiltin(B::kBroadcastShape, {ints({2}), ints({3})}, budget);
  });
  assert(evaluateBuiltin(B::kBroadcastShape, {ints({0}), ints({1})}, budget) ==
         ints({0}));
  assert(!boolean(
      evaluateBuiltin(B::kIsValidAxisList, {ints({}), Integer(-1)}, budget)));
  assert(!boolean(evaluateBuiltin(B::kIsValidAxisList,
                                  {ints({0, 0}), Integer(1)}, budget)));
  assert(
      !boolean(evaluateBuiltin(B::kIsValidAxisList, {ints({}), max}, budget)));
  Budget limited;
  limited.max_list = 2;
  bool limited_out = false;
  try {
    evaluateBuiltin(B::kConcat, {ints({1, 2}), ints({3})}, limited);
  } catch (const LimitError&) {
    limited_out = true;
  }
  assert(limited_out);
}
void rejects(const std::string& source, const std::string& diagnostic) {
  auto parsed = parse(source, "builtins.tepl");
  if (!parsed.ok()) {
    std::cerr << source << "\n";
    for (const auto& d : parsed.diagnostics) std::cerr << d.message << "\n";
  }
  assert(parsed.ok());
  auto checked = core::inspect(*parsed.program);
  assert(!checked.ok());
  bool found =
      std::any_of(checked.diagnostics.begin(), checked.diagnostics.end(),
                  [&](const auto& d) {
                    return d.message.find(diagnostic) != std::string::npos;
                  });
  if (!found) {
    std::cerr << source << "\nExpected: " << diagnostic << "\n";
    for (const auto& d : checked.diagnostics) std::cerr << d.message << "\n";
  }
  assert(found);
}
void visibility() {
  using namespace core::builtins;
  std::set<std::string_view> names;
  for (const auto& entry : catalog()) {
    assert(names.insert(entry.name).second);
    assert(resolveBuiltin(entry.name) ==
           (isPublicBuiltin(entry.builtin) ? &entry : nullptr));
    assert(&signature(entry.builtin) == &entry);
  }
  names.clear();
  for (const auto& entry : axisCatalog()) {
    assert(names.insert(entry.name).second);
    if (auto id = std::get_if<B>(&entry.target)) {
      assert(entry.variadic == signature(*id).variadic);
      assert(entry.arguments.size() == signature(*id).arguments.size());
    }
  }
  for (const auto& entry : core::intrinsics::catalog()) {
    if (entry.intrinsic == I::kConditional)
      continue;  // `if` is syntax, not a call name.
    if (resolveBuiltin(entry.name))
      continue;  // Structural concat shares a spelling.
    const auto name = std::string(entry.name);
    rejects("rule r { X => X where " + name + "() }", "unknown builtin");
    rejects("dialect D { op copy(x:tensor)->tensor { shape(s) { yield " + name +
                "() } } }",
            "unknown shape builtin");
    if (!resolveAxisBuiltin(name))
      rejects(
          "dialect D { op copy(x:tensor)->tensor { axes { bind S { "
          "x:all_axes(x) } assert " +
              name + "() yield S } semantics {yield x } } }",
          "unsupported axis builtin");
  }
  rejects(
      "dialect D { op copy(x:tensor)->tensor { shape(s) { assert "
      "is_float(f32) yield s } } }",
      "not available in shape");
  rejects("rule r { X => X where rank(X) == 0 }", "unknown builtin");
  rejects("rule r { X => X where i64(true) == 0 }",
          "operator does not support");
  rejects("rule r { X => X where u64(1.0) == 0 }", "operator does not support");

  rejects(
      "dialect D { op copy(x:tensor)->tensor { axes { bind S { x:all_axes(x) "
      "} assert all([1]) yield S } semantics {yield x } } }",
      "unsupported axis builtin");
}
void axesAndIntrinsics() {
  Budget budget;
  const std::vector<Value> gather = {ints({4, 5}), ints({1, 0, 1})};
  assert(evaluateAxisCall("permute", gather, budget) ==
         evaluateBuiltin(B::kGather, gather, budget));
  assert(evaluateAxisCall("gather", gather, budget) ==
         evaluateBuiltin(B::kGather, gather, budget));
  const std::vector<Value> exclude = {ints({2, 3, 2}), ints({2})};
  assert(evaluateAxisCall("ordered_complement", exclude, budget) ==
         evaluateBuiltin(B::kExclude, exclude, budget));
  assert(evaluateAxisCall("all_axes", {ints({2, 3})}, budget) == ints({0, 1}));
  assert(integer(evaluateAxisCall("rank", {ints({2, 3})}, budget)) == 2);
  assert(evaluateAxisCall("shape", {ints({2, 3})}, budget) == ints({2, 3}));
  assert(!boolean(
      evaluateIntrinsic(I::kAxisPartition, {ints({0}), Integer(2)}, budget)));
  assert(boolean(
      evaluateBuiltin(B::kIsValidAxisList, {ints({0}), Integer(2)}, budget)));
  assert(evaluateIntrinsic(I::kConcat, {}, budget) == ints({}));
  fails([&] { evaluateBuiltin(B::kConcat, {}, budget); });
  fails([&] {
    evaluateAxisCall("axis_partition", {ints({0}), Integer(1)}, budget);
  });
  fails([&] { evaluateIntrinsic(I::kInverse, {ints({0, 0})}, budget); });
  fails([&] { evaluateSymbolicCall("dtype_unknown", {}, budget); });
}
}  // namespace
int main() {
  nativeContracts();
  visibility();
  axesAndIntrinsics();
}
