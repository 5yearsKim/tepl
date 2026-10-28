#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "src/codegen/rust/names.h"
#include "src/rewrite/builtins/dispatch.h"
#include "src/rewrite/intrinsics.h"
namespace {
using namespace tepl;
using namespace tepl::rewrite;
using B = core::builtins::Builtin;
using I = core::intrinsics::Intrinsic;
Value ints(std::initializer_list<Integer> xs) {
  List out;
  for (auto x : xs) out.emplace_back(x);
  return out;
}
std::string emit(const Value& v) {
  return std::visit(
      [](const auto& x) -> std::string {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>)
          return "Value::None";
        else if constexpr (std::is_same_v<T, Integer>)
          return "Value::Integer(" + integerText(x) + "i128)";
        else if constexpr (std::is_same_v<T, bool>)
          return x ? "Value::Boolean(true)" : "Value::Boolean(false)";
        else if constexpr (std::is_same_v<T, std::string>)
          return "Value::String(" + codegen::rust::quote(x) + ".into())";
        else if constexpr (std::is_same_v<T, List>) {
          std::string out = "Value::List(vec![";
          for (const auto& a : x) out += emit(a) + ",";
          return out + "])";
        } else if constexpr (std::is_same_v<T, Axis>)
          return "Value::Axis(rt::values::Axis {owner:" +
                 std::to_string(x.owner) + ",group:" + std::to_string(x.group) +
                 ",member:" + std::to_string(x.member) +
                 ",extent:" + integerText(x.extent) + "})";
        else
          throw std::runtime_error("unexpected reference value");
      },
      v.data);
}
std::string args(const std::vector<Value>& xs) {
  std::string out = "vec![";
  for (const auto& x : xs) out += emit(x) + ",";
  return out + "]";
}
template <class F>
void check(const char* domain, std::string_view name,
           const std::vector<Value>& a, F eval) {
  std::cout << "// " << domain << "::" << name << "\n";
  const auto call = "rt::" + std::string(domain) + "::call(" +
                    codegen::rust::quote(name) + ", " + args(a) +
                    ", &mut rt::Budget::default())";
  try {
    Budget budget;
    auto expected = eval(a, budget);
    std::cout << "assert_eq!(" << call << ".unwrap(), " << emit(expected)
              << ");\n";
  } catch (const EvaluationError&) {
    std::cout << "assert!(matches!(" << call
              << ",Err(rt::EvalError::Invalid(_))));\n";
  }
}
void builtin(B id, std::vector<Value> a) {
  check("builtins", core::builtins::builtinName(id), a,
        [&](const auto& x, auto& b) { return evaluateBuiltin(id, x, b); });
}
}  // namespace
int main() {
  std::cout << "use tepl_runtime::{self as rt,Value};\n#[test]\nfn "
               "executable_evaluator_reference(){\n";
  const Value x = Axis{0, 0, 0, 2}, y = Axis{0, 1, 0, 2};
  const Value xy = List{x, y}, yx = List{y, x}, xx = List{x, x}, empty = List{};
  std::vector<std::pair<B, std::vector<Value>>> cases = {
      {B::kLen, {ints({4, 5})}},
      {B::kRange, {Integer(3)}},
      {B::kConcat, {ints({1}), ints({2}), ints({3})}},
      {B::kGather, {ints({4, 5}), ints({1, 0, 1})}},
      {B::kExclude, {ints({2, 3, 2, 4}), ints({2})}},
      {B::kSlice, {ints({2, 3, 4}), Integer(1), Integer(3)}},
      {B::kReplace, {ints({2, 3}), Integer(1), Integer(8)}},
      {B::kToI64, {Integer(-7)}},
      {B::kToU64, {Integer(7)}},
      {B::kSum, {ints({2, 3})}},
      {B::kProduct, {ints({2, 3})}},
      {B::kAll, {List{true, false}}},
      {B::kAny, {List{false, true}}},
      {B::kContains, {ints({2, 3}), Integer(3)}},
      {B::kIsDisjoint, {ints({2}), ints({3})}},
      {B::kIsValidAxisList, {ints({2, 0}), Integer(3)}},
      {B::kBroadcastShape, {ints({2, 1, 4}), ints({3, 4})}},
      {B::kMin, {Integer(-7), Integer(3)}},
      {B::kMax, {Integer(-7), Integer(3)}},
      {B::kFloorDiv, {Integer(-7), Integer(3)}},
      {B::kCeilDiv, {Integer(-7), Integer(3)}},
      {B::kIsFloat, {std::string("bf16")}},
      {B::kIsInteger, {std::string("u8")}},
      {B::kIsSignedInteger, {std::string("i64")}},
      {B::kIsUnsignedInteger, {std::string("u64")}},
      {B::kIsNumeric, {std::string("bool")}},
      {B::kAxisFilter, {xy, List{x}}},
      {B::kAxisDifference, {xy, List{y}}},
      {B::kAxisSubsequence, {List{x}, xy}}};
  const auto public_count =
      std::count_if(core::builtins::catalog().begin(),
                    core::builtins::catalog().end(), [](const auto& entry) {
                      return core::builtins::isPublicBuiltin(entry.builtin);
                    });
  assert(cases.size() == static_cast<std::size_t>(public_count));
  for (auto [id, a] : cases) {
    builtin(id, a);
    a.resize(core::builtins::signature(id).arguments.size() - 1);
    builtin(id, a);
  }
  auto max = std::numeric_limits<Integer>::max();
  for (auto [id, a] : std::vector<std::pair<B, std::vector<Value>>>{
           {B::kToI64, {Integer(std::numeric_limits<std::int64_t>::max())}},
           {B::kToI64, {Integer(std::numeric_limits<std::int64_t>::max()) + 1}},
           {B::kToI64, {Integer(std::numeric_limits<std::int64_t>::min())}},
           {B::kToU64, {Integer(std::numeric_limits<std::uint64_t>::max())}},
           {B::kToU64,
            {Integer(std::numeric_limits<std::uint64_t>::max()) + 1}},
           {B::kToU64, {Integer(-1)}},
           {B::kToI64, {true}},
           {B::kProduct, {ints({max, 2, 0})}},
           {B::kProduct, {ints({max, 2})}},
           {B::kSum, {ints({max, 1})}},
           {B::kRange, {Integer(-1)}},
           {B::kGather, {ints({1}), ints({2})}},
           {B::kCeilDiv, {Integer(1), Integer(0)}},
           {B::kBroadcastShape, {ints({0}), ints({1})}},
           {B::kBroadcastShape, {ints({2}), ints({3})}}})
    builtin(id, a);
  std::vector<std::pair<I, std::vector<Value>>> internal = {
      {I::kConcat, {}},
      {I::kIndex, {ints({4, 5}), Integer(1)}},
      {I::kConditional, {true, Integer(1), Integer(2)}},
      {I::kEmptyAxes, {}},
      {I::kAxisPartition, {ints({0}), Integer(2)}},
      {I::kInverse, {ints({2, 0, 1})}},
      {I::kOrderPositions, {ints({3, 1, 1})}},
      {I::kOffsetPositions, {ints({1, 2}), Integer(2)}},
      {I::kOffsetExtents, {ints({1, 2}), Integer(-1)}},
      {I::kScaleExtents, {ints({1, 2}), Integer(2)}},
      {I::kExtentAdd, {ints({2}), ints({3})}},
      {I::kExtentSub, {ints({2}), ints({3})}},
      {I::kExtentMul, {ints({2}), ints({3})}},
      {I::kExtentFloorDiv, {ints({7}), ints({2})}},
      {I::kExtentLe, {ints({2}), ints({3})}},
      {I::kIsValidExtents, {ints({0, 2})}},
      {I::kIsPositiveExtents, {ints({0, 2})}},
      {I::kDistinctAxes, {xy}},
      {I::kAxisExtents, {xy}},
      {I::kPositions, {xy, yx}},
      {I::kMembershipPositions, {xy, List{y}}},
      {I::kMatchedPositions, {xy, yx}},
      {I::kUnionAxes, {xy, yx}},
      {I::kUniqueAxes, {xx}},
      {I::kOrderedComplement, {ints({0, 1, 2}), ints({1})}},
      {I::kFilterAxes, {xy, List{y}}},
      {I::kIntersectionAxes, {xy, List{y}}},
      {I::kDifferenceAxes, {xy, List{y}}},
      {I::kSameSet, {xy, yx}},
      {I::kDisjoint, {xy, empty}},
      {I::kSubset, {List{x}, xy}},
      {I::kReplaceAxes, {xy, List{x}, List{y}}},
      {I::kPlaceAxes, {List{x}, ints({1}), List{y}}}};
  assert(internal.size() == core::intrinsics::catalog().size());
  for (auto [id, a] : internal)
    check("intrinsics", core::intrinsics::signature(id).name, a,
          [&](const auto& v, auto& b) { return evaluateIntrinsic(id, v, b); });
  std::cout << "}\n";
}
