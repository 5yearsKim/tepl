#include <tepl/runtime/axes.h>

#include <cassert>
#include <limits>

namespace rt = tepl_runtime;
int main() {
  auto fail = [](auto action, rt::ErrorKind kind = rt::ErrorKind::Invalid) {
    auto result = rt::evaluate(action);
    assert(!result);
    assert(result.error().kind == kind);
  };
  auto max = rt::integer("170141183460469231731687303715884105727").integer();
  auto min = rt::integer("-170141183460469231731687303715884105728").integer();
  assert(max == std::numeric_limits<rt::Integer>::max());
  assert(min == std::numeric_limits<rt::Integer>::min());
  fail([&] { rt::integer("170141183460469231731687303715884105728"); });
  fail([&] { rt::builtins::arithmetic("+", max, 1); });
  fail([&] { rt::builtins::arithmetic("-", min, 1); });
  fail([&] { rt::builtins::arithmetic("*", max, 2); });
  fail([&] { rt::builtins::arithmetic("/", min, -1); });
  fail([&] { rt::builtins::arithmetic("%", min, -1); });
  fail([&] { rt::builtins::arithmetic("/", 1, 0); });
  fail([&] { rt::from_value<std::uint64_t>(rt::integer("-1")); });
  fail([&] {
    rt::from_value<std::int64_t>(rt::integer("9223372036854775808"));
  });
  fail([&] { rt::floating("1e400"); });
  rt::Budget b;
  assert(rt::builtins::call("floor_div", {rt::integer("-7"), rt::integer("3")},
                            b) == rt::integer("-3"));
  assert(rt::builtins::call("ceil_div", {rt::integer("-7"), rt::integer("3")},
                            b) == rt::integer("-2"));
  assert(rt::builtins::call(
             "product",
             {rt::Value(rt::Value::List{rt::Value(max), rt::Value(max),
                                        rt::integer("0")})},
             b) == rt::integer("0"));
  fail([&] {
    rt::builtins::index(rt::Value(rt::Value::List{}), rt::integer("0"), b);
  });
  rt::Budget small;
  small.max_list = 2;
  fail([&] { rt::builtins::call("range", {rt::integer("3")}, small); },
       rt::ErrorKind::Limit);
  fail(
      [&] {
        rt::builtins::call("concat", {rt::shape({0, 1}), rt::shape({2})},
                           small);
      },
      rt::ErrorKind::Limit);
  fail(
      [&] {
        rt::intrinsics::call(
            "place_axes",
            {rt::shape({0, 1}), rt::shape({0, 1}), rt::shape({2})}, small);
      },
      rt::ErrorKind::Limit);
  small.steps = 0;
  fail([&] { small.tick(); }, rt::ErrorKind::Limit);
  rt::Fact<std::vector<std::uint64_t>> unknown,
      known = decltype(unknown)::known({2, 3});
  assert(unknown.merge(known));
  assert(unknown == known);
  assert(!known.merge({}));
  assert(known.merge(decltype(known)::known({3, 2})));
  assert(known.kind == decltype(known)::Kind::Conflict);
  auto a = rt::TensorFacts::known({2, 3}, rt::DType::F32);
  auto c = a;
  c.shape = {};
  assert(!a.disagrees(c));
  c.dtype = rt::Fact<rt::DType>::known(rt::DType::I32);
  assert(a.disagrees(c));
  rt::Value axis1(rt::Axis{0, 0, 0, 2}), axis2(rt::Axis{0, 0, 1, 2});
  assert(axis1 != axis2);
  fail([&] {
    rt::checked(rt::integer("0"), rt::ScalarType::LogicalAxis, 0, false);
  });
  assert(rt::checked(axis1, rt::ScalarType::LogicalAxis, 0, false) == axis1);
  auto bytes = rt::Region{{1, 2, 3}};
  assert(rt::from_value<rt::Region>(rt::to_value(bytes)) == bytes);
  std::vector<std::vector<bool>> nested{{true, false}, {}};
  assert(rt::from_value<decltype(nested)>(rt::to_value(nested)) == nested);
  std::vector<rt::TensorFacts> operands{
      rt::TensorFacts::known({2, 2}, rt::DType::F32)};
  rt::Value attrs;
  rt::axes::AxisContext ctx(operands, attrs, b);
  ctx.begin_group();
  ctx.bind(0, rt::shape({0, 1}));
  ctx.finish_group({});
  ctx.cover(0);
  auto axes = ctx.group(0);
  assert(axes.list()[0] != axes.list()[1]);
  assert(ctx.output(axes) == std::vector<std::uint64_t>({2, 2}));
  fail([&] {
    ctx.output(rt::Value(rt::Value::List{axes.list()[0], axes.list()[0]}));
  });
}
