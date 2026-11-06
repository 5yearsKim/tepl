#include <functional>

#include "src/core/semantics/graph.h"
#include "src/core/semantics/symbolic/tree.h"

namespace tepl::core::semantics {
E Lowerer::rename(const E& e, const S& from, const S& to) {
  context.tick();
  if ((e->kind == Expr::Kind::Const || e->kind == Expr::Kind::Number) ||
      equal(from, to))
    return e;
  context.transportFrame(from, to);
  if (e->kind == Expr::Kind::Read) {
    auto axes = context.transportInterface(from, to, e->axes);
    // These are consequences of legal source coordinate composition.
    obligations.push_back(call("distinct_axes", {axes}));
    obligations.push_back(call(
        "==",
        {context.extents(axes), make(Symbol::Kind::Shape, {}, e->capture)}));
    return std::make_shared<Expr>(
        Expr{e->kind,
             e->capture,
             e->occurrence,
             axes,
             {},
             0,
             e->access ? indexRename(context, e->access, from, to) : I{}});
  }
  return mapArguments(e, [&](const E& arg) { return rename(arg, from, to); });
}
I Lowerer::indexValue(const IndexValue& v, const Instance& n,
                      const std::vector<TensorMeaning>& inputs,
                      const std::vector<S>& groups) {
  if (v.kind == IndexValue::Kind::Coordinates)
    return coordinates(groups.at(v.group));
  std::vector<I> args;
  for (const auto& a : v.args)
    args.push_back(indexValue(*a, n, inputs, groups));
  if (v.kind == IndexValue::Kind::Reverse) {
    auto positions = metadata(*v.parameter, n, inputs, groups);
    auto length = indexLength(args[0]);
    obligations.push_back(call("is_valid_axis_list", {positions, length}));
    auto extents = context.extents(args[0]->axes);
    context.note("index-reverse-normalization");
    return indexReverse(context, args[0], extents, positions);
  }
  if (v.kind == IndexValue::Kind::Select) {
    auto positions = metadata(*v.parameter, n, inputs, groups);
    obligations.push_back(
        call("is_valid_axis_list", {positions, indexLength(args[0])}));
    context.note("index-selection-normalization");
    return indexSelect(context, args[0], positions);
  }
  if (v.kind == IndexValue::Kind::Scale || v.kind == IndexValue::Kind::Offset) {
    auto scale = metadata(*v.parameter, n, inputs, groups);
    obligations.push_back(
        call("==", {call("len", {scale}), indexLength(args[0])}));
    bool offset = v.kind == IndexValue::Kind::Offset;
    obligations.push_back(
        call(offset ? "is_valid_extents" : "is_positive_extents", {scale}));
    context.note(offset ? "index-offset-normalization"
                        : "index-scale-normalization");
    return offset ? indexOffset(context, scale, args[0])
                  : indexScale(context, scale, args[0]);
  }
  if (v.kind == IndexValue::Kind::Concat) return indexConcat(args);
  for (std::size_t i = 1; i < args.size(); ++i)
    obligations.push_back(
        call("==", {indexLength(args[0]), indexLength(args[i])}));
  return indexAdd(args);
}
E Lowerer::evaluateAt(const E& e, const S& frame, const I& access) {
  context.tick();
  if (e->kind == Expr::Kind::Const || e->kind == Expr::Kind::Number) return e;
  if (e->kind == Expr::Kind::Read) {
    auto composed = indexSubstitute(
        context, e->access ? e->access : coordinates(e->axes), frame, access);
    return std::make_shared<Expr>(
        Expr{e->kind, e->capture, e->occurrence, e->axes, {}, 0, composed});
  }
  if (e->kind == Expr::Kind::Reduce)
    context.require(call("disjoint", {e->axes, indexDependencies(access)}),
                    GuardKind::ReductionDomain);
  return mapArguments(
      e, [&](const E& arg) { return evaluateAt(arg, frame, access); });
}
E Lowerer::value(const Value& v, const Instance& n,
                 const std::vector<TensorMeaning>& inputs,
                 const std::vector<S>& groups) {
  context.tick();
  if (v.kind == Value::Kind::Const) return constant(v.constant);
  if (v.kind == Value::Kind::Operand) {
    auto out = std::make_shared<Expr>(*inputs.at(v.index).value);
    out->dtype = inputs.at(v.index).dtype;
    return out;
  }
  if (v.kind == Value::Kind::Read) {
    const auto& input = inputs.at(v.index);
    auto access = indexValue(*v.access, n, inputs, groups);
    context.note("index-add-normalization");
    context.note("index-substitution");
    obligations.push_back(
        call("==", {indexLength(access), call("len", {input.shape})}));
    // Reflection is decreasing, so both bounds are checked. For an empty
    // coordinate domain the interval may be empty (upper < lower); no element
    // is read. In particular reflecting [0,-1] about -1 gives [0,-1].
    obligations.push_back(
        call("is_valid_extents", {indexLowerBound(context, access)}));
    obligations.push_back(call(
        "extent_le", {indexUpperBound(context, access),
                      call("offset_extents",
                           {input.shape, make(Symbol::Kind::Integer, "-1")})}));
    return evaluateAt(input.value, input.axes, access);
  }
  std::vector<E> args;
  for (const auto& a : v.args) args.push_back(value(*a, n, inputs, groups));
  if (v.kind == Value::Kind::Call) {
    const auto& signature = scalarSignature(v.function);
    std::vector<S> types;
    std::function<S(const Value&)> typeOf = [&](const Value& x) -> S {
      if (x.kind == Value::Kind::Operand || x.kind == Value::Kind::Read)
        return inputs[x.index].dtype;
      if (x.kind == Value::Kind::Const) return {};
      if (x.kind == Value::Kind::Call) {
        auto f = scalarSignature(x.function);
        if (f.types == ScalarTypes::Boolean || f.types == ScalarTypes::Compare)
          return call("dtype_bool", {});
        if (f.types == ScalarTypes::Select) return typeOf(*x.args[1]);
      }
      for (const auto& a : x.args)
        if (auto t = typeOf(*a)) return t;
      return {};
    };
    for (const auto& a : v.args) types.push_back(typeOf(*a));
    S result;
    if (signature.types == ScalarTypes::Boolean) {
      result = call("dtype_bool", {});
      for (const auto& t : types)
        if (t) context.require(call("==", {t, result}));
    } else if (signature.types == ScalarTypes::Select) {
      auto boolean = call("dtype_bool", {});
      if (types[0]) context.require(call("==", {types[0], boolean}));
      result = types[1] ? types[1] : types[2];
      if (types[1] && types[2])
        context.require(call("==", {types[1], types[2]}));
    } else {
      for (const auto& t : types)
        if (t) {
          if (result)
            context.require(call("==", {result, t}));
          else
            result = t;
        }
      if (result)
        context.require(
            signature.types == ScalarTypes::Compare
                ? call("||", {call("is_numeric", {result}),
                              call("==", {result, call("dtype_bool", {})})})
                : call(signature.types == ScalarTypes::Floating ? "is_float"
                                                                : "is_numeric",
                       {result}));
      if (signature.types == ScalarTypes::Compare)
        result = call("dtype_bool", {});
    }
    if (!result && signature.types != ScalarTypes::Compare)
      result = dtype(n, inputs);
    if (signature.types == ScalarTypes::Floating)
      context.require(call("is_float", {result}));
    if (signature.types == ScalarTypes::Numeric)
      context.require(call("is_numeric", {result}));
    auto booleanConstant = [&](const E& value) {
      if ((value->kind == Expr::Kind::Const && value->constant != 0 &&
           value->constant != 1) ||
          value->kind == Expr::Kind::Number)
        throw ProofFailure("Boolean scalar constants must be zero or one");
    };
    if (signature.types == ScalarTypes::Boolean)
      for (const auto& a : args) booleanConstant(a);
    if (signature.types == ScalarTypes::Compare)
      for (const auto& t : types)
        if (t && context.entails(call("==", {t, call("dtype_bool", {})})))
          for (const auto& a : args) booleanConstant(a);
    if (signature.types == ScalarTypes::Select) {
      booleanConstant(args[0]);
      if (result &&
          context.entails(call("==", {result, call("dtype_bool", {})}))) {
        booleanConstant(args[1]);
        booleanConstant(args[2]);
      }
    }
    std::vector<S> parameters;
    for (const auto& p : v.parameters)
      parameters.push_back(metadata(*p, n, inputs, groups));
    return scalarCall(v.function, std::move(args), std::move(parameters),
                      result);
  }
  auto k = v.kind == Value::Kind::Add       ? Expr::Kind::Add
           : v.kind == Value::Kind::Product ? Expr::Kind::Product
                                            : Expr::Kind::Reduce;
  return expression(k, std::move(args),
                    v.kind == Value::Kind::Reduce ? groups.at(v.index) : S{});
}

}  // namespace tepl::core::semantics
