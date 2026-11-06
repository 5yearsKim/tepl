#include <limits>

#include "src/core/semantics/symbolic/metadata_sequences.h"

namespace tepl::core::semantics {
void Context::checkConstructor(const Instance& n, std::size_t field,
                               const S& value) {
  validateSource(value);
  const auto& attributes = plan.attributes(n);
  if (attributes.needsSynthesis() && !attributes.isAuto(field)) {
    if (!equal(canonical(attributes.fields.at(field)), canonical(value)))
      throw ProofFailure("candidate changed a supplied non-auto attribute",
                         FailureKind::InvalidCandidate);
  }
  if (attributes.isAuto(field) && !n.definition->axis_fields.contains(field))
    throw ProofFailure("auto is restricted to axis attributes",
                       FailureKind::InvalidCandidate);
  auto type = plan.metadata_types.find({n.descriptor->value, field});
  if (callIs(value, "none")) {
    if (type != plan.metadata_types.end() && type->second.optional) return;
    throw ProofFailure("none requires an optional attribute",
                       FailureKind::InvalidCandidate);
  }
  auto present_type =
      type != plan.metadata_types.end() ? type->second : core::Type{};
  present_type.optional = false;
  if (type != plan.metadata_types.end() && !elementType(present_type)) {
    const auto sort = scalarMetadataSort(present_type);
    if (sort == value->sort) {
      checkMetadataType(value, type->second);
      return;
    }
    // Opaque attributes can be forwarded unchanged, but cannot acquire a new
    // scalar type merely because their tensor-value semantics ignore them.
    if (!sort && value->kind == Symbol::Kind::Field) {
      auto source = plan.metadata_types.find({value->owner, value->index});
      if (source != plan.metadata_types.end() && source->second == type->second)
        return;
    }
    throw ProofFailure("constructor does not match scalar attribute '" +
                           n.definition->fields.at(field) + "'",
                       FailureKind::InvalidCandidate);
  }
  if (value->sort != Symbol::Sort::Positions &&
      !(n.definition->extent_fields.contains(field) &&
        value->sort == Symbol::Sort::Extents))
    throw ProofFailure("constructor must be a position or extent sequence",
                       FailureKind::InvalidCandidate);
}
void Context::checkMetadataType(const S& raw, core::Type type) {
  tick();
  if (type.optional && callIs(raw, "none")) return;
  type.optional = false;
  auto element = elementType(type);
  auto x = canonical(raw);
  if (element) {
    if (x->kind == Symbol::Kind::List) {
      for (const auto& item : x->args) checkMetadataType(item, *element);
      return;
    }
    if (elementType(*element))
      throw ProofFailure(
          "symbolic nested attribute bounds are not established");
    if (element->kind != TypeKind::kU64 && element->kind != TypeKind::kI64)
      return;
    auto zero = call("extent_sub", {x, x});
    auto minimum =
        call("offset_extents",
             {zero, make(Symbol::Kind::Integer, element->kind == TypeKind::kU64
                                                    ? "0"
                                                    : "-9223372036854775808")});
    auto maximum =
        call("offset_extents",
             {zero, make(Symbol::Kind::Integer, element->kind == TypeKind::kU64
                                                    ? "18446744073709551615"
                                                    : "9223372036854775807")});
    require(call("extent_le", {minimum, x}));
    require(call("extent_le", {x, maximum}));
  } else if (type.kind == TypeKind::kU64 || type.kind == TypeKind::kI64) {
    auto interval = integerInterval(x);
    MetadataInteger lo = type.kind == TypeKind::kU64
                             ? 0
                             : std::numeric_limits<std::int64_t>::min();
    MetadataInteger hi =
        type.kind == TypeKind::kU64
            ? MetadataInteger(std::numeric_limits<std::uint64_t>::max())
            : std::numeric_limits<std::int64_t>::max();
    if (!interval || interval->first < lo || interval->second > hi)
      throw ProofFailure(
          "derived metadata is not established within its attribute type");
  }
}
void Context::checkMetadata(const S& raw) {
  tick();
  validateSource(raw);
  if (callIs(raw, "if")) {
    checkMetadata(raw->args[0]);
    auto condition = canonical(raw->args[0]);
    if (condition->kind == Symbol::Kind::Boolean) {
      checkMetadata(raw->args[truth(condition) ? 1 : 2]);
      return;
    }
    // Proving both branches defined is sufficient; never assume a branch.
    checkMetadata(raw->args[1]);
    checkMetadata(raw->args[2]);
    return;
  }
  if (callIs(raw, "sequence_map")) {
    checkMetadata(raw->args[0]);
    auto seq = canonical(raw->args[0]);
    if (seq->kind == Symbol::Kind::List && seq->args.size() <= 64) {
      for (const auto& x : seq->args)
        checkMetadata(substituteMetadata(raw->args[2], raw->args[1], x));
      return;
    }
    auto lifted =
        liftMetadataMap(*this, raw->args[0], raw->args[1], raw->args[2]);
    if (!lifted)
      throw ProofFailure(
          "comprehension is outside the pointwise metadata fragment: " +
          format(*raw));
    checkMetadata(lifted);
    return;
  }
  for (const auto& x : raw->args) checkMetadata(x);
  if (raw->kind != Symbol::Kind::Call) return;
  auto x = canonical(raw);
  if (callIs(raw, "index") || callIs(raw, "replace")) {
    require(call(">=", {raw->args[1], integer(0)}));
    require(call("<", {raw->args[1], call("len", {raw->args[0]})}));
  } else if (callIs(raw, "slice")) {
    require(call(">=", {raw->args[1], integer(0)}));
    require(call("<=", {raw->args[1], raw->args[2]}));
    require(call("<=", {raw->args[2], call("len", {raw->args[0]})}));
  } else if (callIs(raw, "range")) {
    require(call(">=", {raw->args[0], integer(0)}));
  } else if (callIs(raw, "i64") || callIs(raw, "u64")) {
    checkMetadataType(raw->args.at(0),
                      {callIs(raw, "i64") ? TypeKind::kI64 : TypeKind::kU64});
  } else if (callIs(raw, "+") || callIs(raw, "-") || callIs(raw, "*")) {
    if (!integerInterval(raw))
      throw ProofFailure("checked metadata arithmetic may overflow: " +
                         format(*raw));
  } else if (callIs(raw, "gather")) {
    auto positions = canonical(raw->args[1]);
    auto length = call("len", {raw->args[0]});
    if (positions->kind == Symbol::Kind::List) {
      // General metadata gather permits repeated positions.
      for (const auto& i : positions->args) {
        require(call(">=", {i, integer(0)}));
        require(call("<", {i, length}));
      }
    } else if (callIs(positions, "range")) {
      require(call(">=", {positions->args[0], integer(0)}));
      require(call("<=", {positions->args[0], length}));
    } else
      require(call("is_valid_axis_list", {positions, length}));
  } else if (callIs(raw, "extent_add") || callIs(raw, "extent_sub") ||
             callIs(raw, "extent_mul") || callIs(raw, "offset_extents") ||
             callIs(raw, "offset_positions") || callIs(raw, "scale_extents")) {
    if (callIs(raw, "extent_add") || callIs(raw, "extent_sub") ||
        callIs(raw, "extent_mul"))
      require(call("==",
                   {call("len", {raw->args[0]}), call("len", {raw->args[1]})}));
    // This sufficient bound keeps vector arithmetic in the union of source
    // index/i64 ranges. Scalar intermediates use the full checked i128 domain.
    auto zero = call("extent_sub", {x, x});
    auto minimum =
        call("offset_extents",
             {zero, make(Symbol::Kind::Integer, "-9223372036854775808")});
    auto maximum =
        call("offset_extents",
             {zero, make(Symbol::Kind::Integer, "18446744073709551615")});
    require(call("extent_le", {minimum, x}));
    require(call("extent_le", {x, maximum}));
  } else if (callIs(raw, "inverse")) {
    require(
        call("axis_partition", {raw->args[0], call("len", {raw->args[0]})}));
  } else if (callIs(raw, "/") || callIs(raw, "%") || callIs(raw, "sum") ||
             callIs(raw, "product") || callIs(raw, "floor_div") ||
             callIs(raw, "ceil_div") || callIs(raw, "broadcast_shape")) {
    // These need dedicated checked-domain proofs. Preserve Unknown rather
    // than erase an evaluation failure during later value simplification.
    throw ProofFailure("checked metadata evaluation is not established: " +
                       raw->name);
  }
}
}  // namespace tepl::core::semantics
