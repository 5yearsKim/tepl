#include "src/inference/insertion.h"

#include <algorithm>
#include <charconv>
#include <stdexcept>

#include "src/inference/extensions.h"

namespace tepl::inference {
namespace {
bool core(Kind kind) {
  return kind == Kind::Input || kind == Kind::Permutation ||
         kind == Kind::Summation || kind == Kind::Contraction ||
         kind == Kind::Reverse;
}
struct Normalizer {
  const Program& original;
  Emission result;
  Program kernel;
  EmissionBlock metadata;
  std::size_t next = 0;
  struct Value {
    std::size_t node, placement, extents, shape;
    bool inserted;
  };
  explicit Normalizer(const Program& p) : original(p) {
    result.original = p;
    metadata.source = p;
    for (const auto* code : {&p.source, &p.replacement})
      for (const auto& n : *code) {
        next = std::max(next, n.node + 1);
        if (n.descriptor) next = std::max(next, *n.descriptor + 1);
      }
  }
  std::size_t expr(Op op, std::vector<std::size_t> args = {},
                   std::size_t capture = 0) {
    metadata.expressions.push_back({op, std::move(args), capture});
    return metadata.expressions.size() - 1;
  }
  std::size_t unary(Op op, std::size_t a) { return expr(op, {a}); }
  std::size_t binary(Op op, std::size_t a, std::size_t b) {
    return expr(op, {a, b});
  }
  std::size_t empty() { return expr(Op::Empty); }
  std::size_t range(std::size_t shape) {
    return unary(Op::Range, unary(Op::Length, shape));
  }
  std::size_t field(std::size_t node, std::size_t role = 0) {
    static constexpr Op fields[] = {Op::SourceField, Op::SourceExtents,
                                    Op::SourceField2, Op::SourceField3};
    return expr(fields[role], {}, node);
  }
  void check(Op op, std::size_t a, std::size_t b) {
    metadata.checks.push_back(binary(op, a, b));
  }
  std::size_t append(Kind kind, std::vector<std::size_t> inputs,
                     std::vector<std::size_t> fields) {
    Instruction n;
    n.kind = kind;
    n.node = next++;
    n.descriptor = next++;
    n.inputs = std::move(inputs);
    std::set<std::size_t> captures;
    for (auto input : n.inputs)
      captures.insert(kernel.source[input].captures.begin(),
                      kernel.source[input].captures.end());
    n.captures.assign(captures.begin(), captures.end());
    for (std::size_t i = 0; i < fields.size(); ++i) n.fields.push_back(i);
    metadata.attributes[*n.descriptor] = std::move(fields);
    kernel.source.push_back(std::move(n));
    return kernel.source.size() - 1;
  }
  Value bare(std::size_t node, std::size_t shape) {
    return {node, range(shape), empty(), shape, false};
  }
  Value normalize(std::size_t index) {
    const auto& n = original.source[index];
    auto shape = expr(Op::SourceNodeShape, {}, index);
    if (n.kind == Kind::Input) {
      kernel.source.push_back(n);
      check(Op::ExtentsAtLeast, shape, expr(Op::Singleton, {}, 2));
      return bare(kernel.source.size() - 1, shape);
    }
    auto left = normalize(n.inputs[0]);
    if (n.kind == Kind::Broadcast) {
      auto a = field(index), e = field(index, 1);
      check(Op::ExtentsAtLeast, e, expr(Op::Singleton, {}, 1));
      if (left.inserted) {
        e = expr(Op::MergeInsertionExtents,
                 {left.placement, left.extents, a, e});
        a = binary(Op::Gather, a, left.placement);
      }
      return {left.node, a, e, left.shape, true};
    }
    if (n.kind == Kind::Contraction) {
      auto right = normalize(n.inputs[1]);
      if (right.inserted)
        throw std::invalid_argument(
            "restricted insertion remains inside the right dot operand");
      std::vector<std::size_t> fields;
      for (std::size_t i = 0; i < 4; ++i) fields.push_back(field(index, i));
      if (left.inserted) {
        auto all =
            unary(Op::Range, binary(Op::Plus, unary(Op::Length, left.placement),
                                    unary(Op::Length, left.extents)));
        auto inserted = binary(Op::Complement, all, left.placement);
        check(Op::Subset, inserted, fields[2]);
        check(Op::Subset, fields[0], left.placement);
        auto removed = binary(Op::Gather, fields[3],
                              binary(Op::Pullback, fields[2], inserted));
        right.node = append(Kind::Summation, {right.node}, {removed});
        auto batch = binary(Op::Positions, left.placement, fields[0]);
        auto contract = binary(Op::Positions, left.placement,
                               binary(Op::Complement, fields[2], inserted));
        auto free = binary(Op::Complement, range(left.shape),
                           binary(Op::Concat, batch, contract));
        check(Op::Equal, binary(Op::Gather, left.placement, free),
              binary(Op::Complement, all,
                     binary(Op::Concat, fields[0], fields[2])));
        fields = {
            batch, binary(Op::Remap, fields[1], removed), contract,
            binary(Op::Remap,
                   binary(Op::Gather, fields[3],
                          binary(Op::Pullback, fields[2], left.placement)),
                   removed)};
      }
      return bare(append(n.kind, {left.node, right.node}, fields), shape);
    }
    if (n.kind == Kind::Permutation && left.inserted) {
      auto permutation = field(index);
      auto a =
          binary(Op::Gather, unary(Op::Inverse, permutation), left.placement);
      auto e = expr(Op::BroadcastTransposeExtents,
                    {left.shape, left.placement, left.extents, permutation});
      return {left.node, a, e, left.shape, true};
    }
    if (n.kind == Kind::Summation && left.inserted) {
      auto selected = field(index);
      check(Op::Subset, selected, left.placement);
      auto pulled = binary(Op::Pullback, left.placement, selected);
      auto inner_shape =
          binary(Op::Gather, left.shape,
                 binary(Op::Complement, range(left.shape), pulled));
      return {append(n.kind, {left.node}, {pulled}),
              binary(Op::Remap, left.placement, selected), left.extents,
              inner_shape, true};
    }
    if (n.kind == Kind::Reverse && left.inserted) {
      auto pulled = binary(Op::Pullback, left.placement, field(index));
      left.node = append(n.kind, {left.node}, {pulled});
      return left;
    }
    if (!core(n.kind))
      throw std::invalid_argument(
          "restricted insertion has a non-core interior");
    return bare(append(n.kind, {left.node}, {field(index)}), shape);
  }
  std::size_t term(const MetadataTerm& t) {
    if (t.op.starts_with("#")) {
      std::int64_t value;
      auto parsed =
          std::from_chars(t.op.data() + 1, t.op.data() + t.op.size(), value);
      if (!t.args.empty() || parsed.ec != std::errc{} ||
          parsed.ptr != t.op.data() + t.op.size() || value < 0)
        throw std::invalid_argument("unsupported supplied insertion integer");
      return expr(Op::Singleton, {}, value);
    }
    if (t.op.starts_with("shape:") || t.op.starts_with("field:")) {
      for (std::size_t i = 0; i < original.source.size(); ++i) {
        const auto& n = original.source[i];
        if (n.capture && t == termShape(*n.capture))
          return expr(Op::SourceShape, {}, i);
        if (n.descriptor)
          for (std::size_t role = 0; role < n.fields.size(); ++role)
            if (t == termField(*n.descriptor, n.fields[role]))
              return field(i, role);
      }
      throw std::invalid_argument(
          "supplied insertion program reads non-source metadata");
    }
    std::vector<std::size_t> args;
    for (const auto& arg : t.args) args.push_back(term(arg));
    if (t.op == "list") {
      auto value = empty();
      for (auto arg : args) value = binary(Op::Concat, value, arg);
      return value;
    }
    static const std::map<std::string, std::pair<Op, std::size_t>> operations =
        {
            {"len", {Op::Length, 1}},         {"range", {Op::Range, 1}},
            {"inverse", {Op::Inverse, 1}},    {"+", {Op::Plus, 2}},
            {"gather", {Op::Gather, 2}},      {"index", {Op::Gather, 2}},
            {"exclude", {Op::Complement, 2}}, {"concat", {Op::Concat, 2}},
            {"scatter_n", {Op::Scatter, 3}},
        };
    auto found = operations.find(t.op);
    if (found == operations.end() || args.size() != found->second.second)
      throw std::invalid_argument(
          "unsupported supplied insertion metadata operation: " + t.op);
    return expr(found->second.first, std::move(args));
  }
  Emission run() {
    if (!insertionCandidate(original))
      throw std::invalid_argument("outside restricted insertion syntax");
    auto lhs = normalize(original.source.size() - 1);
    auto rhs = original.replacement.back();
    bool wrapped = rhs.kind == Kind::Broadcast;
    kernel.replacement = original.replacement;
    Emission::Insertion insertion;
    insertion.placement = lhs.placement;
    insertion.extents = lhs.extents;
    std::size_t supplied = empty();
    if (wrapped) {
      if (original.supplied.size() != 1 ||
          original.supplied[0].descriptor != *rhs.descriptor ||
          original.supplied[0].field != rhs.fields[1])
        throw std::invalid_argument(
            "restricted insertion requires supplied extents and independent "
            "axis holes");
      supplied = term(original.supplied[0].value);
      insertion.descriptor = *rhs.descriptor;
      insertion.auxiliary = next++;
      auto& transpose = kernel.replacement.back();
      transpose.kind = Kind::Permutation;
      transpose.operation.reset();
      transpose.descriptor = insertion.auxiliary;
      transpose.fields = {0};
    } else {
      if (!original.supplied.empty())
        throw std::invalid_argument(
            "restricted insertion has supplied inner RHS fields");
      append(Kind::Permutation, {lhs.node},
             {unary(Op::Inverse, lhs.placement)});
    }
    insertion.supplied = supplied;
    insertion.compatible = binary(Op::Equal, lhs.extents, supplied);
    for (const auto& value : original.supplied_checks)
      metadata.checks.push_back(unary(Op::Defined, term(value)));
    // Finite syntax only. Generated source descriptors read the normalization
    // DAG.
    kernel.components = decompose(kernel);
    validate(kernel);
    result.blocks.push_back(std::move(metadata));
    result.blocks.push_back(emitCoreBlock(kernel));
    result.insertion = insertion;
    return std::move(result);
  }
};
}  // namespace
bool insertionCandidate(const Program& p) {
  if (p.mode != Mode::Synthesize || p.source.empty() || p.replacement.empty())
    return false;
  bool insertion = false;
  for (const auto& n : p.source) {
    insertion |= n.kind == Kind::Broadcast;
    if (!core(n.kind) && n.kind != Kind::Broadcast) return false;
  }
  for (std::size_t i = 0; i < p.replacement.size(); ++i) {
    auto kind = p.replacement[i].kind;
    if (kind == Kind::Broadcast && i + 1 == p.replacement.size())
      insertion = true;
    else if (!core(kind))
      return false;
  }
  return insertion;
}
Emission emitInsertion(const Program& p) { return Normalizer(p).run(); }
Result executeInsertion(const Emission& e, const Metadata& m,
                        const Limits& limits, bool build, bool sound_only,
                        ExecutionTiming* timing) {
  if (timing) *timing = {};
  const auto& insertion = *e.insertion;
  std::vector<std::size_t> outputs = {insertion.placement, insertion.extents,
                                      insertion.compatible};
  if (insertion.descriptor) outputs.push_back(insertion.supplied);
  auto normalized =
      executeMetadata(e.blocks[0], m, limits, outputs, insertion.compatible);
  Result result;
  result.work = normalized.work;
  if (normalized.status != Status::Analyzed) {
    result.status = normalized.status;
    if (sound_only && (result.status == Status::OutsideDomain ||
                       result.status == Status::Impossible ||
                       result.status == Status::InvalidSource))
      result.status = Status::Inapplicable;
    result.reason = normalized.reason;
    return result;
  }
  Metadata inner = m;
  for (auto& [descriptor, fields] : normalized.attributes)
    inner.descriptors[descriptor] = std::move(fields);
  auto constructed =
      executeCoreBlock(e.blocks[1], inner,
                       {limits.work - result.work, limits.axes}, build, timing);
  constructed.work += result.work;
  if (sound_only && constructed.status == Status::Impossible)
    constructed.status = Status::Inapplicable;
  if (constructed.status != Status::Solved || !insertion.descriptor)
    return constructed;
  auto auxiliary = constructed.attributes.extract(*insertion.auxiliary);
  const auto& permutation = auxiliary.mapped().at(0);
  const auto& placement = normalized.values.at(insertion.placement);
  if (permutation.size() > limits.work - constructed.work) {
    const auto spent = constructed.work;
    constructed = {};
    constructed.work = spent;
    constructed.status = Status::ResourceLimit;
    constructed.reason = "insertion lift work limit";
    return constructed;
  }
  constructed.work += permutation.size();
  List lifted(permutation.size(), -1);
  for (std::size_t j = 0; j < permutation.size(); ++j)
    lifted.at(permutation[j]) = placement.at(j);
  constructed.attributes[*insertion.descriptor] = {
      std::move(lifted), normalized.values.at(insertion.supplied)};
  return constructed;
}
}  // namespace tepl::inference
