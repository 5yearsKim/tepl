#include <algorithm>
#include <functional>
#include <numeric>
#include <set>

#include "src/core/semantics/context.h"
#include "src/core/semantics/graph.h"
#include "src/core/semantics/suggestion.h"
#include "src/core/attr_synth/ordered/compile.h"
#include "src/core/attr_synth/synthesize.h"
#include "src/inference/evaluate.h"

namespace tepl::core::semantics {
namespace {
struct Predicate {
  S symbolic;
  TypedExprPtr expression;
};

S list(const auto& values) {
  std::vector<S> items;
  for (auto v : values) items.push_back(integer(v));
  return make(Symbol::Kind::List, {}, 0, 0, std::move(items));
}

// A concrete witness is used only to rule out vacuous recommendations. It is
// never evidence for universal correctness, which requires fresh verification.
// This deliberately bounded search may decline to recommend a valid condition.
bool hasWitness(const Plan& plan, const Rule& rule) {
  Plan source_plan = plan;
  source_plan.rhs = plan.lhs;
  auto compiled = ordered::compile(source_plan, inference::Mode::Validate);
  if (!compiled.program) return false;
  const auto& code = compiled.program->source;
  std::uint64_t random = 0x6a09e667f3bcc909ULL;
  auto choose = [&](std::size_t n) {
    random ^= random << 13;
    random ^= random >> 7;
    random ^= random << 17;
    return static_cast<std::size_t>(random % n);
  };
  for (std::size_t attempt = 0; attempt < 96; ++attempt) {
    inference::Metadata metadata;
    std::vector<std::size_t> ranks;
    for (const auto& n : code) {
      if (n.capture) {
        auto rank = attempt < 5 ? attempt : choose(5);
        for (const auto& constraint : rule.constraints) {
          if (constraint.capture != CaptureId{*n.capture}) continue;
          rank = 0;
          for (const auto& e : constraint.shape)
            rank += e.kind == ShapeElement::Kind::kSequence ? 0 : 1;
        }
        metadata.shapes[*n.capture] = std::vector<std::uint64_t>(rank, 2);
        ranks.push_back(rank);
        continue;
      }
      auto left = ranks.at(n.inputs[0]);
      auto& fields = metadata.descriptors[*n.descriptor];
      if (n.kind == inference::Kind::Permutation) {
        if (fields.empty()) {
          fields.resize(1);
          fields[n.fields[0]].resize(left);
          std::iota(fields[n.fields[0]].begin(), fields[n.fields[0]].end(), 0);
          if (attempt > 4 && choose(2))
            std::reverse(fields[n.fields[0]].begin(),
                         fields[n.fields[0]].end());
        }
        ranks.push_back(left);
      } else {
        auto right = ranks.at(n.inputs[1]);
        if (fields.empty()) {
          fields.resize(4);
          auto common = std::min(left, right);
          auto batch = attempt < 5 ? std::size_t{0} : choose(common + 1);
          auto contract =
              attempt < 5 ? std::size_t{0} : choose(common - batch + 1);
          for (std::size_t i = 0; i < batch; ++i) {
            fields[n.fields[0]].push_back(i);
            fields[n.fields[1]].push_back(i);
          }
          for (std::size_t i = 0; i < contract; ++i) {
            fields[n.fields[2]].push_back(left - 1 - i);
            fields[n.fields[3]].push_back(batch + i);
          }
        }
        auto removed =
            fields[n.fields[0]].size() + 2 * fields[n.fields[2]].size();
        if (removed > left + right) break;
        ranks.push_back(left + right - removed);
      }
    }
    if (ranks.size() != code.size()) continue;
    auto analyzed =
        inference::analyze(*compiled.program, metadata, {10000, 128});
    if (analyzed.status != inference::Status::Analyzed) continue;
    for (const auto* dtype : {"dtype_f32", "dtype_f64", "dtype_i32"}) {
      std::function<S(const S&)> substitute = [&](const S& value) -> S {
        if (value->kind == Symbol::Kind::Field)
          return list(metadata.descriptors.at(value->owner).at(value->index));
        if (value->kind == Symbol::Kind::Shape)
          return list(metadata.shapes.at(value->owner));
        if (value->kind == Symbol::Kind::DType) return call(dtype, {});
        std::vector<S> args;
        for (const auto& a : value->args) args.push_back(substitute(a));
        return make(value->kind, value->name, value->owner, value->index,
                    std::move(args));
      };
      try {
        Plan concrete;
        Context check{concrete, 10000};
        bool matches = true;
        std::vector<S> dtypes;
        Lowerer dtype_check{check};
        for (const auto& instruction : code) {
          if (instruction.capture) {
            dtypes.push_back(call(dtype, {}));
            continue;
          }
          std::vector<TensorMeaning> inputs;
          for (auto child : instruction.inputs) {
            TensorMeaning input;
            input.dtype = dtypes.at(child);
            inputs.push_back(std::move(input));
          }
          auto begin = dtype_check.obligations.size();
          dtypes.push_back(
              dtype_check.dtype(plan.nodes.at(instruction.node), inputs));
          for (std::size_t i = begin; i < dtype_check.obligations.size(); ++i) {
            auto condition = substitute(dtype_check.obligations[i]);
            check.checkMetadata(condition);
            if (!check.entails(condition)) {
              matches = false;
              break;
            }
          }
          if (!matches) break;
        }
        if (!matches) continue;
        for (const auto& assumption : plan.assumptions) {
          auto value = substitute(assumption);
          check.checkMetadata(value);
          if (!check.entails(value)) {
            matches = false;
            break;
          }
        }
        if (matches) return true;
      } catch (const ProofFailure&) {
        // Unsupported witness evaluation gives no recommendation.
      } catch (const std::out_of_range&) {
        // A condition references metadata outside this witness fragment.
      }
    }
  }
  return false;
}
}  // namespace

std::string suggestRestrictions(const Plan& input, const Rule& rule,
                                ProgramTables& tables) {
  if (!input.algebraic) return {};
  Plan plan = input;
  std::set<std::string> names;
  for (const auto& c : rule.captures) names.insert(c.name);
  for (const auto& l : rule.locals) names.insert(l.name);
  for (const auto& n : plan.nodes)
    if (!n.label.empty()) names.insert(n.label);
  for (std::size_t i = 0; i < plan.nodes.size(); ++i) {
    auto& n = plan.nodes[i];
    if (n.capture || n.literal || !n.label.empty()) continue;
    std::string base = i == plan.lhs   ? "source"
                       : i == plan.rhs ? "result"
                                       : "node";
    auto name = base;
    for (std::size_t j = 2; !names.insert(name).second; ++j)
      name = base + "_" + std::to_string(j);
    n.label = name;
  }
  auto type = [&](TypeKind kind) {
    for (std::size_t i = 0; i < tables.types.size(); ++i)
      if (tables.types[i].kind == kind) return TypeId{i};
    tables.types.push_back(core::Type{kind});
    return TypeId{tables.types.size() - 1};
  };
  try {
    Context source{plan, 100000};
    source.sourceScope(plan.lhs);
    Lowerer lowerer{source};
    lowerer.lower(plan.lhs);
    std::vector<Predicate> candidates;
    std::set<Key> seen;
    auto add = [&](S a, S b, TypedExprPtr left, TypedExprPtr right) {
      auto condition = call("==", {a, b});
      if (source.entails(condition) || !seen.insert(key(condition)).second)
        return;
      candidates.push_back(
          {condition, std::make_shared<TypedExpr>(TypedExpr{
                          rule.origin, type(TypeKind::kBool),
                          BinaryExpr{BinaryOp::kEqual, left, right}})});
    };
    auto fieldExpr = [&](const Instance& n, std::size_t f) -> TypedExprPtr {
      auto schema =
          *tables.types.at(rule.descriptors.at(n.descriptor->value).type.value)
               .schema;
      return std::make_shared<TypedExpr>(
          TypedExpr{rule.origin, type(TypeKind::kIndexList),
                    DescriptorFieldRef{*n.descriptor, schema, f}});
    };
    // Prefer group relationships over individual axis-number restrictions.
    for (auto id : source.source().nodes) {
      const auto& n = plan.nodes[id];
      if (!n.definition || !n.descriptor) continue;
      auto contract = axis::recognizeOrdered(*n.definition);
      if (!contract) continue;
      if (contract->kind == axis::OrderedContract::Kind::Permutation) {
        auto f = contract->permutation;
        auto value = fieldExpr(n, f);
        auto len = std::make_shared<TypedExpr>(
            TypedExpr{rule.origin, type(TypeKind::kIndex),
                      BuiltinCall{builtins::Builtin::kLen, {value}}});
        auto identity = std::make_shared<TypedExpr>(
            TypedExpr{rule.origin, type(TypeKind::kIndexList),
                      BuiltinCall{builtins::Builtin::kRange, {len}}});
        add(source.field(n, f),
            call("range", {call("len", {source.field(n, f)})}), value,
            identity);
        continue;
      }
      for (std::size_t operand = 0; operand < 2; ++operand) {
        auto child = n.inputs[operand];
        const auto& producer = plan.nodes[child];
        if (!producer.definition) continue;
        auto producer_contract = axis::recognizeOrdered(*producer.definition);
        if (!producer_contract) continue;
        auto batch =
            operand == 0 ? contract->left_batch : contract->right_batch;
        auto contraction =
            operand == 0 ? contract->left_contract : contract->right_contract;
        for (auto group : {producer_contract->batch_group,
                           producer_contract->left_free_group,
                           producer_contract->right_free_group}) {
          if (producer_contract->kind !=
              axis::OrderedContract::Kind::Contraction)
            continue;
          auto positions = source.positions(
              source.source().tensor_axes.at(child),
              source.source().tensor_groups.at(child).at(group));
          auto expr = std::make_shared<TypedExpr>(TypedExpr{
              rule.origin, type(TypeKind::kIndexList),
              OutputAxisGroupRef{producer.label,
                                 producer.definition->groups[group].name}});
          add(source.field(n, batch), positions, fieldExpr(n, batch), expr);
          add(source.field(n, contraction), positions,
              fieldExpr(n, contraction), expr);
        }
      }
    }
    if (candidates.size() > 24) candidates.resize(24);
    std::size_t attempts = 0;
    auto verify = [&](const std::vector<std::size_t>& indices) {
      if (++attempts > 96) return false;
      auto trial = plan;
      for (auto i : indices)
        trial.assumptions.push_back(candidates[i].symbolic);
      auto proof = synthesize(trial, 1000000);
      return proof.status == ProofStatus::Proved && hasWitness(trial, rule);
    };
    std::vector<std::size_t> selected;
    for (std::size_t i = 0; i < candidates.size(); ++i)
      if (verify({i})) {
        selected = {i};
        break;
      }
    if (selected.empty())
      for (std::size_t i = 0; i < candidates.size() && selected.empty(); ++i)
        for (std::size_t j = i + 1; j < candidates.size() && attempts < 96; ++j)
          if (verify({i, j})) {
            selected = {i, j};
            break;
          }
    if (selected.empty()) return {};
    // Fresh synthesis and verification on every deletion; prior assumptions
    // and constructor choices are never reused as evidence.
    for (std::size_t i = 0; i < selected.size();) {
      auto smaller = selected;
      smaller.erase(smaller.begin() + i);
      if (!smaller.empty() && verify(smaller))
        selected = std::move(smaller);
      else
        ++i;
    }
    std::vector<TypedExprPtr> expressions;
    for (auto i : selected) expressions.push_back(candidates[i].expression);
    return restrictedRule(plan, rule, tables, expressions);
  } catch (const ProofFailure&) {
    return {};
  } catch (const std::invalid_argument&) {
    return {};
  }
}
}  // namespace tepl::core::semantics
