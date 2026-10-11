#include "src/core/inference/check.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

#include "src/core/analysis_context.h"
#include "src/core/inference/constraints.h"
#include "src/core/semantics/ordered.h"
#include "src/core/semantics/solver.h"

namespace tepl::core::inference {
namespace {
struct Builder {
  detail::AnalysisContext& context;
  Rule& rule;
  Program result;
  SemanticPlan plan;
  std::map<std::size_t, std::size_t> captures;
  std::map<std::size_t, const Pattern*> bindings;
  std::set<std::size_t> active;
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> outputs;
  [[noreturn]] void fail(const std::string& s) {
    throw std::runtime_error("semantic inference: " + s);
  }
  const Hole* hole(DescriptorId d) {
    for (const auto& h : rule.attribute_holes)
      if (h.descriptor == d) return &h;
    return nullptr;
  }
  std::size_t capture(CaptureId id) {
    if (captures.contains(id.value)) return captures.at(id.value);
    if (bindings.contains(id.value)) {
      if (!active.insert(id.value).second) fail("cyclic intermediate binding");
      auto node = source(*bindings.at(id.value));
      active.erase(id.value);
      captures[id.value] = node;
      return node;
    }
    SemanticNode n;
    n.capture = plan.captures.size();
    plan.captures.push_back(id);
    auto index = plan.nodes.size();
    plan.nodes.push_back(n);
    captures[id.value] = index;
    return index;
  }
  void dtype(const Operation& o) {
    if (!o.dtype) fail("unknown: forwarding numeric dtype contract required");
    const auto& p = *o.dtype;
    auto parameter = [&](const metadata::ExprPtr& e) {
      auto ref = std::get_if<metadata::SymbolRef>(&e->value);
      return ref && std::any_of(p.parameters.begin(), p.parameters.end(),
                                [&](const auto& x) {
                                  return x.symbol == ref->symbol && !x.variadic;
                                });
    };
    if (!parameter(p.result.value))
      fail("unknown: dtype must forward an operand");
    for (const auto& s : p.statements) {
      auto a = std::get_if<metadata::Assert>(&s.value);
      if (!a) fail("unknown: dtype locals are outside this fragment");
      auto b = std::get_if<metadata::Binary>(&a->condition->value);
      if (b && b->op == BinaryOp::kEqual && parameter(b->lhs) &&
          parameter(b->rhs))
        continue;
      auto c = std::get_if<metadata::Call>(&a->condition->value);
      if (c && c->builtin == builtins::Builtin::kIsNumeric &&
          c->arguments.size() == 1 && parameter(c->arguments[0]))
        continue;
      fail("unknown: unsupported dtype assertion");
    }
  }
  std::size_t operation(OpId id, std::optional<DescriptorId> descriptor,
                        std::vector<std::size_t> inputs) {
    const auto& o = context.output.operations.at(id.value);
    if (!o.semantics) fail("missing semantics for operation '" + o.name + "'");
    dtype(o);
    if (!o.shape)
      fail("shape contract required for operation '" + o.name + "'");
    if (inputs.size() != o.operands.size())
      fail("variadic operands unsupported");
    SemanticNode n;
    n.inputs = std::move(inputs);
    std::set<std::size_t> opaque_fields;
    auto emit = [&](std::string op, std::vector<std::size_t> args = {}) {
      n.expressions.push_back({std::move(op), std::move(args)});
      return n.expressions.size() - 1;
    };
    const Hole* h = descriptor ? hole(*descriptor) : nullptr;
    if (o.attributes) {
      if (!descriptor) fail("attribute descriptor required");
      auto& schema = context.output.attribute_schemas.at(o.attributes->value);
      n.fields.resize(schema.fields.size());
      Assignment assignment{*descriptor, *o.attributes};
      for (std::size_t f = 0; f < schema.fields.size(); ++f) {
        auto& field = schema.fields[f];
        const bool opaque =
            field.type != "index" || field.list_depth != 1 || field.optional;
        if (opaque) opaque_fields.insert(f);
        auto key = std::pair(descriptor->value, f);
        if (!outputs.contains(key)) outputs[key] = outputs.size();
        n.fields[f].output = outputs.at(key);
        n.fields[f].bound = static_cast<std::size_t>(-1);
        if (opaque) {
          if (h && !h->overrides.at(f) && !field.empty_default)
            fail("unknown: pin non-index[] field '" + field.name +
                 "' explicitly");
          auto empty =
              std::make_shared<Value>(Value{Value::Kind::kLiteral, "list"});
          n.fields[f].known = -2 - int(result.pinned.size());
          result.pinned.push_back(empty);
          if (h)
            assignment.fields.push_back(h->overrides.at(f) ? h->overrides.at(f)
                                                           : empty);
          continue;
        }
        if (!h) {
          n.fields[f].known = plan.known.size();
          plan.known.push_back({*descriptor, *o.attributes, f});
        } else if (h->overrides.at(f)) {
          n.fields[f].known = -2 - int(result.pinned.size());
          result.pinned.push_back(h->overrides[f]);
        }
        if (h) {
          auto v =
              std::make_shared<Value>(Value{Value::Kind::kSemanticWitness});
          v->field = n.fields[f].output;
          assignment.fields.push_back(v);
        }
      }
      if (h && std::none_of(
                   result.assignments.begin(), result.assignments.end(),
                   [&](const auto& a) { return a.descriptor == *descriptor; }))
        result.assignments.push_back(std::move(assignment));
    }
    std::map<const semantics::Expr*, std::size_t> memo;
    auto lower = [&](auto&& self, const semantics::Expr& e) -> std::size_t {
      if (memo.contains(&e)) return memo.at(&e);
      std::vector<std::size_t> args;
      for (auto& a : e.arguments) args.push_back(self(self, *a));
      return memo[&e] = emit(e.op, args);
    };
    n.root = lower(lower, *o.semantics->result);
    // Lower the existing resolved shape IR to the same primitive bytecode.
    std::map<std::size_t, std::size_t> symbols;
    for (const auto& p : o.shape->parameters) {
      if (p.variadic) fail("variadic shape parameters unsupported");
      symbols[p.symbol.value] =
          emit("shape", {emit("t" + std::to_string(p.operand))});
    }
    auto metadata = [&](auto&& self, const metadata::Expr& e) -> std::size_t {
      if (auto s = std::get_if<metadata::SymbolRef>(&e.value))
        return symbols.at(s->symbol.value);
      if (auto a = std::get_if<metadata::AttributeRef>(&e.value))
        return emit("a" + std::to_string(a->field));
      if (auto i = std::get_if<metadata::Integer>(&e.value))
        return emit(i->spelling);
      if (auto b = std::get_if<bool>(&e.value))
        return emit(*b ? "true" : "false");
      if (auto b = std::get_if<metadata::Binary>(&e.value))
        return emit(std::string(spelling(b->op)),
                    {self(self, *b->lhs), self(self, *b->rhs)});
      if (auto u = std::get_if<metadata::Unary>(&e.value))
        return emit("unary:" + std::string(spelling(u->op)),
                    {self(self, *u->operand)});
      if (auto c = std::get_if<metadata::Call>(&e.value)) {
        std::vector<std::size_t> args;
        for (auto& a : c->arguments) args.push_back(self(self, *a));
        return emit(std::string(builtins::builtinName(c->builtin)), args);
      }
      if (auto l = std::get_if<metadata::List>(&e.value)) {
        std::vector<std::size_t> args;
        for (auto& a : l->elements) args.push_back(self(self, *a));
        return emit("list", args);
      }
      if (auto i = std::get_if<metadata::Index>(&e.value))
        return emit("index", {self(self, *i->value), self(self, *i->index)});
      if (auto c = std::get_if<metadata::Conditional>(&e.value))
        return emit("if",
                    {self(self, *c->condition), self(self, *c->then_value),
                     self(self, *c->else_value)});
      fail("unknown: shape comprehensions are outside the constraint fragment");
    };
    for (const auto& s : o.shape->statements) {
      if (auto l = std::get_if<metadata::Let>(&s.value)) {
        auto value = metadata(metadata, *l->value);
        symbols[l->symbol.value] = value;
        // A let can impose legality even when its result is unused.
        n.assertions.push_back(emit("==", {value, value}));
      } else
        n.assertions.push_back(
            metadata(metadata, *std::get<metadata::Assert>(s.value).condition));
    }
    n.shape = metadata(metadata, *o.shape->result.value);
    for (auto& e : n.expressions)
      if (e.args.empty() && e.op.starts_with('a') && e.op.size() > 1 &&
          opaque_fields.contains(std::stoul(e.op.substr(1))))
        fail(
            "unknown: a non-index[] attribute participates in shape or "
            "semantics");
    // Domain discovery is syntax-directed over assertions, never op names.
    for (auto a : n.assertions) {
      const auto& e = n.expressions[a];
      if (e.op == "is_valid_axis_list" && e.args.size() == 2) {
        const auto& v = n.expressions[e.args[0]];
        if (v.op.starts_with('a') && v.args.empty()) {
          auto f = std::stoul(v.op.substr(1));
          n.fields.at(f).bound = e.args[1];
        }
      }
    }
    for (const auto& f : n.fields)
      if (f.known == -1 && f.bound == static_cast<std::size_t>(-1))
        fail(
            "unknown: auto index[] requires an explicit is_valid_axis_list "
            "bound in shape");
    static const std::set<std::string> primitives = {
        "shape",       "rank",    "len",    "range",
        "gather",      "exclude", "concat", "take",
        "slice",       "scatter", "list",   "index",
        "index_space", "read",    "+",      "-",
        "*",           "/",       "%",      "==",
        "!=",          "<",       "<=",     ">",
        ">=",          "&&",      "||",     "unary:+",
        "unary:-",     "unary:!", "if",     "is_valid_axis_list",
        "is_disjoint", "true",    "false"};
    for (auto& e : n.expressions) {
      bool name =
          e.args.empty() && !e.op.empty() &&
          (e.op[0] == 't' || e.op[0] == 'a' || e.op == "o" || e.op[0] == 'k' ||
           e.op.find_first_not_of("0123456789") == std::string::npos);
      if (!name && !primitives.contains(e.op) && !e.op.starts_with("sum:"))
        fail("unknown: unsupported primitive '" + e.op + "'");
    }
    plan.nodes.push_back(std::move(n));
    return plan.nodes.size() - 1;
  }
  std::size_t source(const Pattern& p) {
    if (auto c = std::get_if<CapturePattern>(&p.value))
      return capture(c->capture);
    if (auto b = std::get_if<BindPattern>(&p.value)) return capture(b->capture);
    if (auto o = std::get_if<MatchOperation>(&p.value)) {
      std::vector<std::size_t> inputs;
      for (auto& c : o->operands) inputs.push_back(source(*c));
      return operation(o->operation, o->descriptor, inputs);
    }
    fail("unknown: tensor literals unsupported");
  }
  std::size_t rhs(const BuildExpr& p) {
    if (auto c = std::get_if<CaptureRef>(&p.value)) return capture(c->capture);
    if (auto o = std::get_if<BuildOperation>(&p.value)) {
      std::vector<std::size_t> inputs;
      for (auto& c : o->operands) inputs.push_back(rhs(*c));
      return operation(o->operation, o->descriptor, inputs);
    }
    fail("unknown: tensor literals unsupported");
  }
  Program run() {
    auto collect = [&](auto&& self, const Pattern& p) -> void {
      if (auto b = std::get_if<BindPattern>(&p.value)) {
        bindings[b->capture.value] = b->expression.get();
        self(self, *b->expression);
      }
      if (auto o = std::get_if<MatchOperation>(&p.value))
        for (auto& c : o->operands) self(self, *c);
    };
    collect(collect, *rule.lhs);
    plan.source_root = source(*rule.lhs);
    plan.source_end = plan.nodes.size();
    plan.rhs_root = rhs(*rule.rhs);
    plan.algebraic =
        context.options.semantic_policy == SemanticPolicy::kAlgebraic;
    plan.timeout_ms = context.options.solver_timeout_ms;
    plan.constraints = compileConstraints(plan);
    result.selection = "deterministic-witness";
    result.coverage = "match-guarded";
    result.proof = {
        "checked ordered construction; otherwise compiled semantic constraints "
        "-> Z3 -> exact verification, with unrestricted metadata-search fallback",
        "invalid means a checked ordering obstruction or exhaustive solver "
        "exclusion at fixed metadata; timeouts are unknown",
        "universal arbitrary-rank coverage is not claimed",
        "algebraic uses exact rational arithmetic; strict compares expanded "
        "evaluation traces"};
    std::size_t certified = 0, operations = 0;
    for (auto& n : plan.nodes) {
      if (n.capture >= 0) continue;
      ++operations;
      certified += bool(semantics::extractOrderedContract(n));
    }
    result.proof.push_back(
        "compiled semantic constraints: " +
        std::string(plan.constraints.available ? "sufficient; " : "fallback; ") +
        plan.constraints.reason);
    result.proof.push_back(
        "ordered semantic contracts: " + std::to_string(certified) + "/" +
        std::to_string(operations) +
        "; eligible algebraic matches use structural inference");
    result.semantic = std::move(plan);
    return std::move(result);
  }
};
bool completeSemantics(detail::AnalysisContext& c, const Rule& rule) {
  bool complete = true;
  auto lhs = [&](auto&& self, const Pattern& p) -> void {
    if (auto b = std::get_if<BindPattern>(&p.value)) self(self, *b->expression);
    if (auto o = std::get_if<MatchOperation>(&p.value)) {
      complete &= bool(c.output.operations[o->operation.value].semantics);
      for (auto& a : o->operands) self(self, *a);
    }
  };
  auto rhs = [&](auto&& self, const BuildExpr& p) -> void {
    if (auto o = std::get_if<BuildOperation>(&p.value)) {
      complete &= bool(c.output.operations[o->operation.value].semantics);
      for (auto& a : o->operands) self(self, *a);
    }
  };
  lhs(lhs, *rule.lhs);
  rhs(rhs, *rule.rhs);
  return complete;
}
}  // namespace
void infer(detail::AnalysisContext& context, Rule& rule) {
  const bool automatic = !rule.attribute_holes.empty();
  if (!automatic && !completeSemantics(context, rule)) return;
  try {
    auto result = Builder{context, rule}.run();
    // An exact scalar counterexample is useful at compile time, without a
    // finite-rank theorem. Only unconditional, unconstrained, attr-free rules
    // qualify; unknown guards must never be ignored.
    if (rule.conditions.empty() && rule.derivations.empty() &&
        rule.constraints.empty() && result.semantic->known.empty() &&
        result.pinned.empty()) {
      bool attr_free = std::all_of(result.semantic->nodes.begin(),
                                   result.semantic->nodes.end(),
                                   [](auto& n) { return n.fields.empty(); });
      if (attr_free) {
        std::vector<semantics::List> shapes(result.semantic->captures.size());
        auto probe = *result.semantic;
        probe.algebraic = true;
        auto counter = semantics::solve(probe, shapes, {}, {});
        if (counter.status == semantics::Status::Invalid) {
          context.report(rule.origin,
                         "semantic inference invalid: exact scalar "
                         "counterexample to unconditional rewrite");
          return;
        }
      }
    }
    rule.inference = std::move(result);
  } catch (const std::runtime_error& e) {
    if (automatic) context.report(rule.origin, e.what());
    // Explicit rules outside the proof fragment retain ordinary TEPL behavior.
  }
}
}  // namespace tepl::core::inference
