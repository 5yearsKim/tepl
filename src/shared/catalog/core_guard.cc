#include "src/shared/catalog/core_guard.h"

#include <algorithm>
#include <charconv>
#include <functional>

#include "src/inference/certificate.h"
namespace tepl::core::completed {
namespace {
bool number(const TypedExpr& e, std::int64_t& n) {
  auto p = std::get_if<NumericConstant>(&e.value);
  if (!p || p->decimal) return false;
  auto r = std::from_chars(p->spelling.data(),
                           p->spelling.data() + p->spelling.size(), n);
  return r.ec == std::errc{} &&
         r.ptr == p->spelling.data() + p->spelling.size();
}
}  // namespace
bool coreTypes(
    const inference::Program& p,
    const std::function<const metadata::Program*(std::size_t)>& lookup) {
  auto numeric = [](const auto& code) {
    return std::any_of(code.begin(), code.end(), [](const auto& n) {
      return n.kind != inference::Kind::Input &&
             n.kind != inference::Kind::Permutation && n.kind != inference::Kind::Reverse && n.kind != inference::Kind::Broadcast && n.kind != inference::Kind::UnitRemoval;
    });
  };
  if (numeric(p.replacement) && !numeric(p.source)) return false;
  auto check = [&](const auto& code) {
    for (const auto& n : code) {
      if (n.kind == inference::Kind::Input) continue;
      if (!n.operation) return false;
      auto d = lookup(*n.operation);
      if (!d || d->kind != metadata::ProgramKind::DType ||
          d->parameters.size() != n.inputs.size() || !d->result.value)
        return false;
      std::vector<metadata::SymbolId> ids(n.inputs.size());
      std::set<std::size_t> seen, symbols;
      for (const auto& a : d->parameters) {
        if (a.variadic || a.operand >= ids.size() ||
            !seen.insert(a.operand).second ||
            !symbols.insert(a.symbol.value).second)
          return false;
        ids[a.operand] = a.symbol;
      }
      auto ref = [](const metadata::ExprPtr& e, metadata::SymbolId id) {
        auto s = e ? std::get_if<metadata::SymbolRef>(&e->value) : nullptr;
        return s && s->symbol == id;
      };
      if (!ref(d->result.value, ids[0])) return false;
      if ((n.kind == inference::Kind::Permutation || n.kind == inference::Kind::Reverse || n.kind == inference::Kind::Broadcast || n.kind == inference::Kind::UnitRemoval) && !d->statements.empty())
        return false;
      bool equal = ids.size() == 1,
           numeric_type = n.kind == inference::Kind::Permutation || n.kind == inference::Kind::Reverse || n.kind == inference::Kind::Broadcast || n.kind == inference::Kind::UnitRemoval;
      for (const auto& statement : d->statements) {
        auto a = std::get_if<metadata::Assert>(&statement.value);
        if (!a || !a->condition) return false;
        if (auto b = std::get_if<metadata::Binary>(&a->condition->value);
            b && b->op == BinaryOp::kEqual && ids.size() == 2 &&
            ((ref(b->lhs, ids[0]) && ref(b->rhs, ids[1])) ||
             (ref(b->lhs, ids[1]) && ref(b->rhs, ids[0])))) {
          equal = true;
          continue;
        }
        if (auto c = std::get_if<metadata::Call>(&a->condition->value);
            c && c->builtin == builtins::Builtin::kIsNumeric &&
            c->arguments.size() == 1 && ref(c->arguments[0], ids[0])) {
          numeric_type = true;
          continue;
        }
        return false;
      }
      if (!equal || !numeric_type) return false;
    }
    return true;
  };
  return check(p.source) && check(p.replacement);
}
bool coreGuard(const RuleData& r) {
  if (!r.core || !r.derivations.empty()) return false;
  inference::LayoutFacts f;
  std::map<std::size_t, TypedExprPtr> locals;
  auto rank = [&](std::size_t id, std::size_t n) {
    auto [it, ok] = f.ranks.emplace(id, n);
    f.inconsistent |= !ok && it->second != n;
  };
  auto field = [&](std::size_t id, std::size_t pos, const inference::List& xs) {
    auto [it, ok] = f.fields.emplace(std::pair{id, pos}, xs);
    f.inconsistent |= !ok && it->second != xs;
  };
  for (const auto& c : r.constraints)
    if (std::none_of(c.shape.begin(), c.shape.end(), [](const auto& e) {
          return e.kind == ShapeElement::Kind::kSequence;
        }))
      rank(c.capture.value, c.shape.size());
  std::map<std::size_t, inference::MetadataTerm> dimension_terms;
  for (const auto& c : r.constraints) {
    const bool full = c.shape.size() == 1 && c.shape[0].kind == ShapeElement::Kind::kSequence;
    const bool fixed = std::none_of(c.shape.begin(), c.shape.end(), [](const auto& x){return x.kind==ShapeElement::Kind::kSequence;});
    for (std::size_t i=0; i<c.shape.size(); ++i) if (c.shape[i].symbol && (full || fixed)) {
      auto term = full ? inference::termShape(c.capture.value) : inference::termCall("index", {inference::termShape(c.capture.value), inference::termNumber(i)});
      auto [it, fresh] = dimension_terms.emplace(c.shape[i].symbol->value, term);
      if (!fresh) f.equalities.emplace_back(it->second, term);
    }
  }
  std::function<const TypedExpr&(const TypedExpr&)> resolve =
      [&](const auto& e) -> const TypedExpr& {
    if (auto l = std::get_if<LocalRef>(&e.value)) {
      auto it = locals.find(l->local.value);
      if (it != locals.end()) return resolve(*it->second);
    }
    return e;
  };
  auto shapeCaptures = [&](const TypedExpr& e) {
    std::vector<std::size_t> caps;
    if (auto x = std::get_if<SymbolicExpr>(&e.value);
        x && x->value->kind == Symbol::Kind::Shape)
      caps.push_back(x->value->owner);
    if (auto d = std::get_if<DimensionRef>(&e.value))
      for (const auto& c : r.constraints)
        if (c.shape.size() == 1 &&
            c.shape[0].kind == ShapeElement::Kind::kSequence &&
            c.shape[0].symbol == d->dimension)
          caps.push_back(c.capture.value);
    return caps;
  };
  using Term = inference::MetadataTerm;
  auto supported = [](const std::string& n) {
    return n == "list" || n == "len" || n == "range" || n == "concat" || n == "gather" || n == "index" || n == "+";
  };
  std::function<std::optional<Term>(const SymbolPtr&)> symbolic = [&](const auto& x) -> std::optional<Term> {
    if (x->kind == Symbol::Kind::Shape) return inference::termShape(x->owner);
    if (x->kind == Symbol::Kind::Field) return inference::termField(x->owner, x->index);
    if (x->kind == Symbol::Kind::Integer) {
      std::int64_t n; auto z = std::from_chars(x->name.data(), x->name.data()+x->name.size(), n);
      if (z.ec != std::errc{} || z.ptr != x->name.data()+x->name.size()) return {};
      return inference::termNumber(n);
    }
    if (x->kind != Symbol::Kind::List && x->kind != Symbol::Kind::Call) return {};
    auto op = x->kind == Symbol::Kind::List ? "list" : x->name;
    if (!supported(op)) return {};
    Term out{op,{}};
    for (const auto& a : x->args) { auto t = symbolic(a); if (!t) return {}; out.args.push_back(*t); }
    return out;
  };
  std::function<std::optional<Term>(const TypedExpr&)> metadata = [&](const auto& input) -> std::optional<Term> {
    const auto& e = resolve(input);
    if (auto x = std::get_if<SymbolicExpr>(&e.value)) return symbolic(x->value);
    if (auto x = std::get_if<DescriptorFieldRef>(&e.value)) return inference::termField(x->descriptor.value, x->field);
    if (auto x = std::get_if<DimensionRef>(&e.value)) {
      for (const auto& c : r.constraints)
        if (c.shape.size() == 1 && c.shape[0].kind == ShapeElement::Kind::kSequence && c.shape[0].symbol == x->dimension)
          return inference::termShape(c.capture.value);
      for (const auto& c : r.constraints)
        for (std::size_t i=0; i<c.shape.size(); ++i)
          if (c.shape[i].symbol == x->dimension && c.shape[i].kind != ShapeElement::Kind::kSequence &&
              std::none_of(c.shape.begin(), c.shape.end(), [](const auto& s){return s.kind==ShapeElement::Kind::kSequence;}))
            return inference::termCall("index", {inference::termShape(c.capture.value), inference::termNumber(i)});
      return {};
    }
    std::int64_t n; if (number(e, n)) return inference::termNumber(n);
    std::vector<TypedExprPtr> children; std::string op;
    if (auto x = std::get_if<BuiltinCall>(&e.value)) { op = builtins::builtinName(x->builtin); children = x->arguments; }
    else if (auto x = std::get_if<BinaryExpr>(&e.value); x && x->op == BinaryOp::kAdd) { op = "+"; children = {x->lhs, x->rhs}; }
    else if (auto x = std::get_if<IndexExpr>(&e.value)) { op = "index"; children = {x->value, x->index}; }
    else if (auto x = std::get_if<ListExpr>(&e.value)) { op = "list"; children = x->elements; }
    else return {};
    if (!supported(op)) return {};
    Term out{op,{}};
    for (const auto& a : children) { auto t = metadata(*a); if (!t) return {}; out.args.push_back(*t); }
    return out;
  };
  auto fact = [&](const TypedExpr& left, const TypedExpr& right) {
    const auto& a = resolve(left);
    const auto& b = resolve(right);
    if (auto x = metadata(a)) if (auto y = metadata(b)) f.equalities.emplace_back(*x, *y);
    if (auto lhs = std::get_if<DescriptorFieldRef>(&a.value))
      if (auto sym = std::get_if<SymbolicExpr>(&b.value)) {
        const auto& v = sym->value;
        if (v->kind == Symbol::Kind::Call && v->name == "range" &&
            v->args.size() == 1) {
          auto n = v->args[0];
          if (n->kind == Symbol::Kind::Call && n->name == "len" &&
              n->args.size() == 1 && n->args[0]->kind == Symbol::Kind::Field) {
            auto x = n->args[0];
            f.batch_prefixes.insert(
                {lhs->descriptor.value, lhs->field, x->owner, x->index});
          }
        }
      }

    if (auto x = std::get_if<DescriptorFieldRef>(&a.value))
      if (auto list = std::get_if<ListExpr>(&b.value)) {
        auto [length, fresh] = f.lengths.emplace(std::pair{x->descriptor.value, x->field}, list->elements.size());
        f.inconsistent |= !fresh && length->second != list->elements.size();
        inference::List xs;
        for (const auto& v : list->elements) {
          std::int64_t n;
          if (!number(resolve(*v), n)) return;
          xs.push_back(n);
        }
        field(x->descriptor.value, x->field, xs);
      }
    auto length = std::get_if<BuiltinCall>(&a.value);
    if (length && length->builtin == builtins::Builtin::kLen &&
        length->arguments.size() == 1) {
      const auto& arg = resolve(*length->arguments[0]);
      std::int64_t n;
      if (auto x = std::get_if<DescriptorFieldRef>(&arg.value);
          x && number(b, n) && n >= 0) {
        auto [it, ok] =
            f.lengths.emplace(std::pair{x->descriptor.value, x->field}, n);
        f.inconsistent |= !ok && it->second != static_cast<std::size_t>(n);
      }
      auto shapes = shapeCaptures(arg);
      auto sum = std::get_if<BinaryExpr>(&b.value);
      if (!shapes.empty() && sum && sum->op == BinaryOp::kAdd)
        for (int i = 0; i < 2; ++i) {
          const auto& le = resolve(*(i ? sum->rhs : sum->lhs));
          const auto& ne = resolve(*(i ? sum->lhs : sum->rhs));
          auto len = std::get_if<BuiltinCall>(&le.value);
          if (number(ne, n) && n == 2 && len &&
              len->builtin == builtins::Builtin::kLen &&
              len->arguments.size() == 1)
            if (auto x = std::get_if<DescriptorFieldRef>(
                    &resolve(*len->arguments[0]).value))
              for (auto cap : shapes)
                f.batch_plus_two.insert({cap, x->descriptor.value, x->field});
        }
    }

    std::int64_t n;
    if (length && length->builtin == builtins::Builtin::kLen &&
        length->arguments.size() == 1 && number(b, n) && n >= 0) {
      for (auto cap : shapeCaptures(resolve(*length->arguments[0])))
        rank(cap, n);
    }
  };
  std::function<void(const TypedExpr&)> collect = [&](const TypedExpr& e) {
    const auto& x = resolve(e);
    if (auto l = std::get_if<LocalLet>(&x.value)) {
      locals[l->local.value] = l->value;
      return;
    }
    if (auto b = std::get_if<BinaryExpr>(&x.value)) {
      if (b->op == BinaryOp::kLogicalAnd) {
        collect(*b->lhs);
        collect(*b->rhs);
      } else if (b->op == BinaryOp::kEqual) {
        fact(*b->lhs, *b->rhs);
        fact(*b->rhs, *b->lhs);
      }
    }
  };
  for (const auto& c : r.conditions) collect(*c);
  return inference::certify(r.core->program, f, {5000000, 4096});
}
}  // namespace tepl::core::completed
