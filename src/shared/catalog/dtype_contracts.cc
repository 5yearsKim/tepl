#include "src/shared/catalog/dtype_contracts.h"

#include <algorithm>
#include <set>
namespace tepl::core::completed {
bool coreTypes(
    const inference::Program& p,
    const std::function<const metadata::Program*(std::size_t)>& lookup) {
  auto numeric = [](const auto& code) {
    return std::any_of(code.begin(), code.end(), [](const auto& n) {
      return n.kind != inference::Kind::Input &&
             n.kind != inference::Kind::Permutation &&
             n.kind != inference::Kind::Reverse &&
             n.kind != inference::Kind::Broadcast &&
             n.kind != inference::Kind::UnitRemoval;
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
      if ((n.kind == inference::Kind::Permutation ||
           n.kind == inference::Kind::Reverse ||
           n.kind == inference::Kind::Broadcast ||
           n.kind == inference::Kind::UnitRemoval) &&
          !d->statements.empty())
        return false;
      bool equal = ids.size() == 1,
           numeric_type = n.kind == inference::Kind::Permutation ||
                          n.kind == inference::Kind::Reverse ||
                          n.kind == inference::Kind::Broadcast ||
                          n.kind == inference::Kind::UnitRemoval;
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
}  // namespace tepl::core::completed
