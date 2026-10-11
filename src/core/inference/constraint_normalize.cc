#include "src/core/inference/constraint_normalize.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace tepl::core::inference {
namespace {
using ExpressionKey = std::pair<std::string, std::vector<std::size_t>>;
using TermKey = std::pair<std::vector<std::size_t>,
                          std::vector<std::pair<std::size_t, std::size_t>>>;
struct Normalizer {
  ConstraintPlan& plan;
  std::map<ExpressionKey, std::size_t> intern;
  std::size_t work = 0;
  void tick(std::size_t depth) {
    if (++work > 100000 || depth > 256)
      throw std::runtime_error("constraint normalization limit");
  }
  void references(std::size_t id, std::set<std::size_t>& visited,
                  std::set<std::size_t>& used, std::size_t depth = 0) {
    tick(depth);
    if (!visited.insert(id).second) return;
    const auto& e = plan.expressions.at(id);
    if (e.op.starts_with("@bound:")) used.insert(std::stoull(e.op.substr(7)));
    for (auto a : e.args) references(a, visited, used, depth + 1);
  }
  bool staticUnit(std::size_t id, std::size_t depth = 0) {
    tick(depth);
    const auto& e = plan.expressions.at(id);
    if (e.op == "index_space") return staticUnit(e.args.at(0), depth + 1);
    if (e.op != "list") return false;
    return std::all_of(e.args.begin(), e.args.end(), [&](auto a) {
      const auto& dim = plan.expressions.at(a);
      return dim.op == "1" && dim.args.empty();
    });
  }
  std::size_t key(std::size_t id,
                  const std::map<std::size_t, std::size_t>& names,
                  std::map<std::size_t, std::size_t>& cache,
                  std::size_t depth = 0) {
    tick(depth);
    if (cache.contains(id)) return cache.at(id);
    const auto& e = plan.expressions.at(id);
    auto op = e.op;
    if (op.starts_with("@bound:"))
      op = "local:" + std::to_string(names.at(std::stoull(op.substr(7))));
    std::vector<std::size_t> args;
    for (auto a : e.args) args.push_back(key(a, names, cache, depth + 1));
    ExpressionKey signature{std::move(op), std::move(args)};
    auto [it, inserted] = intern.emplace(std::move(signature), intern.size());
    return cache[id] = it->second;
  }
  void terms(std::vector<ConstraintTerm>& source) {
    std::map<TermKey, std::size_t> merged;
    std::vector<ConstraintTerm> output;
    for (auto& t : source) {
      std::set<std::size_t> visited, used;
      for (auto& r : t.reads) references(r.indices, visited, used);
      std::erase_if(t.reductions, [&](auto b) {
        return !used.contains(b) && staticUnit(plan.domains.at(b));
      });
      t.unused_reductions.clear();
      std::map<std::size_t, std::size_t> names, cache;
      for (auto b : t.reductions) {
        names.emplace(b, names.size());
        if (!used.contains(b)) t.unused_reductions.push_back(b);
      }
      TermKey signature;
      for (auto b : t.reductions)
        signature.first.push_back(key(plan.domains.at(b), names, cache));
      for (auto& r : t.reads)
        signature.second.emplace_back(r.capture, key(r.indices, names, cache));
      std::sort(signature.second.begin(), signature.second.end());
      std::sort(t.reads.begin(), t.reads.end(), [&](auto a, auto b) {
        return std::pair(a.capture, cache.at(a.indices)) <
               std::pair(b.capture, cache.at(b.indices));
      });
      auto [it, inserted] = merged.emplace(std::move(signature), output.size());
      if (inserted)
        output.push_back(std::move(t));
      else {
        auto& coefficient = output.at(it->second).coefficient;
        auto sum = __int128(coefficient) + t.coefficient;
        if (sum < std::numeric_limits<std::int64_t>::min() ||
            sum > std::numeric_limits<std::int64_t>::max())
          throw std::runtime_error("constraint coefficient limit");
        coefficient = static_cast<std::int64_t>(sum);
      }
    }
    std::erase_if(output, [](const auto& t) { return t.coefficient == 0; });
    source = std::move(output);
  }
};
}  // namespace
void normalizeConstraints(ConstraintPlan& plan) {
  Normalizer normalizer{plan};
  normalizer.terms(plan.lhs);
  normalizer.terms(plan.rhs);
}
}  // namespace tepl::core::inference
