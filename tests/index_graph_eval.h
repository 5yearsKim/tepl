#pragma once

#include <functional>

#include "src/core/semantics/graph.h"
#include "tests/index_constructor_eval.h"

namespace tepl::test {
using namespace core::semantics;
inline void checkGraph(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
// Concrete physical tensors and a graph interpreter. Operation reference
// evaluators stay in their own tests and do not call compiler index functions.
inline void enumerate(const List& sizes,
                      const std::function<void(const List&)>& fn) {
  List at(sizes.size());
  auto loop = [&](auto&& self, std::size_t d) -> void {
    if (d == sizes.size()) {
      fn(at);
      return;
    }
    for (at[d] = 0; at[d] < sizes[d]; ++at[d]) self(self, d + 1);
  };
  loop(loop, 0);
}
struct Dense {
  List shape, values;
  std::int64_t at(const List& p) const {
    checkGraph(p.size() == shape.size(), "read rank");
    std::size_t offset = 0;
    for (std::size_t d = 0; d < p.size(); ++d) {
      checkGraph(p[d] >= 0 && p[d] < shape[d],
                 "calculated read stays in bounds: axis " + std::to_string(d) +
                     " index " + std::to_string(p[d]) + " extent " +
                     std::to_string(shape[d]));
      offset = offset * shape[d] + p[d];
    }
    return values.at(offset);
  }
};
inline Dense data(List shape, int seed) {
  Dense x{shape, {}};
  enumerate(shape, [&](const auto&) {
    x.values.push_back((std::int64_t(x.values.size()) * 3 + seed) % 11 - 5);
  });
  return x;
}
inline Dense add(const Dense& a, const Dense& b) {
  checkGraph(a.shape == b.shape, "addition shape");
  auto out = a;
  for (std::size_t i = 0; i < out.values.size(); ++i)
    out.values[i] += b.values[i];
  return out;
}
inline Dense transpose(const Dense& a, const List& p) {
  Dense out;
  for (auto i : p) out.shape.push_back(a.shape.at(i));
  enumerate(out.shape, [&](const List& at) {
    List original(at.size());
    for (std::size_t i = 0; i < at.size(); ++i) original.at(p[i]) = at[i];
    out.values.push_back(a.at(original));
  });
  return out;
}
struct GraphEvaluator {
  ConstructorEvaluator& metadata;
  const std::vector<Dense>& inputs;
  std::map<Coordinate, std::int64_t> assignment;
  Coordinates axes(const S& x) {
    return std::get<Coordinates>(metadata.eval(x));
  }
  List index(const I& e) {
    if (e->kind == IndexExpr::Kind::Coordinates) {
      List out;
      for (const auto& a : axes(e->axes)) out.push_back(assignment.at(a));
      return out;
    }
    auto out = index(e->args[0]);
    if (e->kind == IndexExpr::Kind::Scale) {
      for (auto& n : out) n *= e->coefficient;
    } else if (e->kind == IndexExpr::Kind::Weighted) {
      auto factors = std::get<List>(metadata.eval(e->axes));
      checkGraph(out.size() == factors.size(), "scale tuple length");
      for (std::size_t i = 0; i < out.size(); ++i) out[i] *= factors[i];
    } else if (e->kind == IndexExpr::Kind::Offset) {
      auto offsets = std::get<List>(metadata.eval(e->axes));
      checkGraph(out.size() == offsets.size(), "offset tuple length");
      for (std::size_t i = 0; i < out.size(); ++i) out[i] += offsets[i];
    } else if (e->kind == IndexExpr::Kind::Select) {
      auto p = std::get<List>(metadata.eval(e->axes));
      List selected;
      for (auto n : p) selected.push_back(out.at(n));
      return selected;
    } else {
      for (std::size_t i = 1; i < e->args.size(); ++i) {
        auto next = index(e->args[i]);
        if (e->kind == IndexExpr::Kind::Concat)
          out.insert(out.end(), next.begin(), next.end());
        else {
          checkGraph(out.size() == next.size(), "paired spatial tuple lengths");
          for (std::size_t j = 0; j < out.size(); ++j) out[j] += next[j];
        }
      }
    }
    return out;
  }
  std::int64_t value(const E& e) {
    if (e->kind == Expr::Kind::Const) return e->constant;
    if (e->kind == Expr::Kind::Read) {
      List p;
      if (e->access)
        p = index(e->access);
      else
        for (const auto& a : axes(e->axes)) p.push_back(assignment.at(a));
      return inputs.at(e->capture).at(p);
    }
    if (e->kind == Expr::Kind::Add)
      return value(e->args[0]) + value(e->args[1]);
    if (e->kind == Expr::Kind::Product)
      return value(e->args[0]) * value(e->args[1]);
    auto coordinates = axes(e->axes);
    auto sizes = std::get<List>(metadata.eval(call("axis_extents", {e->axes})));
    std::int64_t sum = 0;
    enumerate(sizes, [&](const List& at) {
      for (std::size_t i = 0; i < at.size(); ++i) {
        checkGraph(!assignment.contains(coordinates[i]),
                   "fresh reduction binder");
        assignment[coordinates[i]] = at[i];
      }
      sum += value(e->args[0]);
      for (const auto& a : coordinates) assignment.erase(a);
    });
    return sum;
  }
  Dense tensor(const TensorMeaning& t) {
    Dense out{std::get<List>(metadata.eval(t.shape)), {}};
    auto coordinates = axes(t.axes);
    enumerate(out.shape, [&](const List& at) {
      for (std::size_t i = 0; i < at.size(); ++i)
        assignment[coordinates[i]] = at[i];
      out.values.push_back(value(t.value));
      assignment.clear();
    });
    return out;
  }
};
}  // namespace tepl::test
