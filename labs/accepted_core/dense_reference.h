#pragma once
// Correctness fixtures for public compile/load/rewrite APIs, with an independent
// dense integer tensor interpreter.
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

#include "src/core/pipeline.h"
#include "src/rewrite/engine.h"
#include "src/rewrite/value.h"
using namespace tepl;
using Json = nlohmann::json;
using Shape = std::vector<std::uint64_t>;
void require(bool c, const std::string& s) {
  if (!c) throw std::runtime_error(s);
}
std::string read(const std::string& p) {
  std::ifstream f(p);
  require(bool(f), p);
  return {std::istreambuf_iterator<char>(f), {}};
}
std::size_t size(const Shape& s) {
  std::size_t n = 1;
  for (auto x : s) n *= x;
  return n;
}
void each(const Shape& s, const std::function<void(const Shape&)>& f) {
  Shape p(s.size());
  auto loop = [&](auto&& self, std::size_t d) -> void {
    if (d == s.size()) {
      f(p);
      return;
    }
    for (p[d] = 0; p[d] < s[d]; ++p[d]) self(self, d + 1);
  };
  loop(loop, 0);
}
std::size_t offset(const Shape& s, const Shape& p) {
  require(s.size() == p.size(), "rank");
  std::size_t v = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    require(p[i] < s[i], "bounds");
    v = v * s[i] + p[i];
  }
  return v;
}
struct Tensor {
  Shape shape;
  std::vector<std::int64_t> data;
  bool operator==(const Tensor&) const = default;
};
Tensor reference(const executable::ProgramData& p,
                 const core::ConcreteGraph& g) {
  std::vector<Tensor> values;
  for (const auto& n : g.nodes) {
    Tensor out;
    if (n.kind == core::ConcreteNode::Kind::kInput) {
      require(n.shape.has_value(), "input shape available");
      out.shape = *n.shape;
      out.data.assign(size(out.shape), 0);
      unsigned seed = 0;
      for (unsigned char c : n.text) seed = seed * 31 + c;
      for (std::size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = std::int64_t((i * 3 + seed) % 5) - 2;
    } else {
      require(n.kind == core::ConcreteNode::Kind::kOperation,
              "reference literal unsupported");
      const auto& op = p.operations.at(n.operation->value);
      auto field = [&](const std::string& key) {
        std::vector<std::size_t> xs;
        const auto& schema = p.attribute_schemas.at(op.attributes->value);
        for (std::size_t i = 0; i < schema.fields.size(); ++i)
          if (schema.fields[i].name == key) {
            for (const auto& x : n.attributes.at(i).elements)
              xs.push_back(std::stoull(x.text));
            return xs;
          }
        throw std::runtime_error("field " + key);
      };
      const auto& a = values.at(n.operands[0]);
      if (op.name == "add" || op.name == "multiply" || op.name == "reverse" ||
          op.name == "negate")
        out.shape = a.shape;
      else if (op.name == "broadcast_in_dim") {
        auto dims = field("dimensions"), ext = field("new_extents");
        out.shape.resize(a.shape.size() + ext.size());
        std::vector<bool> input(out.shape.size());
        for (std::size_t i = 0; i < dims.size(); ++i) {
          out.shape.at(dims[i]) = a.shape.at(i);
          input[dims[i]] = true;
        }
        std::size_t j = 0;
        for (std::size_t i = 0; i < input.size(); ++i)
          if (!input[i]) out.shape[i] = ext.at(j++);
      } else if (op.name == "transpose")
        for (auto i : field("permutation")) out.shape.push_back(a.shape.at(i));
      else if ((op.name == "reduce_sum" || op.name == "remove_unit_axes")) {
        auto dims = field("dimensions");
        for (std::size_t i = 0; i < a.shape.size(); ++i)
          if (std::find(dims.begin(), dims.end(), i) == dims.end())
            out.shape.push_back(a.shape[i]);
      } else if (op.name == "dot_general") {
        const auto& b = values.at(n.operands[1]);
        auto lb = field("lhs_batching_dimensions"),
             rb = field("rhs_batching_dimensions"),
             lk = field("lhs_contracting_dimensions"),
             rk = field("rhs_contracting_dimensions");
        for (auto i : lb) out.shape.push_back(a.shape.at(i));
        for (std::size_t i = 0; i < a.shape.size(); ++i)
          if (std::find(lb.begin(), lb.end(), i) == lb.end() &&
              std::find(lk.begin(), lk.end(), i) == lk.end())
            out.shape.push_back(a.shape[i]);
        for (std::size_t i = 0; i < b.shape.size(); ++i)
          if (std::find(rb.begin(), rb.end(), i) == rb.end() &&
              std::find(rk.begin(), rk.end(), i) == rk.end())
            out.shape.push_back(b.shape[i]);
      }
      out.data.assign(size(out.shape), 0);
      if (op.name == "add" || op.name == "multiply") {
        const auto& b = values.at(n.operands[1]);
        require(a.shape == b.shape && out.shape == a.shape, "add shape");
        for (std::size_t i = 0; i < out.data.size(); ++i)
          out.data[i] =
              op.name == "add" ? a.data[i] + b.data[i] : a.data[i] * b.data[i];
      } else if (op.name == "reverse") {
        auto dims = field("dimensions");
        each(out.shape, [&](const auto& at) {
          auto in = at;
          for (auto d : dims) in[d] = a.shape[d] - 1 - at[d];
          out.data[offset(out.shape, at)] = a.data[offset(a.shape, in)];
        });
      } else if (op.name == "negate") {
        for (std::size_t i = 0; i < out.data.size(); ++i)
          out.data[i] = -a.data[i];
      } else if (op.name == "broadcast_in_dim") {
        auto dims = field("dimensions");
        each(out.shape, [&](const auto& at) {
          Shape in;
          for (auto d : dims) in.push_back(at[d]);
          out.data[offset(out.shape, at)] = a.data[offset(a.shape, in)];
        });
      } else if (op.name == "transpose") {
        auto order = field("permutation");
        each(out.shape, [&](const auto& at) {
          Shape in(at.size());
          for (std::size_t i = 0; i < at.size(); ++i)
            in.at(order.at(i)) = at[i];
          out.data[offset(out.shape, at)] = a.data[offset(a.shape, in)];
        });
      } else if ((op.name == "reduce_sum" || op.name == "remove_unit_axes")) {
        auto dims = field("dimensions");
        each(a.shape, [&](const auto& at) {
          Shape to;
          for (std::size_t i = 0; i < at.size(); ++i)
            if (std::find(dims.begin(), dims.end(), i) == dims.end())
              to.push_back(at[i]);
          out.data[offset(out.shape, to)] += a.data[offset(a.shape, at)];
        });
      } else if (op.name == "dot_general") {
        const auto& b = values.at(n.operands[1]);
        auto lb = field("lhs_batching_dimensions"),
             rb = field("rhs_batching_dimensions"),
             lk = field("lhs_contracting_dimensions"),
             rk = field("rhs_contracting_dimensions");
        each(a.shape, [&](const auto& ai) {
          each(b.shape, [&](const auto& bi) {
            for (std::size_t j = 0; j < lb.size(); ++j)
              if (ai.at(lb[j]) != bi.at(rb.at(j))) return;
            for (std::size_t j = 0; j < lk.size(); ++j)
              if (ai.at(lk[j]) != bi.at(rk.at(j))) return;
            Shape to;
            for (auto i : lb) to.push_back(ai.at(i));
            for (std::size_t i = 0; i < ai.size(); ++i)
              if (std::find(lb.begin(), lb.end(), i) == lb.end() &&
                  std::find(lk.begin(), lk.end(), i) == lk.end())
                to.push_back(ai[i]);
            for (std::size_t i = 0; i < bi.size(); ++i)
              if (std::find(rb.begin(), rb.end(), i) == rb.end() &&
                  std::find(rk.begin(), rk.end(), i) == rk.end())
                to.push_back(bi[i]);
            out.data[offset(out.shape, to)] +=
                a.data[offset(a.shape, ai)] * b.data[offset(b.shape, bi)];
          });
        });
      } else
        throw std::runtime_error("unsupported reference op " + op.name);
    }
    require(!n.shape || *n.shape == out.shape, "independent shape mismatch");
    values.push_back(std::move(out));
  }
  return values.at(g.root);
}
