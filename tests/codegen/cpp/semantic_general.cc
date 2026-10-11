#include <cassert>
#include <eggc/all.hpp>

#include "ir/generated.h"
namespace ir = tepl_generated;
using Op = ir::dialects::tensor_lang::Op;
using Dot = ir::dialects::tensor_lang::DotGeneralAttrs;
using Transpose = ir::dialects::tensor_lang::TransposeAttrs;
using Graph = eggc::EGraph<ir::OpNode, ir::analysis::TensorAnalysis>;
namespace rules = ir::rules::semantic_general;
ir::OpNode dot(std::vector<std::uint64_t> lc, std::vector<std::uint64_t> rc,
               eggc::Id a, eggc::Id b, std::vector<std::uint64_t> lb = {},
               std::vector<std::uint64_t> rb = {}) {
  return ir::OpNode::make(Op::DotGeneral, Dot{lc, rc, lb, rb}, {a, b});
}
Graph graph(std::vector<std::uint64_t> a, std::vector<std::uint64_t> b) {
  ir::analysis::TensorBindingTable bindings;
  bindings.register_symbol("a", {a, ir::DType::F32});
  bindings.register_symbol("b", {b, ir::DType::F32});
  return Graph{ir::analysis::TensorAnalysis{bindings}};
}
int main() {
  for (unsigned rank : {2, 3, 5, 12})
    for (unsigned extent : {0, 1, 3}) {
      std::vector<std::uint64_t> dims(rank, 2);
      dims.back() = extent;
      auto g = graph(dims, {extent, 5});
      auto a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b"));
      auto root = g.add(dot({rank - 1}, {0}, a, b));
      eggc::run(g, std::vector{rules::rule_identity::build()});
      std::vector<std::uint64_t> p;
      for (unsigned i = 0; i < rank; ++i) p.push_back(i);
      auto t = g.lookup(ir::OpNode::make(Op::Transpose, Transpose{p}, {a}));
      assert(t);
      auto expected = g.lookup(dot({rank - 1}, {0}, *t, b));
      assert(expected && g.find(root) == g.find(*expected));
    }
  {
    auto g = graph({3, 2, 4}, {3, 5});
    auto a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b"));
    auto root = g.add(dot({0}, {0}, a, b));
    eggc::run(g, std::vector{rules::rule_cycle::build()});
    auto t =
        g.lookup(ir::OpNode::make(Op::Transpose, Transpose{{1, 2, 0}}, {a}));
    assert(t);
    auto expected = g.lookup(dot({2}, {0}, *t, b));
    assert(expected && g.find(root) == g.find(*expected));
  }
  for (auto shape : {std::vector<std::uint64_t>{2, 4, 3},
                     std::vector<std::uint64_t>{2, 2, 3}}) {
    auto g = graph(shape, {3, 5});
    auto a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b"));
    g.add(dot({2}, {0}, a, b));
    eggc::run(g, std::vector{rules::rule_cycle::build()});
    assert(
        !g.lookup(ir::OpNode::make(Op::Transpose, Transpose{{1, 2, 0}}, {a})));
  }
  {
    auto g = graph({2, 3, 4, 5, 6}, {2, 3, 5, 6, 7});
    auto a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b"));
    g.add(dot({3, 4}, {2, 3}, a, b, {0, 1}, {0, 1}));
    eggc::run(g, std::vector{rules::rule_identity::build()});
    assert(g.lookup(
        ir::OpNode::make(Op::Transpose, Transpose{{0, 1, 2, 3, 4}}, {a})));
  }
  {
    ir::analysis::TensorBindingTable inputs;
    inputs.register_symbol("a", {{2, 3}, ir::DType::F32});
    inputs.register_symbol("b", {{2, 3}, ir::DType::F32});
    inputs.register_symbol("c", {{3, 4}, ir::DType::F32});
    Graph g{ir::analysis::TensorAnalysis{inputs}};
    auto a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b")),
         c = g.add(ir::OpNode::input("c"));
    auto sum = g.add(ir::OpNode::make(Op::Add, {}, {a, b}));
    auto root = g.add(dot({1}, {0}, sum, c));
    eggc::run(g, std::vector{rules::rule_distribute::build()});
    auto ac = g.lookup(dot({1}, {0}, a, c)), bc = g.lookup(dot({1}, {0}, b, c));
    assert(ac && bc);
    auto expected = g.lookup(ir::OpNode::make(Op::Add, {}, {*ac, *bc}));
    assert(expected && g.find(root) == g.find(*expected));
  }
  {
    ir::analysis::TensorBindingTable inputs;
    inputs.register_symbol("a", {{2, 3}, ir::DType::F32});
    inputs.register_symbol("b", {{3, 4}, ir::DType::F32});
    inputs.register_symbol("c", {{4, 5}, ir::DType::F32});
    Graph g{ir::analysis::TensorAnalysis{inputs}};
    auto a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b")),
         c = g.add(ir::OpNode::input("c"));
    auto ab = g.add(dot({1}, {0}, a, b));
    auto root = g.add(dot({1}, {0}, ab, c));
    eggc::run(g, std::vector{rules::rule_associate::build()});
    auto bc = g.lookup(dot({1}, {0}, b, c));
    assert(bc);
    auto expected = g.lookup(dot({1}, {0}, a, *bc));
    assert(expected && g.find(root) == g.find(*expected));
  }

  for (unsigned batch_rank : {0, 1, 3, 10}) {
    std::vector<std::uint64_t> batch(batch_rank, 2), axes;
    for (unsigned i = 0; i < batch_rank; ++i) axes.push_back(i);
    auto shape = [&](unsigned m, unsigned n) {
      auto s = batch;
      s.push_back(m);
      s.push_back(n);
      return s;
    };
    ir::analysis::TensorBindingTable inputs;
    inputs.register_symbol("a", {shape(3, 4), ir::DType::F32});
    inputs.register_symbol("b", {shape(4, 5), ir::DType::F32});
    inputs.register_symbol("c", {shape(5, 6), ir::DType::F32});
    Graph g{ir::analysis::TensorAnalysis{inputs}};
    auto a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b")),
         c = g.add(ir::OpNode::input("c"));
    auto ab = g.add(dot({batch_rank + 1}, {batch_rank}, a, b, axes, axes));
    auto root = g.add(dot({batch_rank + 1}, {batch_rank}, ab, c, axes, axes));
    eggc::run(g, std::vector{rules::rule_associate::build()});
    auto bc = g.lookup(dot({batch_rank + 1}, {batch_rank}, b, c, axes, axes));
    assert(bc);
    auto expected =
        g.lookup(dot({batch_rank + 1}, {batch_rank}, a, *bc, axes, axes));
    assert(expected && g.find(root) == g.find(*expected));
  }
  {
    ir::analysis::TensorBindingTable inputs;
    inputs.register_symbol("x", {{2, 3}, ir::DType::F32});
    inputs.register_symbol("w", {{3, 5}, ir::DType::F32});
    inputs.register_symbol("a", {{3, 4}, ir::DType::F32});
    inputs.register_symbol("b", {{4, 5}, ir::DType::F32});
    Graph g{ir::analysis::TensorAnalysis{inputs}};
    auto x = g.add(ir::OpNode::input("x")), w = g.add(ir::OpNode::input("w")),
         a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b"));
    auto ab = g.add(dot({1}, {0}, a, b));
    auto sum = g.add(ir::OpNode::make(Op::Add, {}, {w, ab}));
    auto root = g.add(dot({1}, {0}, x, sum));
    eggc::run(g, std::vector{rules::rule_lora::build()});
    auto xw = g.lookup(dot({1}, {0}, x, w)), xa = g.lookup(dot({1}, {0}, x, a));
    assert(xw && xa);
    auto out = g.lookup(dot({1}, {0}, *xa, b));
    assert(out);
    auto expected = g.lookup(ir::OpNode::make(Op::Add, {}, {*xw, *out}));
    assert(expected && g.find(root) == g.find(*expected));
  }

  {
    auto g = graph({2}, {2});
    auto a = g.add(ir::OpNode::input("a"));
    auto root = g.add(dot({}, {}, a, a));
    eggc::run(g, std::vector{rules::rule_symmetric_product::build()});
    auto expected =
        g.lookup(ir::OpNode::make(Op::Transpose, Transpose{{1, 0}}, {root}));
    assert(expected && g.find(root) == g.find(*expected));
  }

  {
    auto g = graph({2, 3}, {2});
    auto a = g.add(ir::OpNode::input("a"));
    using Reduce = ir::dialects::tensor_lang::Reduction;
    auto root = g.add(ir::OpNode::make(Op::SquaredSum, Reduce{{1}}, {a}));
    eggc::run(g, std::vector{rules::rule_reused_reduction::build()});
    auto fold = g.lookup(ir::OpNode::make(Op::Fold, Reduce{{1}}, {a}));
    assert(fold);
    auto expected = g.lookup(dot({}, {}, *fold, *fold, {0}, {0}));
    assert(expected && g.find(root) == g.find(*expected));
  }

  {
    auto g = graph({2, 3}, {3, 4});
    auto a = g.add(ir::OpNode::input("a")), b = g.add(ir::OpNode::input("b"));
    auto root = g.add(dot({1}, {0}, a, b));
    eggc::run(g, std::vector{rules::rule_bound_intermediate::build()});
    auto expected =
        g.lookup(ir::OpNode::make(Op::Transpose, Transpose{{0, 1}}, {root}));
    assert(expected && g.find(root) == g.find(*expected));
  }
}
