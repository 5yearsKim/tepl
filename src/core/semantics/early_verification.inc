// Included in solver.cc's private namespace to share the primitive metadata
// interpreter. This checker never invokes attribute synthesis or sufficient
// index-isomorphism matching.
struct EarlyEngine : Engine {
  using Engine::Engine;
  std::size_t degree_limit = 8;
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> degrees;
  std::size_t degree(std::size_t node, std::size_t expr) {
    tick();
    auto key = std::pair(node, expr);
    if (degrees.contains(key)) return degrees.at(key);
    const auto& n = plan.nodes.at(node);
    if (n.capture >= 0) return 1;
    const auto& e = n.expressions.at(expr);
    auto child = [&](std::size_t i) { return degree(node, e.args.at(i)); };
    std::size_t d = 0;
    if (e.op == "read") {
      const auto& input = n.expressions.at(e.args.at(0));
      if (!input.op.starts_with('t')) throw Limit("unsupported tensor read");
      auto id = n.inputs.at(std::stoul(input.op.substr(1)));
      d = plan.nodes.at(id).capture >= 0 ? 1 : degree(id, plan.nodes[id].root);
    } else if (e.op.starts_with("sum:")) {
      d = child(1);
    } else if (e.op == "+" || e.op == "-") {
      d = std::max(child(0), child(1));
    } else if (e.op == "*") {
      d = child(0) + child(1);
    } else if (e.op == "unary:+" || e.op == "unary:-") {
      d = child(0);
    } else if (e.op == "if") {
      d = std::max(child(1), child(2));
    } else if (e.op.empty() || !e.args.empty() ||
               e.op.find_first_not_of("0123456789") != std::string::npos) {
      throw Limit("outside polynomial sparse-witness fragment: " + e.op);
    }
    if (d > degree_limit) throw Limit("sparse witness degree limit");
    degrees.emplace(key, d);
    return d;
  }
  void fixedMetadata(std::size_t end) {
    generating = false;
    for (std::size_t id = 0; id < end; ++id) {
      tick();
      const auto& n = plan.nodes.at(id);
      attrs.emplace_back();
      if (n.capture >= 0) {
        shapes.push_back(literal(captures.at(n.capture)));
        continue;
      }
      for (auto& f : n.fields) {
        if (f.known == -1)
          throw Limit("early verification requires explicit attributes");
        attrs.back().push_back(literal(f.known >= 0 ? known.at(f.known)
                                                    : pinned.at(-2 - f.known)));
      }
      for (auto a : n.assertions) require(eval(id, a).scalar);
      shapes.push_back(literal(constants(vector(eval(id, n.shape)))));
    }
  }
  struct Slot {
    z3::expr enabled, tensor, value;
    std::vector<z3::expr> coordinates;
  };
  std::vector<Slot> slots;
  void allocateSlots(std::size_t d) {
    std::size_t rank = 0;
    for (auto& s : captures) rank = std::max(rank, s.size());
    for (std::size_t j = 0; j < d; ++j) {
      tick();
      auto name = "sparse_" + std::to_string(j);
      Slot slot{ctx.bool_const((name + "_enabled").c_str()),
                ctx.int_const((name + "_tensor").c_str()),
                ctx.int_const((name + "_value").c_str()),
                {}};
      for (std::size_t axis = 0; axis < rank; ++axis)
        slot.coordinates.push_back(
            ctx.int_const((name + "_axis_" + std::to_string(axis)).c_str()));
      auto legal = ctx.bool_val(false);
      for (std::size_t c = 0; c < captures.size(); ++c) {
        auto selected = slot.tensor == integer(c);
        for (std::size_t axis = 0; axis < rank; ++axis)
          selected = selected && (axis < captures[c].size()
                                      ? slot.coordinates[axis] >= 0 &&
                                            slot.coordinates[axis] <
                                                integer(captures[c][axis])
                                      : slot.coordinates[axis] == 0);
        legal = legal || selected;
      }
      solver.add(z3::implies(
          slot.enabled, legal && slot.value >= 1 && slot.value <= integer(d)));
      // Enabled slots form a prefix; distinct physical entries preserve
      // aliases.
      if (j) solver.add(z3::implies(slot.enabled, slots.back().enabled));
      for (const auto& prior : slots) {
        auto same = slot.tensor == prior.tensor;
        for (std::size_t axis = 0; axis < rank; ++axis)
          same = same && slot.coordinates[axis] == prior.coordinates[axis];
        solver.add(z3::implies(slot.enabled && prior.enabled, !same));
      }
      slots.push_back(std::move(slot));
    }
  }
  z3::expr inputValue(std::size_t capture, const List& coordinates) {
    auto value = ctx.real_val(0);
    for (const auto& slot : slots) {
      auto selected = slot.enabled && slot.tensor == integer(capture);
      for (std::size_t axis = 0; axis < coordinates.size(); ++axis)
        selected =
            selected && slot.coordinates[axis] == integer(coordinates[axis]);
      value =
          value + z3::ite(selected, z3::to_real(slot.value), ctx.real_val(0));
    }
    return value;
  }
  void indices(const List& dims,
               const std::function<void(const List&)>& visit) {
    List index(dims.size());
    auto loop = [&](auto&& self, std::size_t axis) -> void {
      tick();
      if (axis == dims.size()) {
        visit(index);
        return;
      }
      for (std::uint64_t i = 0; i < dims[axis]; ++i) {
        index[axis] = i;
        self(self, axis + 1);
      }
    };
    loop(loop, 0);
  }
  z3::expr element(std::size_t node, const List& output) {
    tick();
    auto dims = constants(shapes.at(node));
    if (output.size() != dims.size()) throw Illegal{};
    for (std::size_t i = 0; i < dims.size(); ++i)
      if (output[i] >= dims[i]) throw Illegal{};
    auto& n = plan.nodes.at(node);
    if (n.capture >= 0) return inputValue(n.capture, output);
    Environment env{{"o", literal(output)}};
    return expression(node, n.root, env);
  }
  z3::expr expression(std::size_t node, std::size_t id, Environment& env) {
    tick();
    auto& n = plan.nodes.at(node);
    auto& e = n.expressions.at(id);
    if (e.op == "read") {
      auto child =
          n.inputs.at(std::stoul(n.expressions.at(e.args.at(0)).op.substr(1)));
      return element(child, constants(vector(eval(node, e.args.at(1), env))));
    }
    if (e.op.starts_with("sum:")) {
      auto dims = constants(vector(eval(node, e.args.at(0), env)));
      auto name = e.op.substr(4);
      auto saved = env.contains(name) ? std::optional<Vector>(env.at(name))
                                      : std::nullopt;
      auto result = ctx.real_val(0);
      indices(dims, [&](const List& index) {
        env.insert_or_assign(name, literal(index));
        result = result + expression(node, e.args.at(1), env);
      });
      if (saved)
        env.insert_or_assign(name, *saved);
      else
        env.erase(name);
      return result;
    }
    if (e.op == "if") {
      auto c = eval(node, e.args.at(0), env).scalar.simplify();
      if (!c.is_true() && !c.is_false())
        throw Limit("nonconcrete semantic condition");
      return expression(node, e.args.at(c.is_true() ? 1 : 2), env);
    }
    if (e.args.empty() && !e.op.empty() &&
        e.op.find_first_not_of("0123456789") == std::string::npos)
      return ctx.real_val(e.op.c_str());
    if (e.op == "unary:+") return expression(node, e.args.at(0), env);
    if (e.op == "unary:-") return -expression(node, e.args.at(0), env);
    auto a = expression(node, e.args.at(0), env);
    auto b = expression(node, e.args.at(1), env);
    if (e.op == "+") return a + b;
    if (e.op == "-") return a - b;
    if (e.op == "*") return a * b;
    throw Limit("nonpolynomial scalar expression");
  }
  z3::expr confirm(std::size_t root, const List& output,
                   const std::vector<SparseEntry>& entries) {
    // Independent existing polynomial evaluator at the witness coordinate.
    auto polynomial = expand(root, output).polynomial;
    auto result = ctx.real_val(0);
    for (auto& [monomial, coefficient] : polynomial) {
      auto product = coefficient;
      for (auto& read : monomial) {
        std::uint64_t v = 0;
        for (auto& entry : entries)
          if (!read.empty() && read[0] == entry.capture &&
              List(read.begin() + 1, read.end()) == entry.coordinates)
            v = entry.value;
        product = product * ctx.real_val(std::to_string(v).c_str());
      }
      result = result + product;
    }
    return result.simplify();
  }
  VerificationResult verify(std::size_t max_degree) {
    degree_limit = max_degree;
    VerificationResult result;
    // Admission is independent of equality and covers both sides.
    auto rootDegree = [&](std::size_t root) {
      return plan.nodes.at(root).capture >= 0
                 ? std::size_t(1)
                 : degree(root, plan.nodes[root].root);
    };
    result.degree =
        std::max(rootDegree(plan.source_root), rootDegree(plan.rhs_root));
    if (result.degree > max_degree) throw Limit("sparse witness degree limit");
    try {
      fixedMetadata(plan.source_end);
    } catch (const Illegal&) {
      result.status = VerificationStatus::Skipped;
      result.reason = "illegal LHS metadata";
      return result;
    }
    try {
      // Keep the established source metadata while instantiating RHS nodes.
      auto end = plan.nodes.size();
      for (std::size_t id = plan.source_end; id < end; ++id) {
        tick();
        auto& n = plan.nodes[id];
        attrs.emplace_back();
        if (n.capture >= 0) {
          shapes.push_back(literal(captures.at(n.capture)));
          continue;
        }
        for (auto& f : n.fields) {
          if (f.known == -1)
            throw Limit("early verification requires explicit attributes");
          attrs.back().push_back(literal(
              f.known >= 0 ? known.at(f.known) : pinned.at(-2 - f.known)));
        }
        for (auto a : n.assertions) require(eval(id, a).scalar);
        shapes.push_back(literal(constants(vector(eval(id, n.shape)))));
      }
    } catch (const Illegal&) {
      result.status = VerificationStatus::Counterexample;
      result.reason = "RHS is illegal for legal LHS metadata";
      return result;
    }
    auto dims = constants(shapes.at(plan.source_root));
    if (dims != constants(shapes.at(plan.rhs_root))) {
      result.status = VerificationStatus::Counterexample;
      result.reason = "LHS and RHS output shapes differ";
      return result;
    }
    // Existing exact indexed normalization can cheaply discharge correct
    // rules. Failure is only inconclusive; sparse search remains independent.
    if (plan.algebraic) {
      try {
        if (indexedEqual()) {
          result.status = VerificationStatus::NoCounterexample;
          result.reason =
              "exact semantic normalization proved equality at this metadata";
          return result;
        }
      } catch (const Limit&) {
      }
    }
    allocateSlots(result.degree);
    std::vector<List> outputs;
    std::vector<z3::expr> left, right;
    auto different = ctx.bool_val(false);
    indices(dims, [&](const List& index) {
      outputs.push_back(index);
      left.push_back(element(plan.source_root, index));
      right.push_back(element(plan.rhs_root, index));
      // Normalize sparse arithmetic before asking SMT to solve. This exposes
      // cancellation/distribution without introducing dense tensor variables.
      z3::params normalization(ctx);
      normalization.set("som", true);
      auto delta = (left.back() - right.back()).simplify(normalization);
      different = different || delta != ctx.real_val(0);
    });
    solver.add(different);
    auto status = checkSolver();
    if (status == z3::unknown) throw Limit(solver.reason_unknown());
    if (status == z3::unsat) {
      result.status = VerificationStatus::NoCounterexample;
      result.reason =
          "all tensor values verified at this metadata by the sparse witness "
          "theorem";
      return result;
    }
    auto model = solver.get_model();
    for (auto& slot : slots) {
      if (!model.eval(slot.enabled, true).is_true()) continue;
      SparseEntry entry;
      entry.capture = model.eval(slot.tensor, true).get_numeral_uint64();
      entry.value = model.eval(slot.value, true).get_numeral_uint64();
      for (std::size_t axis = 0; axis < captures.at(entry.capture).size();
           ++axis)
        entry.coordinates.push_back(
            model.eval(slot.coordinates[axis], true).get_numeral_uint64());
      result.entries.push_back(std::move(entry));
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (!model.eval(left[i] != right[i], true).is_true()) continue;
      auto a = confirm(plan.source_root, outputs[i], result.entries);
      auto b = confirm(plan.rhs_root, outputs[i], result.entries);
      if (!(a != b).simplify().is_true())
        throw Limit("sparse witness confirmation failed");
      result.output = outputs[i];
      result.lhs_value = a.to_string();
      result.rhs_value = b.to_string();
      result.status = VerificationStatus::Counterexample;
      result.reason = "independently confirmed sparse counterexample";
      return result;
    }
    throw Limit("solver model has no distinguishing output");
  }
};
