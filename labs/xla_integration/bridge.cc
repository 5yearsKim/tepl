#include "labs/xla_integration/bridge.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "src/executable/program.h"
#include "src/rewrite/matcher.h"
using namespace tepl;
namespace {
struct KeyHash {
  std::size_t operator()(const std::vector<int64_t>& key) const {
    std::size_t h = 0;
    for (auto n : key)
      h ^= std::hash<int64_t>{}(n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};
struct Cached {
  std::vector<int64_t> fields;
  std::array<std::size_t, 4> lengths;
};
struct Context {
  std::vector<executable::ExecutableProgram> programs;
  std::mutex mutex;
  std::unordered_map<std::vector<int64_t>, Cached, KeyHash> cache;
  uint64_t hits = 0, misses = 0;
};
std::string read(const char* path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("program unavailable");
  return {std::istreambuf_iterator<char>(f), {}};
}
core::AttributeValue axisList(TeplList xs) {
  core::AttributeValue a{core::AttributeValue::Kind::kList, {}, {}};
  for (size_t i = 0; i < xs.size; ++i)
    a.elements.push_back(
        {core::AttributeValue::Kind::kInteger, std::to_string(xs.data[i]), {}});
  return a;
}
void errorText(char* out, size_t n, const char* value) {
  if (out && n) std::snprintf(out, n, "%s", value);
}
const std::string names[] = {
    "lhs_contracting_dimensions", "rhs_contracting_dimensions",
    "lhs_batching_dimensions", "rhs_batching_dimensions"};
}  // namespace
extern "C" void* tepl_xla_load(const char* inferred, char* error, size_t size) {
  try {
    auto c = std::make_unique<Context>();
    c->programs.push_back(executable::load(read(inferred)));
    return c.release();
  } catch (const std::exception& e) {
    errorText(error, size, e.what());
    return nullptr;
  }
}
extern "C" int tepl_xla_fold(void* opaque, int side, TeplList lhs, TeplList rhs,
                             TeplList permutation, const TeplList fields[4],
                             int64_t* output, size_t capacity,
                             size_t lengths[4], char* error,
                             size_t error_size) {
  try {
    if (!opaque || side < 0 || side > 1)
      throw std::runtime_error("invalid bridge request");
    auto& c = *static_cast<Context*>(opaque);
    std::lock_guard lock(c.mutex);
    // Cache the complete metadata input, not just ranks. Extents are required
    // to preserve source-legality checks when a similar shape is invalid.
    std::vector<int64_t> key;
    std::size_t key_size = 8 + lhs.size + rhs.size + permutation.size;
    for (int i = 0; i < 4; ++i) key_size += fields[i].size;
    key.reserve(key_size);
    key.push_back(side);
    auto append = [&](TeplList list) {
      key.push_back(list.size);
      for (std::size_t i = 0; i < list.size; ++i) key.push_back(list.data[i]);
    };
    append(lhs);
    append(rhs);
    append(permutation);
    for (int i = 0; i < 4; ++i) append(fields[i]);
    if (auto it = c.cache.find(key); it != c.cache.end()) {
      const auto& hit = it->second;
      if (hit.fields.size() > capacity)
        throw std::runtime_error("output capacity");
      std::copy(hit.fields.begin(), hit.fields.end(), output);
      std::copy(hit.lengths.begin(), hit.lengths.end(), lengths);
      ++c.hits;
      return 0;
    }
    ++c.misses;
    const auto& p = c.programs.at(0);
    core::ConcreteGraph graph;
    graph.name = "xla_candidate";
    for (int i = 0; i < 2; ++i) {
      auto shape = i ? rhs : lhs;
      core::ConcreteNode n{};
      n.kind = core::ConcreteNode::Kind::kInput;
      n.text = i ? "Y" : "X";
      n.dtype = core::DType::kF32;
      n.shape = std::vector<std::uint64_t>();
      for (size_t j = 0; j < shape.size; ++j) {
        if (shape.data[j] < 0) throw std::runtime_error("negative extent");
        n.shape->push_back(shape.data[j]);
      }
      graph.nodes.push_back(std::move(n));
    }
    auto operation = [&](const char* name) -> const executable::Operation& {
      for (const auto& op : p.ir().operations)
        if (op.name == name) return op;
      throw std::runtime_error("operation missing");
    };
    const auto& transpose = operation("transpose");
    const auto& dot = operation("dot_general");
    core::ConcreteNode t{};
    t.kind = core::ConcreteNode::Kind::kOperation;
    t.operation = transpose.id;
    t.operands = {size_t(side)};
    t.attributes = {axisList(permutation)};
    graph.nodes.push_back(std::move(t));
    core::ConcreteNode d{};
    d.kind = core::ConcreteNode::Kind::kOperation;
    d.operation = dot.id;
    d.operands = side ? std::vector<size_t>{0, 2} : std::vector<size_t>{2, 1};
    const auto& schema = p.ir().attribute_schemas.at(dot.attributes->value);
    for (const auto& f : schema.fields) {
      size_t i = 0;
      for (; i < 4 && f.name != names[i]; ++i) {
      }
      if (i == 4) throw std::runtime_error("unexpected dot field");
      d.attributes.push_back(axisList(fields[i]));
    }
    graph.nodes.push_back(std::move(d));
    graph.root = 3;
    std::size_t rule_id = p.ir().rules.size();
    for (const auto& r : p.ir().rules)
      if (r.name == (side ? "fold_rhs" : "fold_lhs")) rule_id = r.id.value;
    if (rule_id == p.ir().rules.size() || !p.ir().rules[rule_id].core)
      throw std::runtime_error("expected certified folding rule");
    rewrite::Budget budget;
    for (std::size_t i = 0; i < graph.nodes.size(); ++i)
      rewrite::infer(p.ir(), graph, i, budget);
    rewrite::Bindings bindings;
    const auto& rule = p.ir().rules[rule_id];
    if (!rewrite::match(p.ir(), rule, graph, graph.root, bindings, budget))
      throw std::runtime_error("candidate did not match");
    rewrite::Evaluator evaluator(p, graph, rule_id, bindings, budget);
    if (!evaluator.evaluateWhere())
      throw std::runtime_error("candidate guard rejected");
    auto id = *rule.core->program.replacement.back().descriptor;
    const auto& result = bindings.descriptors.at(id);
    size_t at = 0;
    for (size_t i = 0; i < 4; ++i) {
      size_t f = 0;
      while (schema.fields.at(f).name != names[i]) ++f;
      const auto& a = rewrite::list(result.fields.at(f));
      lengths[i] = a.size();
      if (at + lengths[i] > capacity)
        throw std::runtime_error("output capacity");
      for (const auto& x : a) output[at++] = rewrite::position(x);
    }
    Cached entry;
    entry.fields.assign(output, output + at);
    std::copy(lengths, lengths + 4, entry.lengths.begin());
    if (c.cache.size() >= 256) c.cache.clear();
    c.cache.emplace(std::move(key), std::move(entry));
    return 0;
  } catch (const std::exception& e) {
    errorText(error, error_size, e.what());
    return 1;
  }
}
extern "C" int tepl_xla_cache(void* opaque, int clear, uint64_t* hits,
                              uint64_t* misses) {
  try {
    if (!opaque) return 1;
    auto& c = *static_cast<Context*>(opaque);
    std::lock_guard lock(c.mutex);
    if (clear) c.cache.clear();
    if (hits) *hits = c.hits;
    if (misses) *misses = c.misses;
    return 0;
  } catch (...) {
    return 1;
  }
}
extern "C" void tepl_xla_destroy(void* context) {
  delete static_cast<Context*>(context);
}
