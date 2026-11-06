#pragma once
#include "metadata.h"
namespace tepl_runtime::axes {
struct AxisContext {
  const std::vector<TensorFacts>& operands;
  const Value& attrs;
  Budget& budget;
  std::vector<Value> groups;
  std::vector<std::set<std::size_t>> used;
  std::optional<Value::List> extents;
  AxisContext(const std::vector<TensorFacts>& operands, const Value& attrs,
              Budget& budget)
      : operands(operands),
        attrs(attrs),
        budget(budget),
        used(operands.size()) {}
  Value operand(std::size_t i) const {
    require(i < operands.size(), "missing operand");
    return operands[i].shape_value();
  }
  Value field(std::size_t i) const { return attrs.field(i); }
  Value group(std::size_t i) const {
    require(i < groups.size(), "missing axis group");
    return groups[i];
  }
  void begin_group() { extents.reset(); }
  void bind(std::size_t i, Value positions) {
    auto s = operand(i);
    require(builtins::valid(positions.list(), Integer(s.list().size()), budget),
            "invalid axis binding");
    for (const auto& p : positions.list())
      require(used[i].insert(p.position()).second, "overlapping axis bindings");
    auto selected = builtins::call("gather", {s, positions}, budget).list();
    if (extents)
      require(*extents == selected, "axis group extents differ");
    else
      extents = selected;
  }
  void finish_group(std::optional<Value> domain) {
    if (domain) extents = domain->list();
    require(extents.has_value(), "axis group has no domain");
    Value::List out;
    for (const auto& x : *extents)
      out.emplace_back(Axis{0, groups.size(), out.size(),
                            Integer(from_value<std::uint64_t>(x))});
    groups.emplace_back(out);
    extents.reset();
  }
  void cover(std::size_t i) const {
    require(i < used.size() && used[i].size() == operand(i).list().size(),
            "axis bindings do not cover operand");
  }
  std::vector<std::uint64_t> output(Value v) {
    require(builtins::distinct(v.list(), budget), "duplicate output axes");
    return intrinsics::call("axis_extents", {v}, budget).dimensions();
  }
};
}  // namespace tepl_runtime::axes
