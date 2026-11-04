#include "labs/accepted_core/dense_reference.h"
executable::ExecutableProgram compile(const syntax::ParsedModule& module) {
  auto r = core::compile(module, {core::SemanticPolicy::kAlgebraic, true});
  if (!r.ok()) {
    std::string e;
    for (const auto& d : r.diagnostics) e += d.message + "\n";
    throw std::runtime_error(e);
  }
  return std::move(*r.program);
}
void invalid(const Json& j) {
  bool bad = false;
  try {
    executable::load(j.dump());
  } catch (const std::invalid_argument&) {
    bad = true;
  }
  require(bad, "corrupted portable program accepted");
}
int main(int argc, char** argv) {
  try {
    bool test = argc > 1 && std::string(argv[1]) == "--test";
    Json rows = Json::array();
    unsigned negative = 0, numeric = 0;
    for (int arg = test ? 2 : 1; arg < argc; ++arg) {
      auto source = read(argv[arg]);
      auto parsed = parse(source, argv[arg]);
      require(parsed.ok(), "fixture parsing " + std::string(argv[arg]));
      auto p = compile(*parsed.program);
      const auto rule = p.ir().rules.at(0).name;
      require(p.ir().rules.at(0).core.has_value(), "core missing");
      auto serialized = executable::serialize(p);
      auto loaded = executable::load(serialized);
      require(executable::serialize(loaded) == serialized, "roundtrip");
      Json j = Json::parse(serialized), q = j;
      q["program"]["rules"][0]["core"]["nodes"][0][0]["capture"] = 999999;
      invalid(q);
      ++negative;
      q = j;
      q["program"]["rules"][0]["core"]["checks"][0].push_back(999999);
      invalid(q);
      ++negative;
      q = j;
      q["program"]["rules"][0]["core"]["attributes"][0][0][1] = 999999;
      invalid(q);
      ++negative;
      q = j;
      q["policy"] = "strict";
      invalid(q);
      ++negative;
      q = j;
      q["capabilities"] =
          Json::array({"tepl.core.v3", "tepl.axis-metadata.v2"});
      invalid(q);
      ++negative;
      if (rule == "batched_reassociation") {
        auto changed = source;
        auto pos = changed.find("op dot_general");
        pos = changed.find("yield a;", pos);
        require(pos != std::string::npos, "dtype fixture");
        changed.replace(pos, 8, "yield f64;");
        auto bad = parse(changed, "dtype.tepl");
        require(bad.ok(), "dtype parse");
        require(!core::compile(*bad.program,
                               {core::SemanticPolicy::kAlgebraic, true})
                     .ok(),
                "custom dtype accepted");
        ++negative;
        q = j;
        for (auto& op : q["program"]["operations"])
          if (op["name"] == "dot_general")
            op["dtype"]["statements"] = Json::array();
        invalid(q);
        ++negative;

        q = j;
        q["program"]["rules"][0]["conditions"] = Json::array();
        auto conditional = executable::load(q.dump());
        for (const auto& g : conditional.ir().graphs) {
          auto out = rewrite::run(conditional, g);
          require(reference(conditional.ir(), g) == reference(conditional.ir(), out.graph), "conditional rewrite values");
        }
        ++negative;
      }
      if (rule.starts_with("chain_automatic")) {
        q = j;
        q["program"]["rules"][0]["conditions"] = Json::array();
        auto conditional = executable::load(q.dump());
        for (const auto& g : conditional.ir().graphs) {
          auto out = rewrite::run(conditional, g);
          require(reference(conditional.ir(), g) == reference(conditional.ir(), out.graph), "conditional rewrite values");
        }
        ++negative;
        for (bool disjunction : {false, true}) {
          auto changed = source;
          std::string text = "&& l1.attrs.lhs_contracting_dimensions == [2]";
          auto pos = changed.find(text);
          require(pos != std::string::npos, "negative fixture");
          changed.replace(
              pos, text.size(),
              disjunction ? "&& (l1.attrs.lhs_contracting_dimensions == [2] || "
                            "l1.attrs.lhs_contracting_dimensions == [1])"
                          : "");
          auto bad = parse(changed, "missing.tepl");
          require(bad.ok(), "negative parse");
          require(core::compile(*bad.program,
                                 {core::SemanticPolicy::kAlgebraic, true})
                       .ok(),
                  "conditional constructor compiles without implying feasibility");
          ++negative;
        }
      }
      if (rule == "transpose_reduce") {
        auto changed = source.substr(0, source.find("\nrule ")) +
                       "\nrule needs_numeric { (transpose X) => (reduce_sum "
                       "(transpose X)) }";
        auto bad = parse(changed, "numeric.tepl");
        require(bad.ok(), "numeric premise parse");
        require(!core::compile(*bad.program,
                               {core::SemanticPolicy::kAlgebraic, true})
                     .ok(),
                "new numeric premise accepted");
        ++negative;
      }
      for (const auto& graph : loaded.ir().graphs) {
        rewrite::Options options;
        options.max_steps = 1;
        options.rule = rule;
        auto result = rewrite::run(loaded, graph, options);
        require(
            result.rewrite_count == 1,
            "rewrite failed " + rule + ":" + graph.name + ":" + result.detail);
        executable::validateGraph(loaded.ir(), result.graph);
        require(reference(loaded.ir(), graph) ==
                    reference(loaded.ir(), result.graph),
                "numeric mismatch " + rule + ":" + graph.name);
        ++numeric;
      }
      if (!test)
        for (int round = -1; round < 10; ++round) {
          double compile_ms = timeMs([&] {
            auto z = compile(*parsed.program);
            require(z.ir().rules[0].core.has_value(), "compile");
          });
          double load_ms = timeMs([&] {
            auto z = executable::load(serialized);
            require(z.ir().rules[0].core.has_value(), "load");
          });
          rows.push_back({{"rule", rule},
                          {"phase", "compile"},
                          {"round", round},
                          {"ms", compile_ms},
                          {"bytes", serialized.size()}});
          rows.push_back({{"rule", rule},
                          {"phase", "load_validate"},
                          {"round", round},
                          {"ms", load_ms}});
          for (const auto& graph : loaded.ir().graphs) {
            rewrite::Options options;
            options.max_steps = 1;
            options.rule = rule;
            std::size_t rewrites = 0;
            double runtime_ms =
                timeMs([&] {
                  for (int i = 0; i < 100; ++i)
                    rewrites +=
                        rewrite::run(loaded, graph, options).rewrite_count;
                }) /
                100;
            require(rewrites == 100, "runtime count");
            rows.push_back({{"rule", rule},
                            {"phase", "rewrite"},
                            {"graph", graph.name},
                            {"round", round},
                            {"batch", 100},
                            {"ms", runtime_ms}});
          }
        }
    }
    std::cout << Json({{"numeric_graph_checks", numeric},
                       {"negative_checks", negative},
                       {"samples", rows}})
                     .dump(2)
              << "\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
