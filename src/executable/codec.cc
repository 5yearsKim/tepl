#include "src/executable/codec.h"

#include <algorithm>
#include <charconv>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>

#include "src/executable/schema_data.h"

namespace tepl::executable {
namespace detail {
std::uint64_t parseExtent(const std::string& s) {
  std::uint64_t value;
  auto result = std::from_chars(s.data(), s.data() + s.size(), value);
  if (result.ec != std::errc{} || result.ptr != s.data() + s.size())
    throw std::invalid_argument("extent outside u64 range");
  return value;
}
std::int64_t parseSigned(const std::string& s) {
  std::int64_t value;
  auto result = std::from_chars(s.data(), s.data() + s.size(), value);
  if (result.ec != std::errc{} || result.ptr != s.data() + s.size())
    throw std::invalid_argument("integer outside i64 range");
  return value;
}
}  // namespace detail
namespace {
using detail::Json;
// This validator implements exactly the vocabulary used by our checked-in
// schema. It does not fetch remote references or claim to be a general schema
// implementation. The same schema is usable with standard validators.
void check(const Json& value, const Json& spec, const Json& root,
           const std::string& path, std::size_t depth = 0) {
  auto fail = [&](const std::string& message) {
    throw std::invalid_argument(path + ": " + message);
  };
  if (depth > 512) fail("maximum schema nesting exceeded");
  if (spec.contains("$ref")) {
    auto ref = spec.at("$ref").get<std::string>();
    check(value, root.at(Json::json_pointer(ref.substr(1))), root, path,
          depth + 1);
    return;
  }
  for (const auto* key : {"oneOf", "anyOf"})
    if (spec.contains(key)) {
      unsigned matches = 0;
      for (const auto& alt : spec.at(key)) try {
          check(value, alt, root, path, depth + 1);
          ++matches;
        } catch (const std::invalid_argument&) {
        }
      if (!matches || (std::string_view(key) == "oneOf" && matches != 1))
        fail("does not match " + std::string(key));
    }
  if (spec.contains("const") && value != spec.at("const"))
    fail("unexpected constant or unsupported version/capability");
  if (spec.contains("enum") &&
      std::find(spec.at("enum").begin(), spec.at("enum").end(), value) ==
          spec.at("enum").end())
    fail("unknown enum value");
  if (spec.contains("type")) {
    const auto& t = spec.at("type");
    bool ok = (t == "object" && value.is_object()) ||
              (t == "array" && value.is_array()) ||
              (t == "string" && value.is_string()) ||
              (t == "integer" && value.is_number_integer()) ||
              (t == "boolean" && value.is_boolean()) ||
              (t == "null" && value.is_null());
    if (!ok) fail("expected " + t.get<std::string>());
  }
  if (spec.contains("minimum") && value < spec.at("minimum"))
    fail("below minimum");
  if (spec.contains("maximum") && value > spec.at("maximum"))
    fail("above maximum");
  if (spec.contains("pattern") &&
      !std::regex_match(value.get<std::string>(),
                        std::regex(spec.at("pattern").get<std::string>())))
    fail("invalid decimal spelling");
  if (value.is_object() && spec.contains("properties")) {
    for (const auto& key : spec.at("required"))
      if (!value.contains(key.get<std::string>()))
        fail("missing " + key.get<std::string>());
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (!spec.at("properties").contains(it.key()))
        fail("unknown field " + it.key());
      check(it.value(), spec.at("properties").at(it.key()), root,
            path + "/" + it.key(), depth + 1);
    }
  }
  if (value.is_array() && spec.contains("items")) {
    for (std::size_t i = 0; i < value.size(); ++i)
      check(value[i], spec.at("items"), root, path + "/" + std::to_string(i),
            depth + 1);
    if (spec.value("uniqueItems", false))
      for (std::size_t i = 0; i < value.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
          if (value[i] == value[j]) fail("duplicate item");
  }
}
}  // namespace
std::string_view schema() { return detail::kSchema; }
ExecutableProgram load(std::string_view text) {
  try {
    if (text.size() > 64 * 1024 * 1024)
      throw std::invalid_argument("compiled program exceeds 64 MiB");
    std::map<int, std::set<std::string>> keys;
    auto j = Json::parse(text, [&](int depth, Json::parse_event_t event,
                                   Json& value) {
      if (depth > 256)
        throw std::invalid_argument("compiled program nesting limit exceeded");
      if (event == Json::parse_event_t::object_start) keys[depth + 1].clear();
      if (event == Json::parse_event_t::key &&
          !keys[depth].insert(value.get<std::string>()).second)
        throw std::invalid_argument("duplicate JSON object key");
      return true;
    });
    const auto spec = Json::parse(schema());
    check(j, spec, spec, "$");
    ExecutableProgram p;
    p.policy_ = j.at("policy");
    p.ir_ = detail::unpack<ProgramData>(j.at("program"));
    bool inference = false;
    for (const auto& r : p.ir_.rules) inference |= r.inference.has_value();
    if (inference && j.at("capabilities").size() != 3)
      throw std::invalid_argument("missing inferred-applicability capability");
    p.axes_ = detail::unpack<decltype(p.axes_)>(j.at("axis_definitions"));
    validate(p);
    return p;
  } catch (const Json::exception& e) {
    throw std::invalid_argument(std::string("invalid compiled program: ") +
                                e.what());
  }
}
std::string serialize(const ExecutableProgram& p) {
  validate(p);
  Json j = {{"format", "tepl.rewrite"},
            {"version", 2},
            {"policy", p.policy_},
            {"capabilities", {"tepl.core.v2", "tepl.axis-metadata.v2"}},
            {"program", detail::pack(p.ir_)},
            {"axis_definitions", detail::pack(p.axes_)}};
  for (const auto& r : p.ir_.rules)
    if (r.inference) {
      j["capabilities"].push_back("tepl.inferred-applicability.v1");
      break;
    }
  const auto spec = Json::parse(schema());
  check(j, spec, spec, "$");
  return j.dump(2) + "\n";
}
}  // namespace tepl::executable
