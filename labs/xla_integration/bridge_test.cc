#include "labs/xla_integration/bridge.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "src/core/pipeline.h"
void require(bool x, const char* why) {
  if (!x) throw std::runtime_error(why);
}
int main(int argc, char** argv) {
  try {
    require(argc == 3, "expected program and source");
    char error[2048] = {};
    auto* c = tepl_xla_load(argv[1], error, sizeof(error));
    require(c, error);
    size_t checked = 0;
    for (int rank = 2; rank <= 8; ++rank)
      for (int first = 0; first < rank; ++first)
        for (int second = first + 1; second < rank; ++second)
          for (int reverse = 0; reverse < 2; ++reverse)
            for (int side = 0; side < 2; ++side)
              for (int extent : {0, 1, 2}) {
                std::vector<int64_t> a(rank, extent == 1 ? 1 : 2),
                    b = a, p(rank), batch, k{first};
                std::iota(p.begin(), p.end(), 0);
                std::swap(p[first], p[second]);
                for (int i = 0; i < rank; ++i)
                  if (i != first && i != second) batch.push_back(i);
                if (reverse) std::reverse(batch.begin(), batch.end());
                if (extent == 0 && !batch.empty()) {
                  a[batch[0]] = 0;
                  b[batch[0]] = 0;
                }
                TeplList fields[] = {{k.data(), k.size()},
                                     {k.data(), k.size()},
                                     {batch.data(), batch.size()},
                                     {batch.data(), batch.size()}};
                std::vector<int64_t> out(rank * 4 + 16);
                size_t lengths[4];
                int code = tepl_xla_fold(
                    c, side, {a.data(), a.size()}, {b.data(), b.size()},
                    {p.data(), p.size()}, fields, out.data(), out.size(),
                    lengths, error, sizeof(error));
                require(!code, error);
                size_t at = 0;
                for (int i = 0; i < 4; ++i) {
                  std::vector<int64_t> expected(
                      fields[i].data, fields[i].data + fields[i].size);
                  if (i == side || i == side + 2)
                    for (auto& n : expected) n = p[n];
                  require(lengths[i] == expected.size(), "length differs");
                  for (auto n : expected)
                    require(out[at++] == n,
                            "attribute differs from upstream gather");
                }
                ++checked;
              }
    uint64_t hits = 0, misses = 0;
    require(!tepl_xla_cache(c, 0, &hits, &misses), "cache statistics");
    require(hits > 0 && misses > 0, "cache hit and miss exercised");
    std::vector<int64_t> a{2, 2}, b = a, p{1, 0}, lc{1}, rc{0};
    TeplList fields[] = {
        {lc.data(), 1}, {rc.data(), 1}, {nullptr, 0}, {nullptr, 0}};
    int64_t out[16];
    size_t lengths[4];
    require(!tepl_xla_fold(c, 1, {a.data(), 2}, {b.data(), 2}, {p.data(), 2},
                           fields, out, 16, lengths, error, sizeof(error)),
            "prime cache");
    b[1] = 3;
    require(tepl_xla_fold(c, 1, {a.data(), 2}, {b.data(), 2}, {p.data(), 2},
                          fields, out, 16, lengths, error, sizeof(error)) != 0,
            "cached shapes must retain extent legality");
    b[1] = 2;
    p = {0, 1};
    require(tepl_xla_fold(c, 1, {a.data(), 2}, {b.data(), 2}, {p.data(), 2},
                          fields, out, 16, lengths, error, sizeof(error)) != 0,
            "cached permutation must retain guard");
    tepl_xla_destroy(c);
    std::ifstream f(argv[2]);
    std::string original{std::istreambuf_iterator<char>(f), {}};
    const std::string guards[] = {
        "&& len(XDims) == len(d.attrs.lhs_batching_dimensions)+2",
        "&& len(YDims) == len(d.attrs.rhs_batching_dimensions)+2",
        "&& len(d.attrs.lhs_contracting_dimensions)==1",
        "&& len(d.attrs.rhs_contracting_dimensions)==1"};
    for (const auto& guard : guards) {
      auto text = original;
      auto pos = text.find(guard);
      require(pos != std::string::npos, "guard fixture");
      text.erase(pos, guard.size());
      auto parsed = tepl::parse(text, "missing_guard.tepl");
      require(parsed.ok(), "negative parses");
      require(
          tepl::core::compile(*parsed.program,
                               {tepl::core::SemanticPolicy::kAlgebraic, true})
               .ok(),
          "conditional constructor should compile without guard implication");
    }
    std::cout << checked
              << " native metadata cases match handwritten attributes; 4 "
                 "conditional compilations and 2 cache-boundary controls checked\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
