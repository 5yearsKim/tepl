#include <iostream>
#include <string>
#include <string_view>

#include "antlr4-runtime.h"
#include "grammar/TeplLexer.h"
#include "grammar/TeplParser.h"
#include "src/parse.h"

namespace {

int failures = 0;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    ++failures;
  }
}

struct Case {
  std::string_view name;
  std::string_view source;
};

void TestValidSyntax() {
  const Case cases[] = {
      {"minimal", "rule r { X => X }"},
      {"scalar shape", "rule r { X: [] X => X }"},
      {"fixed dimensions", "rule r { X: [M, _, N] X => X }"},
      {"anonymous sequence", "rule r { X: [...] X => X }"},
      {"named sequence", "rule r { X: [Batch...] X => X }"},
      {"leading sequence", "rule r { X: [..., M, K] X => X }"},
      {"middle sequence", "rule r { X: [M, Batch..., K] X => X }"},
      {"trailing sequence", "rule r { X: [M, K, Tail...] X => X }"},
      {"descriptor", "rule r { (dot[@d] X W) => (dot[@d] X W) }"},
      {"nullary operator", "rule r { (zero) => (zero) }"},
      {"root binding", "rule r { ?Y = (dot X W) => ?Y }"},
      {"nested binding",
       "rule r { (add (?Y = (dot X W)) ?Y) => (?Z = (mul ?Y ?Y)) }"},
      {"tuple", "rule r { (get[0] (tuple X Y)) => X }"},
      {"empty sections", "rule r { X => X where {} derive {} }"},
      {"derive only", "rule r { X => (copy[@t] X) derive { @t = infer(X); } }"},
      {"host expressions",
       "rule r { X => X where { f(g(X, @d), ?Y, 4); "
       "K % 128 == 0; -K + +N / 2 != 0; M - 1 < N; "
       "rank(X) >= 2 && !(N <= 0) || true; M > N; } "
       "derive { @out = infer(?Y, @d); } }"},
      {"comments", "// rule\nrule r { /* graph */ X => X } // end\n"},
      {"multiple rules", "rule a { X => X } rule b { Y => Y }"},
  };
  for (const auto& test : cases) {
    auto result = tepl::Parse(test.source);
    Check(result.ok(),
          std::string("Valid syntax rejected: ") + std::string(test.name));
    if (!result.ok()) {
      for (const auto& diagnostic : result.diagnostics) {
        std::cerr << diagnostic.message << '\n';
      }
    }
    Check(!result.tree.empty(), "Successful parse must expose its tree");
  }
}

void TestInvalidSyntax() {
  const Case cases[] = {
      {"empty program", ""},
      {"missing name", "rule { X => X }"},
      {"missing arrow", "rule r { X X }"},
      {"wrong arrow", "rule r { X -> X }"},
      {"missing rhs", "rule r { X => }"},
      {"unclosed rule", "rule r { X => X"},
      {"unclosed graph", "rule r { (add X Y => X }"},
      {"unclosed attribute", "rule r { (dot[@d X W) => X }"},
      {"missing attribute sigil", "rule r { (dot[d] X W) => X }"},
      {"empty binder", "rule r { ? = X => X }"},
      {"empty descriptor", "rule r { (dot[@] X W) => X }"},
      {"two shape sequences", "rule r { X: [A..., B...] X => X }"},
      {"two anonymous sequences", "rule r { X: [..., M, ...] X => X }"},
      {"mixed shape sequences", "rule r { X: [M, A..., _, ...] X => X }"},
      {"shape arithmetic", "rule r { X: [M + 1] X => X }"},
      {"trailing shape comma", "rule r { X: [M,] X => X }"},
      {"missing shape comma", "rule r { X: [M N] X => X }"},
      {"shape in wrong location", "rule r { X => X X: [M] }"},
      {"missing guard terminator", "rule r { X => X where { rank(X) == 2 } }"},
      {"incomplete comparison", "rule r { X => X where { K >=; } }"},
      {"chained comparison", "rule r { X => X where { M < N < K; } }"},
      {"chained equality", "rule r { X => X where { M == N == K; } }"},
      {"trailing call comma", "rule r { X => X where { f(X,); } }"},
      {"assignment in where", "rule r { X => X where { @d = f(X); } }"},
      {"derive without descriptor", "rule r { X => X derive { d = f(X); } }"},
      {"derive without terminator", "rule r { X => X derive { @d = f(X) } }"},
      {"wrong section order", "rule r { X => X derive {} where {} }"},
      {"duplicate section", "rule r { X => X where {} where {} }"},
      {"negative tuple index", "rule r { (get[-1] X) => X }"},
      {"symbolic tuple index", "rule r { (get[N] X) => X }"},
      {"missing tuple operand", "rule r { (get[0]) => X }"},
      {"extra tuple operand", "rule r { (get[0] X Y) => X }"},
      {"variadic operands deferred", "rule r { (concat Xs...) => X }"},
      {"multiple roots deferred", "rule r { match { X Y } rewrite { X } }"},
      {"trailing input", "rule r { X => X } garbage"},
      {"lexical error", "rule r { X => X $ }"},
      {"unterminated comment", "rule r { X => X } /* unfinished"},
  };
  for (const auto& test : cases) {
    auto result = tepl::Parse(test.source);
    Check(!result.ok(),
          std::string("Invalid syntax accepted: ") + std::string(test.name));
    Check(result.tree.empty() && result.rule_count == 0,
          "Failed parse must not expose recovered rules or a tree");
  }
}

void TestDiagnostics() {
  auto lexical = tepl::Parse("rule r {\n X => X $\n}");
  Check(lexical.diagnostics.size() == 1, "Expected one lexical diagnostic");
  if (lexical.diagnostics.size() == 1) {
    Check(
        lexical.diagnostics[0].line == 2 && lexical.diagnostics[0].column == 9,
        "Lexer coordinates must be one-based and point to '$'");
  }
  auto syntax = tepl::Parse("rule r {\n X =>\n}");
  Check(syntax.diagnostics.size() == 1, "Expected one parser diagnostic");
  if (syntax.diagnostics.size() == 1) {
    Check(syntax.diagnostics[0].line == 3 && syntax.diagnostics[0].column == 1,
          "Parser coordinates must point to the unexpected closing brace");
  }
}

void TestPrecedence() {
  const std::string source =
      "rule r { X => X where { K + 2 * N >= 128 && !false || true; } }";
  if (!tepl::Parse(source).ok()) {
    Check(false, "Precedence example must parse");
    return;
  }

  antlr4::ANTLRInputStream input(source);
  tepl_generated::TeplLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  tepl_generated::TeplParser parser(&tokens);
  auto* expression = parser.program()
                         ->ruleDecl(0)
                         ->whereBlock()
                         ->constraintExpr(0)
                         ->logicalOr();
  Check(expression->logicalAnd().size() == 2,
        "Logical OR must be the outermost operation");
  auto* conjunction = expression->logicalAnd(0);
  Check(conjunction->equality().size() == 2,
        "Logical AND must bind more tightly than OR");
  auto* comparison = conjunction->equality(0)->comparison(0);
  Check(comparison->additive().size() == 2,
        "Comparison must bind more tightly than AND");
  auto* sum = comparison->additive(0);
  Check(sum->multiplicative().size() == 2,
        "Addition must bind more tightly than comparison");
  Check(sum->multiplicative(1)->getText() == "2*N" &&
            sum->multiplicative(1)->unary().size() == 2,
        "Multiplication must bind more tightly than addition");
}

}  // namespace

int main() {
  TestValidSyntax();
  TestInvalidSyntax();
  TestDiagnostics();
  TestPrecedence();
  return failures == 0 ? 0 : 1;
}
