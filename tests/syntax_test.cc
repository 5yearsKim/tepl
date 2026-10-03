#include <iostream>
#include <string>
#include <string_view>

#include "antlr4-runtime.h"
#include "grammar/TeplLexer.h"
#include "grammar/TeplParser.h"
#include "src/parse.h"

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    ++failures;
  }
}

struct Case {
  std::string_view name;
  std::string_view source;
};

void testValidSyntax() {
  const Case cases[] = {
      {"minimal", "rule r { X => X }"},
      {"integer operand", "rule r { (add X 1) => (add 1 X) }"},
      {"float operand", "rule r { (add X 1.0) => (add 1.0 X) }"},
      {"signed numbers", "rule r { (add -12 +0.5) => -0.25 }"},
      {"root literals", "rule r { 1 => 1.0 }"},
      {"bound literal", "rule r { let Y = -1.0 => Y }"},
      {"float constraints",
       "rule r { X => X where { f(1.0, -0.5); 0.25 + 0.5 < 1.0; } "
       "derive { @d = infer(2.0); } }"},
      {"abstract operation",
       "abstract rule commute(F: op<(tensor, tensor) -> tensor>) { "
       "(F X Y) => (F Y X) }"},
      {"abstract host function",
       "abstract rule r(F: op<() -> tensor>, allowed: fn<(tensor) -> bool>) { "
       "(F) => X where { allowed(X); } }"},
      {"abstract empty parameters", "abstract rule r() { X => X }"},
      {"signature named type",
       "abstract rule r(F: op<(custom) -> custom>) { (F X) => X }"},
      {"signature attrs type",
       "abstract rule r(F: fn<(attrs) -> attrs>) { X => X }"},
      {"selected rule imports", "from \"abstract.tepl\" import {a, b};"},
      {"inherited rule", "rule r extends commute(F = t.add);"},
      {"inherited restrictions",
       "rule r extends commute(F = t.add, allowed = $check) { "
       "X: [N] Y: [N] where { N <= 1024; } }"},
      {"inherited empty body", "rule r extends base() {}"},
      {"rank-zero tensor", "rule r { X: [] X => X }"},
      {"mixed declarations",
       "rule r { X: [M, K] S: [] (mul X S) => (mul S X) }"},
      {"scalar is an identifier",
       "rule scalar { scalar: [] scalar => scalar }"},
      {"fixed dimensions", "rule r { X: [M, _, N] X => X }"},
      {"anonymous sequence", "rule r { X: [...] X => X }"},
      {"named sequence", "rule r { X: [Batch...] X => X }"},
      {"leading sequence", "rule r { X: [..., M, K] X => X }"},
      {"middle sequence", "rule r { X: [M, Batch..., K] X => X }"},
      {"trailing sequence", "rule r { X: [M, K, Tail...] X => X }"},
      {"descriptor", "rule r { (dot[@d] X W) => (dot[@d] X W) }"},
      {"nullary operator", "rule r { (zero) => (zero) }"},
      {"root binding", "rule r { let Y = (dot X W) => Y }"},
      {"nested binding", "rule r { (add (let Y = (dot X W)) Y) => (mul Y Y) }"},
      {"tuple", "rule r { (get[0] (tuple X Y)) => X }"},
      {"RHS projection", "rule r { (tuple X Y) => (get[0] (tuple X Y)) }"},
      {"empty sections", "rule r { X => X where {} derive {} }"},
      {"derive only", "rule r { X => (copy[@t] X) derive { @t = infer(X); } }"},
      {"host expressions",
       "rule r { X => X where { $f($g(X, @d), Y, 4); "
       "K % 128 == 0; -K + +N / 2 != 0; M - 1 < N; "
       "$rank(X) >= 2 && !(N <= 0) || true; M > N; } "
       "derive { @out = $infer(Y, @d); } }"},
      {"native and host calls", "rule r { X => X where { f($f(X), g(X)); } }"},
      {"comments", "// rule\nrule r { /* graph */ X => X } // end\n"},
      {"multiple rules", "rule a { X => X } rule b { Y => Y }"},
      {"dialect and import",
       "import \"tensor.tepl\"; dialect tensor { "
       "attrs CollectiveReduce { kind: string; } "
       "op add(lhs: tensor, rhs: tensor) -> tensor; "
       "op dot_general(lhs: tensor, rhs: tensor) -> tensor { "
       "alias: dot; attrs { axes: index[] = []; } } "
       "op all_reduce(input: tensor) -> tensor { "
       "attrs: CollectiveReduce; } "
       "op concat(first: tensor, rest: tensor...) -> tensor; }"},
      {"named dialect import",
       "from \"tensor.tepl\" import TensorLang as t; "
       "use t::{add, dot}; rule r { (t.dot[@d] X Y) => (add X Y) }"},
      {"attributes before alias",
       "dialect t { op dot_general(lhs: tensor, rhs: tensor) -> tensor { "
       "attrs { axes: index[] = []; } alias: dot; } }"},
      {"shared attributes before alias",
       "dialect t { attrs Metadata { kind: string; } "
       "op dot_general(lhs: tensor, rhs: tensor) -> tensor { "
       "attrs: Metadata; alias: dot; } }"},
      {"open dialect",
       "from \"tensor.tepl\" import TensorLang as t; "
       "use t; rule r { (add X Y) => (add Y X) }"},
  };
  for (const auto& test : cases) {
    auto result = tepl::parse(test.source);
    check(result.ok(),
          std::string("Valid syntax rejected: ") + std::string(test.name));
    if (!result.ok()) {
      for (const auto& diagnostic : result.diagnostics) {
        std::cerr << diagnostic.message << '\n';
      }
    }
    check(!result.tree.empty(), "Successful parse must expose its tree");
  }
}

void testInvalidSyntax() {
  const Case cases[] = {
      {"bare RHS binding", "rule r { X => let Y = (negate X) }"},
      {"parenthesized RHS binding", "rule r { X => (let Y = (negate X)) }"},
      {"nested RHS binding", "rule r { X => (add X (let Y = X)) }"},
      {"RHS binding in projection", "rule r { X => (get[0] (let Y = X)) }"},
      {"abstract RHS binding", "abstract rule r() { X => (let Y = X) }"},

      {"empty program", ""},
      {"missing fractional digits", "rule r { (add X 1.) => X }"},
      {"missing integer digits", "rule r { (add X .5) => X }"},
      {"malformed decimal", "rule r { (add X 1.2.3) => X }"},
      {"unsupported exponent", "rule r { (add X 1e3) => X }"},
      {"unsupported float suffix", "rule r { (add X 1.0f) => X }"},
      {"missing number after sign", "rule r { (add X -) => X }"},
      {"float tuple index", "rule r { (get[1.0] X) => X }"},
      {"number in shape", "rule r { X: [1.0] X => X }"},
      {"abstract missing parameters", "abstract rule r { X => X }"},
      {"concrete parameters", "rule r(F: op<(tensor) -> tensor>) { X => X }"},
      {"unknown parameter kind", "abstract rule r(F: tensor) { X => X }"},
      {"signature missing arrow",
       "abstract rule r(F: op<(tensor) tensor>) { X => X }"},
      {"signature trailing comma",
       "abstract rule r(F: op<(tensor,) -> tensor>) { X => X }"},
      {"signature missing result",
       "abstract rule r(F: fn<(tensor) ->>) { X => X }"},
      {"parameter trailing comma",
       "abstract rule r(F: op<() -> tensor>,) { X => X }"},
      {"abstract inherited rule", "abstract rule r() extends base();"},
      {"inherited missing terminator", "rule r extends base()"},
      {"inherited positional binding", "rule r extends base(t.add);"},
      {"inherited expression binding", "rule r extends base(F = check(X));"},
      {"inherited trailing comma", "rule r extends base(F = t.add,);"},
      {"inherited graph replacement", "rule r extends base() { X => X }"},
      {"inherited derivation",
       "rule r extends base() { derive { @d = f(); } }"},
      {"empty selected rule imports", "from \"abstract.tepl\" import {};"},
      {"selected rule import trailing comma",
       "from \"abstract.tepl\" import {a,};"},
      {"missing name", "rule { X => X }"},
      {"missing arrow", "rule r { X X }"},
      {"wrong arrow", "rule r { X -> X }"},
      {"missing rhs", "rule r { X => }"},
      {"unclosed rule", "rule r { X => X"},
      {"unclosed graph", "rule r { (add X Y => X }"},
      {"unclosed attribute", "rule r { (dot[@d X W) => X }"},
      {"missing attribute sigil", "rule r { (dot[d] X W) => X }"},
      {"empty binder", "rule r { let = X => X }"},
      {"legacy binder sigil", "rule r { ?Y = X => Y }"},
      {"empty descriptor", "rule r { (dot[@] X W) => X }"},
      {"two shape sequences", "rule r { X: [A..., B...] X => X }"},
      {"two anonymous sequences", "rule r { X: [..., M, ...] X => X }"},
      {"mixed shape sequences", "rule r { X: [M, A..., _, ...] X => X }"},
      {"shape arithmetic", "rule r { X: [M + 1] X => X }"},
      {"trailing shape comma", "rule r { X: [M,] X => X }"},
      {"missing shape comma", "rule r { X: [M N] X => X }"},
      {"shape in wrong location", "rule r { X => X X: [M] }"},
      {"unknown scalar marker", "rule r { S: scalr S => S }"},
      {"quoted scalar marker", "rule r { S: \"scalar\" S => S }"},
      {"removed scalar declaration", "rule r { S: scalar S => S }"},
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
      {"unnamed operand", "dialect t { op add(tensor) -> tensor; }"},
      {"variadic not last",
       "dialect t { op bad(xs: tensor..., last: tensor) -> tensor; }"},
      {"missing attribute type",
       "dialect t { op bad(x: tensor) -> tensor { attrs { axis:; } } }"},
      {"duplicate aliases",
       "dialect t { op bad(x: tensor) -> tensor { "
       "alias: one; alias: two; } }"},
      {"duplicate inline attributes",
       "dialect t { op bad(x: tensor) -> tensor { "
       "attrs { axis: index; } attrs { shape: index[]; } } }"},
      {"shared and inline attributes",
       "dialect t { attrs Metadata { kind: string; } "
       "op bad(x: tensor) -> tensor { "
       "attrs: Metadata; attrs { axis: index; } } }"},
      {"duplicate shared attributes",
       "dialect t { attrs First {} attrs Second {} "
       "op bad(x: tensor) -> tensor { "
       "attrs: First; attrs: Second; } }"},
      {"named import missing dialect", "from \"tensor.tepl\" import;"},
      {"empty selected use", "use t::{}; rule r { X => X }"},
      {"selected use missing brace", "use t::{add; rule r { X => X }"},
      {"unknown operation property",
       "dialect D { op x() -> tensor { typo() {} } }"},
      {"unclosed shape block",
       "dialect D { op x() -> tensor { shape() { yield []; } }"},
      {"lexical error", "rule r { X => X # }"},
      {"missing host name", "rule r { X => X where { $(X); } }"},
      {"duplicate host sigil", "rule r { X => X where { $$f(X); } }"},
      {"host reference without call", "rule r { X => X where { $f; } }"},
      {"qualified host name", "rule r { X => X where { $t.f(X); } }"},
      {"host graph operator", "rule r { ($f X) => X }"},
      {"unterminated comment", "rule r { X => X } /* unfinished"},
  };
  for (const auto& test : cases) {
    auto result = tepl::parse(test.source);
    check(!result.ok(),
          std::string("Invalid syntax accepted: ") + std::string(test.name));
    check(result.tree.empty() && result.rule_count == 0,
          "Failed parse must not expose recovered rules or a tree");
  }
}

void testDiagnostics() {
  auto lexical = tepl::parse("rule r {\n X => X #\n}");
  check(lexical.diagnostics.size() == 1, "Expected one lexical diagnostic");
  if (lexical.diagnostics.size() == 1) {
    check(
        lexical.diagnostics[0].line == 2 && lexical.diagnostics[0].column == 9,
        "Lexer coordinates must be one-based and point to '#'");
  }
  auto syntax = tepl::parse("rule r {\n X =>\n}");
  check(syntax.diagnostics.size() == 1, "Expected one parser diagnostic");
  if (syntax.diagnostics.size() == 1) {
    check(syntax.diagnostics[0].line == 3 && syntax.diagnostics[0].column == 1,
          "Parser coordinates must point to the unexpected closing brace");
  }
}

void testPrecedence() {
  const std::string source =
      "rule r { X => X where { K + 2 * N >= 128 && !false || true; } }";
  if (!tepl::parse(source).ok()) {
    check(false, "Precedence example must parse");
    return;
  }

  antlr4::ANTLRInputStream input(source);
  tepl_generated::TeplLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  tepl_generated::TeplParser parser(&tokens);
  auto* expression = parser.program()
                         ->ruleDecl(0)
                         ->rewriteBody()
                         ->whereBlock()
                         ->constraintExpr(0)
                         ->logicalOr();
  check(expression->logicalAnd().size() == 2,
        "Logical OR must be the outermost operation");
  auto* conjunction = expression->logicalAnd(0);
  check(conjunction->equality().size() == 2,
        "Logical AND must bind more tightly than OR");
  auto* comparison = conjunction->equality(0)->comparison(0);
  check(comparison->additive().size() == 2,
        "Comparison must bind more tightly than AND");
  auto* sum = comparison->additive(0);
  check(sum->multiplicative().size() == 2,
        "Addition must bind more tightly than comparison");
  check(sum->multiplicative(1)->getText() == "2*N" &&
            sum->multiplicative(1)->unary().size() == 2,
        "Multiplication must bind more tightly than addition");
}

}  // namespace

int main() {
  testValidSyntax();
  testInvalidSyntax();
  testDiagnostics();
  testPrecedence();
  return failures == 0 ? 0 : 1;
}
