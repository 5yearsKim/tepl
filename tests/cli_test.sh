#!/usr/bin/env bash
set -euo pipefail

tepl="${TEST_SRCDIR}/${TEST_WORKSPACE}/$1"
example="${TEST_SRCDIR}/${TEST_WORKSPACE}/$2"
output="${TEST_TMPDIR}/output"

check_exit() {
  local expected="$1"
  shift
  local actual=0
  "$tepl" "$@" >"$output" 2>&1 || actual=$?
  if [[ "$actual" != "$expected" ]]; then
    cat "$output" >&2
    echo "Expected exit $expected, got $actual for: $*" >&2
    exit 1
  fi
}

check_exit 0 --version
grep -Fxq '0.2.0' "$output"

check_exit 0 --help
grep -Fq 'parse' "$output"
grep -Fq 'init' "$output"
check_exit 0 -h
check_exit 2 check "$example" --suggest-where
check_exit 2 check "$example" --certified

# Initialization creates a usable project without relying on external templates.
check_exit 0 init --help
grep -Fq 'dir' "$output"
check_exit 0 init -h
check_exit 2 init
check_exit 2 init ""
check_exit 2 init "${TEST_TMPDIR}/unused" extra
check_exit 2 init "${TEST_TMPDIR}/unused" --unknown
test ! -e "${TEST_TMPDIR}/unused"
starter="${TEST_TMPDIR}/new parent/starter project"
check_exit 0 init "$starter"
grep -Fq 'Initialized TEPL project in' "$output"
grep -Fq 'tepl check' "$output"
test -f "$starter/dialects/your_dialect.tepl"
test -f "$starter/rules/your_rule.tepl"
grep -Fxq 'tepl-version = "0.2.0"' "$starter/tepl.toml"
grep -Fq 'from "dialects/your_dialect.tepl"' "$starter/rules/your_rule.tepl"
check_exit 0 check "$starter"
grep -Fq 'MatchOp(YourDialect.add' "$output"
check_exit 0 check "$starter/rules/your_rule.tepl"
check_exit 0 parse "$starter/rules/your_rule.tepl"
# Existing directories and unrelated files are preserved; '.' and relative paths work.
init_current="${TEST_TMPDIR}/init_current"
mkdir -p "$init_current/dialects" "$init_current/rules"
printf 'keep me\n' > "$init_current/notes.txt"
(
  cd "$init_current"
  check_exit 0 init .
  check_exit 0 check .
  check_exit 0 init "relative project"
)
grep -Fxq 'keep me' "$init_current/notes.txt"
cmp "$starter/dialects/your_dialect.tepl" "$init_current/dialects/your_dialect.tepl"
cmp "$starter/rules/your_rule.tepl" "$init_current/rules/your_rule.tepl"
cmp "$starter/tepl.toml" "$init_current/tepl.toml"
check_exit 0 check "$init_current/relative project"

# All starter destinations are checked before any are written.
for part in dialect rule; do
  conflict="${TEST_TMPDIR}/init_conflict_${part}"
  mkdir -p "$conflict/${part}s"
  printf 'user content\n' > "$conflict/${part}s/your_${part}.tepl"
  check_exit 2 init "$conflict"
  grep -Fq 'refusing to overwrite' "$output"
  grep -Fxq 'user content' "$conflict/${part}s/your_${part}.tepl"
  test ! -e "$conflict/tepl.toml"
  if [[ "$part" == dialect ]]; then
    test ! -e "$conflict/rules"
  else
    test ! -e "$conflict/dialects"
  fi
done
conflict="${TEST_TMPDIR}/init_conflict_config"
mkdir -p "$conflict"
printf '# user config\n' > "$conflict/tepl.toml"
check_exit 2 init "$conflict"
grep -Fq 'refusing to overwrite' "$output"
grep -Fxq '# user config' "$conflict/tepl.toml"
test ! -e "$conflict/dialects"
test ! -e "$conflict/rules"
check_exit 2 init "$starter"
cmp "$starter/dialects/your_dialect.tepl" "$init_current/dialects/your_dialect.tepl"
cmp "$starter/rules/your_rule.tepl" "$init_current/rules/your_rule.tepl"

# Directories and dangling symlinks at starter filenames also count as conflicts.
conflict="${TEST_TMPDIR}/init_file_directory"
mkdir -p "$conflict/rules/your_rule.tepl"
check_exit 2 init "$conflict"
test ! -e "$conflict/dialects"

# Config destinations receive the same directory and symlink protection.
for kind in directory symlink; do
  conflict="${TEST_TMPDIR}/init_config_${kind}"
  mkdir -p "$conflict"
  if [[ "$kind" == directory ]]; then
    mkdir "$conflict/tepl.toml"
  else
    ln -s "${TEST_TMPDIR}/missing_config_target" "$conflict/tepl.toml"
  fi
  check_exit 2 init "$conflict"
  test ! -e "$conflict/dialects"
  test ! -e "$conflict/rules"
done

# Root imports work from nested sources, independently of the invocation cwd.
mkdir -p "$starter/rules/nested" "$starter/graphs"
cp "$starter/rules/your_rule.tepl" "$starter/rules/nested/commute.tepl"
(
  cd "$init_current"
  check_exit 0 check "$starter/rules/nested/commute.tepl"
)
printf 'from "dialects/your_dialect.tepl" import YourDialect as d graph g { input X input Y yield (d.add X Y) }\n' > "$starter/graphs/g.tepl"
check_exit 0 compile "$starter/graphs/g.tepl" --rules "$starter/rules/nested" -o "${TEST_TMPDIR}/root-imports.json"
check_exit 0 validate "${TEST_TMPDIR}/root-imports.json"
check_exit 0 generate "$starter/graphs/g.tepl" --rules "$starter/rules/nested" --no-format --out "${TEST_TMPDIR}/root-imports-generated"

# Configured source directories and exclusions work for all source commands.
configured="${TEST_TMPDIR}/configured project"
mkdir -p "$configured/schema" "$configured/patterns/nested" \
  "$configured/patterns/experimental" "$configured/patterns/fixtures/deep" \
  "$configured/programs" "$configured/fixtures" "$configured/rules"
cat > "$configured/tepl.toml" <<'TOML'
tepl-version = "0.2.0"
[sources]
dialects = ["schema"]
rules = ["patterns", "patterns/nested"]
graphs = ["programs", "fixtures"]
exclude = ["patterns/experimental/**", "**/fixtures/**", "patterns/*.skip.tepl", "patterns/ignored?.tepl"]
[check]
semantic-policy = "strict"
TOML
printf 'dialect D { op negate(x:tensor)->tensor }\n' > "$configured/schema/d.tepl"
printf 'from "schema/d.tepl" import D as d rule negate { (d.negate X) => X }\n' > "$configured/patterns/nested/negate.tepl"
printf 'from "schema/d.tepl" import D as d graph g { input X yield (d.negate X) }\n' > "$configured/programs/g.tepl"
for file in patterns/experimental/bad.tepl patterns/fixtures/deep/bad.tepl \
  fixtures/bad.tepl patterns/bad.skip.tepl patterns/ignored1.tepl rules/unselected.tepl; do
  printf 'invalid source\n' > "$configured/$file"
done
(
  cd "$init_current"
  check_exit 0 check "$configured"
  check_exit 0 compile "$configured" -o "${TEST_TMPDIR}/configured.json"
  check_exit 0 generate "$configured" --no-format --out "${TEST_TMPDIR}/configured-generated"
  check_exit 0 rewrite "$configured"
  grep -Fq 'yield X' "$output"
  check_exit 0 compile "$configured" --rules "$configured/patterns" -o "${TEST_TMPDIR}/selected-rules.json"
  check_exit 0 rewrite "$configured/programs" --rules "$configured/patterns"
  grep -Fq 'yield X' "$output"
)
# Explicit files and imported dependencies still load when excluded from discovery.
check_exit 1 parse "$configured/patterns/experimental/bad.tepl"
printf 'dialect Extra { op identity(x:tensor)->tensor }\n' > "$configured/patterns/experimental/imported.tepl"
printf 'from "patterns/experimental/imported.tepl" import Extra\n' > "$configured/schema/import.tepl"
check_exit 0 check "$configured"
# An empty array disables discovery for that category, with other defaults intact.
cat > "$configured/tepl.toml" <<'TOML'
[sources]
dialects = []
rules = []
graphs = ["programs"]
TOML
check_exit 0 check "$configured"
# Empty projects must still validate their config instead of hiding config errors.
printf '[sources]\nrules = "patterns"\n' > "$configured/tepl.toml"
check_exit 1 check "$configured"
grep -Fq 'sources.rules must be an array' "$output"
printf '[check]\nsemantic-policy = "fast"\n' > "$configured/tepl.toml"
check_exit 1 check "$configured/programs/g.tepl"
grep -Fq "check.semantic-policy must be 'strict' or 'algebraic'" "$output"
printf '[sources]\nrulse = []\n' > "$configured/tepl.toml"
check_exit 1 check "$configured"
grep -Fq "unknown project configuration key 'sources.rulse'" "$output"

# Nearest configs win, including an empty marker, and relative imports stay local.
nested="$starter/nested_project"
check_exit 0 init "$nested"
printf '# empty marker\n' > "$nested/tepl.toml"
printf 'dialect Local { op negate(x: tensor) -> tensor }\n' > "$nested/dialects/local.tepl"
printf 'from "dialects/local.tepl" import Local as d rule local { (d.negate X) => X }\n' > "$nested/rules/local.tepl"
check_exit 0 check "$nested/rules/local.tepl"
printf 'from "./local.tepl" import Local\n' > "$nested/dialects/relative.tepl"
check_exit 0 check "$nested/dialects/relative.tepl"
printf 'from "../dialects/local.tepl" import Local as d rule local { (d.negate X) => X }\n' > "$nested/rules/relative.tepl"
check_exit 0 check "$nested/rules/relative.tepl"
# A bare path never falls back to the importing directory in a configured project.
printf 'from "local.tepl" import Local\n' > "$nested/dialects/no_fallback.tepl"
check_exit 1 parse "$nested/dialects/no_fallback.tepl"
grep -Fq "cannot open import 'local.tepl'" "$output"
# Filesystem absolute paths continue to work in a configured project.
printf 'from "%s" import Local\n' "$nested/dialects/local.tepl" > "$nested/dialects/absolute.tepl"
check_exit 0 check "$nested/dialects/absolute.tepl"

# Config errors include the config location and fail even for standalone sources.
printf 'rule identity { X => X }\n' > "$nested/rules/standalone.tepl"
printf 'tepl-version = "99.0.0"\n' > "$nested/tepl.toml"
check_exit 1 check "$nested/rules/standalone.tepl"
grep -Fq "$nested/tepl.toml:1:" "$output"
grep -Fq "unsupported tepl-version '99.0.0'" "$output"
printf 'tepl-version = 123\n' > "$nested/tepl.toml"
check_exit 1 parse "$nested/rules/standalone.tepl"
grep -Fq 'tepl-version must be a string' "$output"
printf 'tepl-version = [\n' > "$nested/tepl.toml"
check_exit 1 parse "$nested/rules/standalone.tepl"
grep -Fq 'invalid tepl.toml' "$output"
rm "$nested/tepl.toml"
ln -s "${TEST_TMPDIR}/missing_config_target" "$nested/tepl.toml"
check_exit 1 parse "$nested/rules/standalone.tepl"
grep -Fq 'cannot read project config' "$output"
rm "$nested/tepl.toml"
mkdir "$nested/tepl.toml"
check_exit 1 parse "$nested/rules/standalone.tepl"
grep -Fq 'cannot read project config' "$output"
conflict="${TEST_TMPDIR}/init_file_symlink"
mkdir -p "$conflict/rules"
ln -s "${TEST_TMPDIR}/missing_link_target" "$conflict/rules/your_rule.tepl"
check_exit 2 init "$conflict"
test -L "$conflict/rules/your_rule.tepl"
test ! -e "${TEST_TMPDIR}/missing_link_target"
test ! -e "$conflict/dialects"

# Files blocking required directories fail before writing any samples.
printf 'keep me\n' > "${TEST_TMPDIR}/init_blocked"
check_exit 2 init "${TEST_TMPDIR}/init_blocked"
check_exit 2 init "${TEST_TMPDIR}/init_blocked/child"
grep -Fxq 'keep me' "${TEST_TMPDIR}/init_blocked"
for part in dialects rules; do
  blocked="${TEST_TMPDIR}/init_blocked_${part}"
  mkdir -p "$blocked"
  printf 'keep me\n' > "$blocked/$part"
  check_exit 2 init "$blocked"
  grep -Fxq 'keep me' "$blocked/$part"
  test ! -e "$blocked/dialects/your_dialect.tepl"
  test ! -e "$blocked/rules/your_rule.tepl"
done

check_exit 0 parse --help
grep -Fq -- '--tree' "$output"
grep -Fq -- '--ast' "$output"
check_exit 0 parse -h

check_exit 0 parse "$example"
grep -Fxq 'Parsed 1 rule(s).' "$output"
check_exit 0 parse "$example" --tree
grep -Fq '(program' "$output"
cp "$output" "${TEST_TMPDIR}/tree"
check_exit 0 parse --tree "$example"
cmp "$output" "${TEST_TMPDIR}/tree"
check_exit 0 parse "$example" --ast
grep -Fq '  rule commute_add' "$output"
grep -Fq '      operator add' "$output"
grep -Fq '  from "dialects/tensor.tepl" import TensorLang as t' "$output"
grep -Fq '  use t::{add}' "$output"
grep -Fq '  dialect TensorLang' "$output"
cp "$output" "${TEST_TMPDIR}/ast"
check_exit 0 parse --ast "$example"
cmp "$output" "${TEST_TMPDIR}/ast"

check_exit 2
check_exit 2 parse
check_exit 2 unknown "$example"
check_exit 2 host-template "$example"
check_exit 2 parse "$example" --impl
check_exit 2 parse "$example" --unknown
check_exit 2 parse "$example" --tree --ast
check_exit 2 parse "$example" extra
check_exit 2 parse "${TEST_TMPDIR}/missing.tepl"
grep -Fq 'cannot open file' "$output"

invalid="${TEST_TMPDIR}/invalid.tepl"
printf 'rule r { X => }\n' >"$invalid"
check_exit 1 parse "$invalid"
grep -Fq "${invalid}:1:" "$output"

missing_import="${TEST_TMPDIR}/missing_import.tepl"
printf 'import "missing.tepl" rule r { X => X }\n' >"$missing_import"
check_exit 1 parse "$missing_import"
grep -Fq "cannot open import 'missing.tepl'" "$output"

bad_arity="${TEST_TMPDIR}/bad_arity.tepl"
printf 'import "%s" rule r { (add X) => X }\n' "$example" >"$bad_arity"
# An import must contain dialect declarations only.
check_exit 1 parse "$bad_arity"
grep -Fq 'contains rules' "$output"

dialect="${TEST_SRCDIR}/${TEST_WORKSPACE}/tests/fixtures/language/dialects/tensor.tepl"
printf 'from "%s" import TensorLang as t use t rule r { (multiply X Y) => (multiply Y X) }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

printf 'from "%s" import TensorLang as t rule r { (t.add X Y) => (t.add Y X) }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

printf 'from "%s" import TensorLang as t rule r { (d: t.dot X Y) => (t.dot[d.attrs] X Y) }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

# Parsing preserves unresolved operation names; check resolves them through core.
printf 'from "%s" import TensorLang as t rule r { (missing.add X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity" --ast
grep -Fq 'operator missing.add' "$output"
check_exit 1 check "$bad_arity"
grep -Fq "unknown operation 'missing.add'" "$output"

printf 'from "%s" import TensorLang use TensorLang::{add} rule r { (add X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

printf 'from "%s" import TensorLang as t use t::{add} rule r { (multiply X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown operation 'multiply'" "$output"

printf 'from "%s" import TensorLang as t use t::{missing} rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown operation 'missing' in dialect 't'" "$output"

printf 'from "%s" import Missing as t use t rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "dialect 'Missing' is not defined" "$output"

printf 'from "%s" import TensorLang as t use missing rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown dialect alias 'missing'" "$output"

printf 'import "%s" rule r { (add X) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq 'expects exactly 2 operands' "$output"

other="${TEST_TMPDIR}/other.tepl"
printf 'dialect Other { op add(lhs: tensor, rhs: tensor) -> tensor }\n' >"$other"
printf 'from "%s" import TensorLang as t from "other.tepl" import Other as o use t::{add} use o::{add} rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "ambiguous operation 'add'" "$output"

same_name="${TEST_TMPDIR}/same_name.tepl"
printf 'dialect TensorLang { op add(lhs: tensor, rhs: tensor) -> tensor }\n' >"$same_name"
printf 'from "%s" import TensorLang as t from "same_name.tepl" import TensorLang as other rule r { (t.add X Y) => (other.add X Y) }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

duplicate="${TEST_TMPDIR}/duplicate.tepl"
printf 'dialect Duplicate {} dialect Duplicate {}\n' >"$duplicate"
printf 'from "duplicate.tepl" import Duplicate as d rule r { X => X }\n' >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "duplicate dialect 'Duplicate'" "$output"

printf 'import "%s" rule r { (dot X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 0 check "$bad_arity"

printf 'import "%s" rule r { (missing X) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown operation 'missing'" "$output"

printf 'dialect t { op bad(x: tensor) -> tensor { attrs { axis: mystery } } }\n' >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown attribute type 'mystery'" "$output"

printf 'dialect t { op bad(x: tensor) -> tensor { attrs { kind: symbol } } }\n' >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown attribute type 'symbol'" "$output"

cycle_a="${TEST_TMPDIR}/a.tepl"
cycle_b="${TEST_TMPDIR}/b.tepl"
printf 'import "b.tepl" rule r { X => X }\n' >"$cycle_a"
printf 'import "a.tepl"\n' >"$cycle_b"
check_exit 1 parse "$cycle_a"
grep -Fq 'cyclic import' "$output"

abstract="${TEST_SRCDIR}/${TEST_WORKSPACE}/tests/fixtures/language/rules/abstract.tepl"
inherited="${TEST_SRCDIR}/${TEST_WORKSPACE}/tests/fixtures/language/rules/inherited.tepl"
check_exit 0 parse "$abstract"
grep -Fxq 'Parsed 3 rule(s).' "$output"
check_exit 0 parse "$inherited"
grep -Fxq 'Parsed 7 rule(s).' "$output"
check_exit 0 parse "$inherited" --ast
grep -Fq 'parameter F: op<(tensor, tensor) -> tensor>' "$output"
grep -Fq 'extends associate_right' "$output"

printf 'from "%s" import {missing}\n' "$abstract" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "rule 'missing'" "$output"

printf 'from "%s" import {commute_add}\n' "$example" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq 'must be abstract' "$output"

template="${TEST_TMPDIR}/template.tepl"
printf 'abstract rule a() { X => X } abstract rule b() { Y => Y }\n' >"$template"
printf 'from "template.tepl" import {a} from "template.tepl" import {b}\n' >"$bad_arity"
check_exit 0 parse "$bad_arity" --ast
grep -Fq 'abstract rule a' "$output"
grep -Fq 'abstract rule b' "$output"

printf 'from "template.tepl" import {a}\n' >"$bad_arity"
check_exit 0 parse "$bad_arity" --ast
grep -Fq 'abstract rule a' "$output"
if grep -Fq 'abstract rule b' "$output"; then
  echo 'Unselected template leaked into the AST' >&2
  exit 1
fi

printf 'abstract rule a() { X => X } abstract rule a() { Y => Y }\n' >"$template"
check_exit 1 parse "$bad_arity"
grep -Fq "duplicate rule 'a'" "$output"

printf 'from "b.tepl" import {b} abstract rule a() { X => X }\n' >"$cycle_a"
printf 'from "a.tepl" import {a} abstract rule b() { X => X }\n' >"$cycle_b"
check_exit 1 parse "$cycle_a"
grep -Fq 'cyclic import' "$output"

# Parse preserves dtype annotations; core validates them through check.
basic="${TEST_SRCDIR}/${TEST_WORKSPACE}/tests/fixtures/language/rules/basic.tepl"
check_exit 0 parse "$basic" --ast
grep -Fq 'tensor X f32[N]' "$output"
grep -Fq 'float 1.0:f32' "$output"
grep -Fq 'integer 1:i32' "$output"
check_exit 0 check "$basic"
grep -Fq 'same_dtype' "$output"

bad_dtype="${TEST_TMPDIR}/bad_dtype.tepl"
printf 'rule r { X: float32[N] X => X }\n' >"$bad_dtype"
check_exit 0 parse "$bad_dtype" --ast
grep -Fq 'tensor X float32[N]' "$output"
check_exit 1 check "$bad_dtype"
grep -Fq "${bad_dtype}:1:10: unknown dtype 'float32'" "$output"
printf 'rule r { X => 256:u8 }\n' >"$bad_dtype"
check_exit 0 parse "$bad_dtype"
check_exit 1 check "$bad_dtype"
grep -Fq "invalid for dtype 'u8'" "$output"

# Semantic analysis emits the checked IR and rejects invalid rules.
check_exit 0 check --help
check_exit 2 check
check_exit 2 check "${TEST_TMPDIR}/missing.tepl"
check_exit 0 check "$example"
grep -Fq 'CheckedProgram' "$output"
grep -Fq 'MatchOp(TensorLang.add' "$output"
grep -Fq 'BuildOp(TensorLang.add' "$output"
check_exit 0 check "$inherited"
grep -Fq 'commute_small_vectors' "$output"
grep -Fq 'expanded at' "$output"
check_exit 1 check "$bad_dtype"
grep -Fq "invalid for dtype 'u8'" "$output"
printf 'rule r { X => missing }\n' >"$invalid"
check_exit 1 check "$invalid"
grep -Fq "unknown RHS capture 'missing'" "$output"
grep -Fq "${invalid}:1:" "$output"

# Rust generation is active; unimplemented targets do not write output.
check_exit 0 generate "$example" --no-format --out "${TEST_TMPDIR}/generated"
test -f "${TEST_TMPDIR}/generated/mod.rs"
check_exit 0 generate "$example" --no-format --check --out "${TEST_TMPDIR}/generated"
check_exit 0 generate "$example" --target cpp --cpp-namespace app::ir --no-format --out "${TEST_TMPDIR}/generated-cpp"
test -f "${TEST_TMPDIR}/generated-cpp/generated.h"
grep -Fq 'namespace app::ir' "${TEST_TMPDIR}/generated-cpp/program.h"
check_exit 0 generate "$example" --target cpp --cpp-namespace app::ir --no-format --check --out "${TEST_TMPDIR}/generated-cpp"
check_exit 1 generate "$example" --target cpp --cpp-namespace std --no-format --out "${TEST_TMPDIR}/invalid-cpp"
test ! -e "${TEST_TMPDIR}/invalid-cpp"
check_exit 1 generate "$example" --target python --no-format --out "${TEST_TMPDIR}/deferred"
grep -Fq 'python code generation is not implemented' "$output"
test ! -e "${TEST_TMPDIR}/deferred"
