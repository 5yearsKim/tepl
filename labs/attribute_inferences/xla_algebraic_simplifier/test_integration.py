"""Run real XLA pass comparisons; requires an explicitly built driver."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    yield "no_match", """HloModule no_match
ENTRY main {
 a = f32[2,3] parameter(0)
 ROOT result = f32[3,2] transpose(a), dimensions={1,0}
}""", False
    yield "identity", """HloModule identity
ENTRY main {
 a = f32[2,3] parameter(0)
 ROOT result = f32[2,3] transpose(a), dimensions={0,1}
}""", True
    for rank in (2, 3):
        dims = list(range(2, rank + 2))
        permutations = [p for p in itertools.permutations(range(rank)) if p != tuple(range(rank))]
        for p in permutations:
            for q in permutations:
                def shape(xs):
                    return "f32[" + ",".join(map(str, xs)) + "]"
                def axes(xs):
                    return "{" + ",".join(map(str, xs)) + "}"
                middle = [dims[i] for i in p]
                final = [middle[i] for i in q]
                name = f"compose_{rank}_{''.join(map(str,p))}_{''.join(map(str,q))}"
                yield name, (f"HloModule {name}\nENTRY main {{\n"
                             f" a = {shape(dims)} parameter(0)\n"
                             f" t = {shape(middle)} transpose(a), dimensions={axes(p)}\n"
                             f" ROOT result = {shape(final)} transpose(t), dimensions={axes(q)}\n}}"), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    binaries = {key: sha(getattr(args, key)) for key in ("driver", "library", "program")}
    rows = []
    for name, source, should_delegate in cases():
        path = args.out / (name + ".hlo")
        path.write_text(source + "\n")
        argv = [str(p.resolve()) for p in (args.driver, args.library, args.program, path)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        (args.out / (name + ".stdout")).write_text(result.stdout)
        (args.out / (name + ".stderr")).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{name}: {result.stderr}")
        data = json.loads(result.stdout)
        applications = sum(r["applications"] for r in data["tepl_applications"])
        if bool(applications) != should_delegate or not data["graph_equal"] or data["numerical_input_sets"] != 3:
            raise ValueError("incorrect integration evidence: " + name)
        rows.append(dict(case=name, input_sha256=sha(path), **data))
        print(name + ": passed", flush=True)
    if binaries != {key: sha(getattr(args, key)) for key in binaries}:
        raise ValueError("binaries changed during evaluation")
    summary = dict(scope="two transpose rules in AlgebraicSimplifier; pass-level comparison",
                   cases=len(rows), numerical_input_sets=3 * len(rows),
                   binary_sha256=binaries, results=rows)
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Passed {len(rows)} XLA integration cases.")


if __name__ == "__main__":
    main()
