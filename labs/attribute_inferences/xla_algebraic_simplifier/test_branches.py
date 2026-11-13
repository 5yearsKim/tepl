#!/usr/bin/env python3
"""Positive and negative branch controls against unmodified XLA behavior."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def module(body, reducer=""):
    return "HloModule branch_control\n" + reducer + "\nENTRY main {\n" + body + "\n}\n"


def controls():
    yield "constant_reciprocal", 2590, module(" a = f32[2] parameter(0)\n c = f32[2] constant({2,3})\n ROOT r = f32[2] divide(a,c)")
    yield "divide_broadcast", 2614, module(" a = f32[2,3] parameter(0)\n b = f32[3] parameter(1)\n c = f32[2,3] broadcast(b), dimensions={1}\n ROOT r = f32[2,3] divide(a,c)")
    yield "multiply_constants", 5132, module(" a = f32[] parameter(0)\n c = f32[] constant(2)\n d = f32[] constant(3)\n b = f32[] multiply(a,c)\n ROOT r = f32[] multiply(b,d)")
    yield "multiply_broadcast_constant", 5165, module(" a = f32[2,3] parameter(0)\n b = f32[3] parameter(1)\n c = f32[] constant(3)\n bc = f32[2,3] broadcast(c), dimensions={}\n bb = f32[2,3] broadcast(b), dimensions={1}\n m = f32[2,3] multiply(a,bc)\n ROOT r = f32[2,3] multiply(m,bb)")
    yield "broadcast_drop_reshape", 5645, module(" a = f32[3] parameter(0)\n b = f32[1,3] reshape(a)\n ROOT r = f32[2,1,3] broadcast(b), dimensions={1,2}")
    yield "scalar_broadcast_user", 5681, module(" a = f32[] parameter(0)\n b = f32[2,3] broadcast(a), dimensions={}\n ROOT r = f32[3,2] reshape(b)")
    yield "sink_broadcast_unary", 6358, module(" a = f32[3] parameter(0)\n b = f32[2,3] broadcast(a), dimensions={1}\n ROOT r = f32[2,3] negate(b)")
    yield "sink_broadcast_binary", 6358, module(" a = f32[3] parameter(0)\n b = f32[3] parameter(1)\n x = f32[2,3] broadcast(a), dimensions={1}\n y = f32[2,3] broadcast(b), dimensions={1}\n ROOT r = f32[2,3] add(x,y)")
    yield "sink_broadcast_clamp", 6358, module(" a = f32[3] parameter(0)\n b = f32[3] parameter(1)\n c = f32[3] parameter(2)\n x = f32[2,3] broadcast(a), dimensions={1}\n y = f32[2,3] broadcast(b), dimensions={1}\n z = f32[2,3] broadcast(c), dimensions={1}\n ROOT r = f32[2,3] clamp(x,y,z)")
    yield "sink_broadcast_dtype_guard", None, module(" a = pred[3] parameter(0)\n b = f32[3] parameter(1)\n c = f32[3] parameter(2)\n x = pred[2,3] broadcast(a), dimensions={1}\n y = f32[2,3] broadcast(b), dimensions={1}\n z = f32[2,3] broadcast(c), dimensions={1}\n ROOT r = f32[2,3] select(x,y,z)")
    yield "transpose_dot", 10187, module(" a = f32[2,3] parameter(0)\n b = f32[3,4] parameter(1)\n d = f32[2,4] dot(a,b), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n ROOT r = f32[4,2] transpose(d), dimensions={1,0}")
    yield "dot_unit_contract", 2742, module(" a = f32[2,1,3] parameter(0)\n b = f32[1,3,4] parameter(1)\n ROOT r = f32[2,4] dot(a,b), lhs_contracting_dims={1,2}, rhs_contracting_dims={0,1}")
    yield "dot_unit_output", 2744, module(" a = f32[1,2,3] parameter(0)\n b = f32[3,4] parameter(1)\n ROOT r = f32[1,2,4] dot(a,b), lhs_contracting_dims={2}, rhs_contracting_dims={0}")
    yield "dot_no_contract", 3785, module(" a = f32[2] parameter(0)\n b = f32[3] parameter(1)\n ROOT r = f32[2,3] dot(a,b), lhs_contracting_dims={}, rhs_contracting_dims={}")
    yield "reshape_broadcast", 6898, module(" a = f32[3] parameter(0)\n b = f32[2,3,4] broadcast(a), dimensions={1}\n ROOT r = f32[1,2,3,4] reshape(b)")
    yield "reshape_iota", 6945, module(" a = s32[6] iota(), iota_dimension=0\n ROOT r = s32[2,3] reshape(a)")
    reducer = "sum {\n a = f32[] parameter(0)\n b = f32[] parameter(1)\n ROOT r = f32[] add(a,b)\n}\n"
    yield "reduce_negate", 8870, module(" a = f32[2,3] parameter(0)\n n = f32[2,3] negate(a)\n z = f32[] constant(0)\n ROOT r = f32[2] reduce(n,z), dimensions={1}, to_apply=sum", reducer)
    yield "reduce_reshape_input", 9023, module(" a = f32[2,3,4] parameter(0)\n x = f32[2,12] reshape(a)\n z = f32[] constant(0)\n ROOT r = f32[2] reduce(x,z), dimensions={1}, to_apply=sum", reducer)
    yield "reduce_reshape_output", 9050, module(" a = f32[2,3,4] parameter(0)\n x = f32[6,4] reshape(a)\n z = f32[] constant(0)\n ROOT r = f32[6] reduce(x,z), dimensions={1}, to_apply=sum", reducer)
    yield "conv_fold_pad", 10505, module(" a = f32[1,4,4,1] parameter(0)\n k = f32[3,3,1,1] parameter(1)\n z = f32[] constant(0)\n p = f32[1,6,6,1] pad(a,z), padding=0_0x1_1x1_1x0_0\n ROOT r = f32[1,4,4,1] convolution(p,k), window={size=3x3}, dim_labels=b01f_01io->b01f")
    yield "conv_fold_pad_signed_window", 10505, module(" a = f32[1,4,4,1] parameter(0)\n k = f32[3,3,1,1] parameter(1)\n z = f32[] constant(0)\n p = f32[1,6,6,1] pad(a,z), padding=0_0x1_1x1_1x0_0\n ROOT r = f32[1,3,3,1] convolution(p,k), window={size=3x3 pad=-1_0x-1_0}, dim_labels=b01f_01io->b01f")
    yield "conv_fold_pad_interior", 10505, module(" a = f32[1,3,3,1] parameter(0)\n k = f32[3,3,1,1] parameter(1)\n z = f32[] constant(0)\n p = f32[1,5,5,1] pad(a,z), padding=0_0x0_0_1x0_0_1x0_0\n ROOT r = f32[1,3,3,1] convolution(p,k), window={size=3x3}, dim_labels=b01f_01io->b01f")
    yield "add_order", 934, module(" a = f32[] parameter(0)\n c = f32[] constant(2)\n ROOT r = f32[] add(c,a)")
    yield "transpose_identity", 10114, module(" a = f32[2,3] parameter(0)\n ROOT r = f32[2,3] transpose(a), dimensions={0,1}")
    yield "and_true", 1405, module(" a = pred[2] parameter(0)\n c = pred[] constant(true)\n b = pred[2] broadcast(c), dimensions={}\n ROOT r = pred[2] and(b,a)")
    yield "uniform_constant", 2311, module(" ROOT r = f32[2,3] constant({{2,2,2},{2,2,2}})")
    yield "iota_constant", 2317, module(" ROOT r = s32[4] constant({0,1,2,3})")
    for name, site, operation in (("divide_one", 2478, "divide"), ("multiply_one", 5025, "multiply")):
        yield name, site, module(f" a = f32[2] parameter(0)\n c = f32[] constant(1)\n b = f32[2] broadcast(c), dimensions={{}}\n ROOT r = f32[2] {operation}(a,b)")
    yield "cancel_negates", 5056, module(" a = f32[2] parameter(0)\n b = f32[2] parameter(1)\n n = f32[2] negate(a)\n m = f32[2] negate(b)\n ROOT r = f32[2] multiply(n,m)")
    yield "tuple_projection", 5389, module(" a = f32[2] parameter(0)\n b = f32[2] parameter(1)\n t = (f32[2],f32[2]) tuple(a,b)\n ROOT r = f32[2] get-tuple-element(t), index=1")
    yield "broadcast_to_reshape", 5605, module(" a = f32[2,3] parameter(0)\n ROOT r = f32[2,1,3] broadcast(a), dimensions={0,2}")
    yield "broadcast_iota", 5690, module(" a = s32[3] iota(), iota_dimension=0\n ROOT r = s32[2,3] broadcast(a), dimensions={1}")
    yield "broadcast_drop_unit", 5723, module(" a = f32[1,3] parameter(0)\n ROOT r = f32[2,1,3] broadcast(a), dimensions={1,2}")
    for name, site, comparison in (("nonnegative_lt", 5761, "LT"), ("nonnegative_ge", 5767, "GE")):
        yield name, site, module(f" a = f32[2] parameter(0)\n x = f32[2] abs(a)\n z = f32[] constant(0)\n b = f32[2] broadcast(z), dimensions={{}}\n ROOT r = pred[2] compare(x,b), direction={comparison}")
    yield "reshape_identity", 6744, module(" a = f32[2,3] parameter(0)\n ROOT r = f32[2,3] reshape(a)")
    yield "reshape_compose", 6750, module(" a = f32[2,3] parameter(0)\n b = f32[6] reshape(a)\n ROOT r = f32[3,2] reshape(b)")
    reducer = "sum {\n a = f32[] parameter(0)\n b = f32[] parameter(1)\n ROOT r = f32[] add(a,b)\n}\n"
    yield "reduce_unit", 8661, module(" a = f32[1,3] parameter(0)\n z = f32[] constant(0)\n ROOT r = f32[3] reduce(a,z), dimensions={0}, to_apply=sum", reducer)
    for name, site, predicate in (("select_true", 9835, "true"), ("select_false", 9840, "false")):
        yield name, site, module(f" a = f32[2] parameter(0)\n b = f32[2] parameter(1)\n c = pred[] constant({predicate})\n p = pred[2] broadcast(c), dimensions={{}}\n ROOT r = f32[2] select(p,a,b)")
    yield "transpose_broadcast", 2824, module(" a = f32[2] parameter(0)\n b = f32[3,2] broadcast(a), dimensions={1}\n ROOT r = f32[2,3] transpose(b), dimensions={1,0}")
    for dtype in ("f32", "bf16", "s32", "pred"):
        # Constant inputs cover integer/predicate transpose without needing the
        # evaluator's parameter generator to make value-domain assumptions.
        yield "transpose_" + dtype, 10118, module(f" a = {dtype}[2,3] parameter(0)\n b = {dtype}[3,2] transpose(a), dimensions={{1,0}}\n ROOT r = {dtype}[2,3] transpose(b), dimensions={{1,0}}")
    yield "no_match", None, module(" a = f32[2] parameter(0)\n b = f32[2] parameter(1)\n ROOT r = f32[2] add(a,b)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
              for key in ("driver", "library", "automatic", "branches")}
    rows = []
    for name, expected, source in controls():
        path = args.out / (name + ".hlo")
        path.write_text(source)
        argv = [str(args.driver.resolve()), "--branches", str(args.library.resolve()),
                str(args.automatic.resolve()), str(args.branches.resolve()), str(path.resolve()),
                str((args.out / name).resolve())]
        process = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        (args.out / (name + ".stdout")).write_text(process.stdout)
        (args.out / (name + ".stderr")).write_text(process.stderr)
        if process.returncode:
            raise RuntimeError(f"{name}: {process.stderr}")
        result = json.loads(process.stdout)
        delegated = {r["source_site"] for r in result["tepl_applications"] if r["applications"]}
        if expected is not None and expected not in delegated:
            raise ValueError(f"{name}: expected actual delegation of AS-{expected}, got {delegated}")
        if expected is None and delegated:
            raise ValueError("no-match control delegated")
        rows.append(dict(case=name, expected_site=expected, **result))
        print(name + ": passed", flush=True)
    if hashes != {key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest() for key in hashes}:
        raise ValueError("experiment binaries changed")
    result = dict(scope="branch substitutions inside AlgebraicSimplifier", binary_sha256=hashes,
                  cases=len(rows), verified_sites=sorted({r["expected_site"] for r in rows if r["expected_site"]}),
                  numerical_input_sets=3 * len(rows), results=rows)
    registry = json.loads((Path(__file__).parent.parent / "xla_workload_count/support-registry.json").read_text())
    required = {int(name[3:]) for name in registry["rules"]}
    covered = {entry["source_site"] for row in rows for entry in row["tepl_applications"] if entry["applications"]}
    if not required <= covered:
        raise ValueError("missing workload-branch controls: " + str(sorted(required - covered)))
    result["all_workload_variants_covered"] = True
    result["covered_workload_variants"] = sorted(required & covered)
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Passed {len(rows)} branch integration controls.")


if __name__ == "__main__":
    main()
