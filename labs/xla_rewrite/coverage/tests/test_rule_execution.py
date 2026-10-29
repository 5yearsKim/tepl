"""Execute representative compiled rules and compare independent dense meanings."""

import itertools
import json
import math
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from _support import HERE, TEPL
ROOT = HERE.parents[2]
COMPILER = ROOT / "bazel-bin/tepl"


def coords(shape):
    return itertools.product(*(range(d) for d in shape))


@dataclass
class Tensor:
    shape: tuple
    values: tuple

    @classmethod
    def build(cls, shape, value):
        shape = tuple(shape)
        return cls(shape, tuple(value(at) for at in coords(shape)))

    def at(self, position):
        assert len(position) == len(self.shape)
        offset = 0
        for p, d in zip(position, self.shape):
            assert 0 <= p < d, (position, self.shape)
            offset = offset * d + p
        return self.values[offset]


def scalar_attribute(value):
    if value["kind"] == "list":
        return [scalar_attribute(v) for v in value["elements"]]
    if value["text"] in {"true", "false"}:
        return value["text"] == "true"
    try:
        return int(value["text"])
    except ValueError:
        return value["text"]


def meaning(program):
    """Physical array semantics; does not use TEPL's semantic graph or normalizer."""
    program = program["program"]
    graph = program["graphs"][0]
    operations = {op["id"]: op for op in program["operations"]}
    schemas = {s["id"]: s for s in program["attribute_schemas"]}
    values = []
    for node in graph["nodes"]:
        if node["kind"] == "input":
            shape = tuple(int(d) for d in node["shape"])
            seed = sum(ord(c) for c in node["text"])
            if node["dtype"] == "bool":
                values.append(Tensor(shape, tuple(bool((seed+i) % 2)
                                                 for i in range(math.prod(shape)))))
                continue
            offset = 0 if node["dtype"] in {"u8", "u16", "u32", "u64"} else 6
            values.append(Tensor(shape, tuple((seed + 3*i) % 13 - offset
                                             for i in range(math.prod(shape)))))
            continue
        op = operations[node["operation"]]
        fields = schemas[op["attributes"]]["fields"] if op["attributes"] is not None else []
        attrs = {f["name"]: scalar_attribute(v)
                 for f, v in zip(fields, node["attributes"])}
        args = [values[n] for n in node["operands"]]
        values.append(operation(op["name"], args, attrs))
    return values[graph["root"]]


def operation(name, xs, a):
    if name == "scalar":
        return Tensor((), (a["value"],))
    if name == "fill":
        return Tensor.build(a["shape"], lambda at: a["value"])
    x = xs[0]
    if name in {"exp", "log", "logical_not"}:
        function = {"exp": math.exp, "log": math.log, "logical_not": lambda a: not a}[name]
        return Tensor.build(x.shape, lambda at: function(x.at(at)))
    if name in {"minimum", "maximum"}:
        assert x.shape == xs[1].shape
        function = min if name == "minimum" else max
        return Tensor.build(x.shape, lambda at: function(x.at(at), xs[1].at(at)))
    if name == "abs":
        return Tensor.build(x.shape, lambda at: abs(x.at(at)))
    if name == "power":
        assert x.shape == xs[1].shape
        return Tensor.build(x.shape, lambda at: x.at(at)**xs[1].at(at))
    if name == "reverse":
        return Tensor.build(x.shape, lambda at: x.at(tuple(
            x.shape[i]-1-p if i in a["dimensions"] else p
            for i, p in enumerate(at))))
    if name == "slice":
        return Tensor.build(a["sizes"], lambda at: x.at(tuple(
            start + stride*p for start, stride, p
            in zip(a["starts"], a["strides"], at))))
    if name in {"pad", "nonnegative_pad"}:
        shape = [d + max(d-1, 0)*interior + low + high
                 for d, interior, low, high
                 in zip(x.shape, a["interior"], a["low"], a["high"])]

        def padded(at):
            source = []
            for i, p in enumerate(at):
                index, remainder = divmod(p-a["low"][i], a["interior"][i]+1)
                if remainder or index < 0 or index >= x.shape[i]:
                    return xs[1].at(())
                source.append(index)
            return x.at(tuple(source))

        return Tensor.build(shape, padded)
    if name == "remove_unit_axes":
        removed = set(a["dimensions"])
        assert len(removed) == len(a["dimensions"])
        assert all(0 <= i < len(x.shape) and x.shape[i] == 1 for i in removed)
        shape = [size for i, size in enumerate(x.shape) if i not in removed]

        def original_position(at):
            remaining = iter(at)
            return tuple(0 if i in removed else next(remaining)
                         for i in range(len(x.shape)))

        return Tensor.build(shape, lambda at: x.at(original_position(at)))
    if name == "reshape":
        assert math.prod(a["shape"]) == len(x.values)
        return Tensor(tuple(a["shape"]), x.values)
    if name == "transpose":
        shape = [x.shape[i] for i in a["permutation"]]

        def transposed(at):
            source = [0]*len(at)
            for i, axis in enumerate(a["permutation"]):
                source[axis] = at[i]
            return x.at(tuple(source))

        return Tensor.build(shape, transposed)
    if name == "broadcast_in_dim":
        shape, extra = [], iter(a["new_extents"])
        for i in range(len(x.shape)+len(a["new_extents"])):
            shape.append(x.shape[a["dimensions"].index(i)]
                         if i in a["dimensions"] else next(extra))
        return Tensor.build(shape, lambda at: x.at(tuple(at[i] for i in a["dimensions"])))
    if name == "multiply":
        assert x.shape == xs[1].shape
        return Tensor.build(x.shape, lambda at: x.at(at)*xs[1].at(at))
    if name == "reduce_sum":
        keep = [i for i in range(len(x.shape)) if i not in a["dimensions"]]
        shape = [x.shape[i] for i in keep]

        def reduced(at):
            result = 0
            for reduced_at in coords([x.shape[i] for i in a["dimensions"]]):
                source = [0]*len(x.shape)
                for i, p in zip(keep, at):
                    source[i] = p
                for i, p in zip(a["dimensions"], reduced_at):
                    source[i] = p
                result += x.at(tuple(source))
            return result

        return Tensor.build(shape, reduced)
    if name == "dot_general":
        assert a["lhs_batching_dimensions"] == a["rhs_batching_dimensions"] == []
        y = xs[1]
        lc, rc = a["lhs_contracting_dimensions"], a["rhs_contracting_dimensions"]
        lf = [i for i in range(len(x.shape)) if i not in lc]
        rf = [i for i in range(len(y.shape)) if i not in rc]
        shape = [x.shape[i] for i in lf] + [y.shape[i] for i in rf]

        def dot(at):
            result = 0
            for contraction in coords([x.shape[i] for i in lc]):
                left, right = [0]*len(x.shape), [0]*len(y.shape)
                for i, p in zip(lf, at[:len(lf)]):
                    left[i] = p
                for i, p in zip(rf, at[len(lf):]):
                    right[i] = p
                for i, j, p in zip(lc, rc, contraction):
                    left[i], right[j] = p, p
                result += x.at(tuple(left))*y.at(tuple(right))
            return result

        return Tensor.build(shape, dot)
    if name == "general_convolution":
        w = xs[1]
        assert a["batch_group_count"] == a["feature_group_count"] == 1
        assert len(a["input_spatial_dimensions"]) == 1
        ia, ka, oa = (a[key][0] for key in (
            "input_spatial_dimensions", "kernel_spatial_dimensions", "output_spatial_dimensions"))
        channels = w.shape[a["kernel_input_feature_dimension"]]

        def convolution(at):
            result = 0
            for k in range(w.shape[ka]):
                offset = at[oa]*a["window_strides"][0] - a["padding_low"][0] + k*a["rhs_dilation"][0]
                position, remainder = divmod(offset, a["lhs_dilation"][0])
                if remainder or position < 0 or position >= x.shape[ia]:
                    continue
                for c in range(channels):
                    left, right = [0]*3, [0]*3
                    left[ia] = position
                    left[a["input_batch_dimension"]] = at[a["output_batch_dimension"]]
                    left[a["input_feature_dimension"]] = c
                    right[ka] = w.shape[ka]-1-k if a["window_reversal"][0] else k
                    right[a["kernel_input_feature_dimension"]] = c
                    right[a["kernel_output_feature_dimension"]] = at[a["output_feature_dimension"]]
                    result += x.at(tuple(left))*w.at(tuple(right))
            return result

        return Tensor.build(a["shape"], convolution)
    raise AssertionError("reference evaluator does not implement " + name)


@unittest.skipUnless(COMPILER.is_file(), "build //:tepl to run compiled rewrite checks")
class CompiledRuleExecutionTest(unittest.TestCase):
    def test_broadcast_transpose_inference_execution(self):
        # Different axis orders, scalar input, empty axes, and repeated extents
        # exercise the original runtime metadata program as well as the proof.
        cases = [
            ([2, 3], [0, 2], [4, 2], [2, 3, 0, 1]),
            ([2, 3], [2, 0], [4, 2], [1, 3, 2, 0]),
            ([], [], [2, 3], [1, 0]),
            ([2, 2], [1, 0], [], [1, 0]),
            ([0, 3], [0, 2], [2], [2, 0, 1]),
            ([], [], [], []),
        ]
        rule = TEPL / "rules/experiments/broadcast_transpose_broadcast.tepl"
        for shape, dims, extents, permutation in cases:
            with self.subTest(shape=shape, dims=dims, extents=extents,
                              permutation=permutation), tempfile.TemporaryDirectory(dir=TEPL) as directory:
                directory = Path(directory)
                graph = directory / "input.tepl"
                graph.write_text(
                    'from "dialects/hlo.tepl" import HLO as hlo;\n'
                    + f'graph test {{input X:f32{shape}; yield '
                    + f'(hlo.transpose[permutation={permutation}] '
                    + f'(hlo.broadcast[dimensions={dims},new_extents={extents}] X));}}'
                )

                def cli(*args):
                    result = subprocess.run([str(COMPILER), *map(str, args)],
                                            cwd=ROOT, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    return result

                before, after = directory / "before.json", directory / "after.json"
                exported = directory / "after.tepl"
                cli("compile", graph, "--rules", rule, "--semantic-policy=algebraic", "-o", before)
                run = cli("rewrite", before, "--rule", "broadcast_transpose_broadcast",
                          "--trace", "--max-steps", 1, "-o", exported)
                self.assertIn("Successfully applied rule 'broadcast_transpose_broadcast'.", run.stderr)
                cli("compile", exported, "--semantic-policy=algebraic", "-o", after)
                self.assertEqual(meaning(json.loads(before.read_text())),
                                 meaning(json.loads(after.read_text())))

    def test_broadcast_transpose_rejects_wrong_extents(self):
        source = (TEPL / "rules/experiments/broadcast_transpose_broadcast.tepl").read_text()
        before = "new_extents: gather(gather(oldshape, p.attrs.permutation), newpos)"
        self.assertIn(before, source)
        source = source.replace(before, "new_extents: b.attrs.new_extents")
        with tempfile.TemporaryDirectory(dir=TEPL) as directory:
            rule = Path(directory) / "wrong.tepl"
            rule.write_text(source)
            result = subprocess.run([str(COMPILER), "check", str(rule),
                                     "--semantic-policy=algebraic", "--rule",
                                     "broadcast_transpose_broadcast"],
                                    cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("status=proved", result.stdout)

    def test_compiled_rules_preserve_dense_results(self):
        fixtures = {
            "reverse_compose": "expr_reverse_compose_general",
            "slice_compose": "expr_slice_compose_general",
            "signed_pad_crop": "catalog_indexing_pad_negative",
            "replication_sum": "catalog_reduction_broadcast_replication_axes",
            "pointwise_convolution": "catalog_convolution_pointwise_dot",
            "swapped_convolution": "catalog_convolution_swap_operands",
            "power_two_splat": "native_power_two_splat",
            "abs_unsigned": "native_abs_unsigned",
        }
        for graph, rule in fixtures.items():
            with self.subTest(graph=graph), tempfile.TemporaryDirectory(dir=TEPL) as directory:
                directory = Path(directory)

                def cli(*args):
                    result = subprocess.run([str(COMPILER), *map(str, args)],
                                            cwd=ROOT, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    return result

                before, after = directory/"before.json", directory/"after.json"
                exported = directory/"after.tepl"
                cli("compile", TEPL/"graphs"/(graph+".tepl"), "--rules",
                    TEPL/"rules/structural"/(rule+".tepl"), "-o", before)
                run = cli("rewrite", before, "--rule", rule, "--trace",
                          "--max-steps", 1, "-o", exported)
                self.assertIn("Successfully applied rule '" + rule + "'.", run.stderr)
                cli("compile", exported, "-o", after)
                self.assertEqual(meaning(json.loads(before.read_text())),
                                 meaning(json.loads(after.read_text())))

    def test_verified_scalar_rules_preserve_dense_results(self):
        fixtures = (
            ("expr_minimum_nested", "input X:f32[2,3]; input Y:f32[2,3];",
             "(hlo.minimum (hlo.minimum X Y) X)"),
            ("expr_log_exp", "input X:f32[2,3];", "(hlo.log (hlo.exp X))"),
            ("expr_double_not", "input X:bool[2,3];", "(hlo.logical_not (hlo.logical_not X))"),
        )
        for rule, inputs, expression in fixtures:
            with self.subTest(rule=rule), tempfile.TemporaryDirectory(dir=TEPL) as directory:
                directory = Path(directory)
                graph = directory / "input.tepl"
                graph.write_text('from "dialects/hlo.tepl" import HLO as hlo;\n'
                                 + "graph scalar_case {" + inputs + "yield " + expression + ";}")

                def cli(*args):
                    result = subprocess.run([str(COMPILER), *map(str, args)],
                                            cwd=ROOT, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    return result

                before, after = directory / "before.json", directory / "after.json"
                exported = directory / "after.tepl"
                cli("compile", graph, "--rules", TEPL / "rules/semantic" / (rule + ".tepl"),
                    "--semantic-policy=algebraic", "-o", before)
                run = cli("rewrite", before, "--rule", rule, "--trace",
                          "--max-steps", 1, "-o", exported)
                self.assertIn("Successfully applied rule '" + rule + "'.", run.stderr)
                cli("compile", exported, "--semantic-policy=algebraic", "-o", after)
                lhs, rhs = meaning(json.loads(before.read_text())), meaning(json.loads(after.read_text()))
                self.assertEqual(lhs.shape, rhs.shape)
                for a, b in zip(lhs.values, rhs.values, strict=True):
                    self.assertAlmostEqual(a, b)

    def test_axis_unit_views_preserve_dense_results(self):
        fixtures = [
            ("bool", 0, []), ("bool", 3, [2]),
            ("f32", 1, [0, 2]), ("f64", 4, []), ("i64", 2, [3]),
        ]
        for rule in ["catalog_broadcast_inserted_unit_dims", "catalog_broadcast_drop_unit_dims"]:
            for dtype, size, extra in fixtures:
                with self.subTest(rule=rule, dtype=dtype, size=size, extra=extra), tempfile.TemporaryDirectory(dir=TEPL) as directory:
                    directory = Path(directory)
                    inserting = rule == "catalog_broadcast_inserted_unit_dims"
                    shape = [size] if inserting else [1, size]
                    inner = "(hlo.broadcast[dimensions=[1],new_extents=[1]] X)" if inserting else "X"
                    expression = f"(hlo.broadcast[dimensions=[0,1],new_extents={extra}] {inner})"
                    graph = directory / "input.tepl"
                    graph.write_text('from "dialects/hlo.tepl" import HLO as hlo;\n'
                                     + f"graph unit_view {{input X:{dtype}{shape}; yield {expression};}}")

                    def cli(*args):
                        result = subprocess.run([str(COMPILER), *map(str, args)],
                                                cwd=ROOT, text=True, capture_output=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                        return result

                    before, after = directory / "before.json", directory / "after.json"
                    exported = directory / "after.tepl"
                    cli("compile", graph, "--rules", TEPL / "rules/axis" / (rule + ".tepl"),
                        "--semantic-policy=algebraic", "-o", before)
                    rewritten = cli("rewrite", before, "--rule", rule, "--trace", "--max-steps", 1, "-o", exported)
                    self.assertIn("Successfully applied rule '" + rule + "'.", rewritten.stderr)
                    cli("compile", exported, "--semantic-policy=algebraic", "-o", after)
                    self.assertEqual(meaning(json.loads(before.read_text())),
                                     meaning(json.loads(after.read_text())))

    def test_native_conditions_reject_other_values_and_types(self):
        fixtures = (
            ("power_two_splat", "native_power_two_splat", "value=2", "value=3"),
            ("abs_unsigned", "native_abs_unsigned", "u32", "i32"),
        )
        for graph, rule, old, new in fixtures:
            with self.subTest(graph=graph), tempfile.TemporaryDirectory(dir=TEPL) as directory:
                directory = Path(directory)
                source = (TEPL / "graphs" / (graph + ".tepl")).read_text()
                self.assertIn(old, source)
                source = source.replace(old, new)
                graph_file = directory / "input.tepl"
                graph_file.write_text(source)
                artifact = directory / "program.json"
                compiled = subprocess.run(
                    [str(COMPILER), "compile", str(graph_file), "--rules",
                     str(TEPL / "rules/structural" / (rule + ".tepl")),
                     "-o", str(artifact)],
                    cwd=ROOT, text=True, capture_output=True, timeout=30)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                result = subprocess.run(
                    [str(COMPILER), "rewrite", str(artifact), "--rule", rule,
                     "--trace", "--max-steps", "1", "-o", str(directory / "out.tepl")],
                    cwd=ROOT, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Rewrites: 0;", result.stderr)


if __name__ == "__main__":
    unittest.main()
