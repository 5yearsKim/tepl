"""Independent permutation and protocol checks for the versioned JSON ABI."""
import argparse
import ctypes as c
import itertools
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("automatic_program")
    args = parser.parse_args()
    library = c.CDLL(args.library)
    library.tepl_branch_load.restype = c.c_void_p
    library.tepl_branch_load.argtypes = [c.c_char_p, c.c_char_p, c.c_size_t]
    library.tepl_branch_apply.argtypes = [c.c_void_p, c.c_char_p, c.POINTER(c.c_char_p), c.c_char_p, c.c_size_t]
    library.tepl_branch_destroy.argtypes = [c.c_void_p]
    error = c.create_string_buffer(4096)
    context = library.tepl_branch_load(args.automatic_program.encode(), error, len(error))
    if not context:
        raise RuntimeError(error.value.decode())
    checks = 0

    def apply(request):
        output = c.c_char_p(b"stale output must be cleared")
        code = library.tepl_branch_apply(context, json.dumps(request).encode(), c.byref(output), error, len(error))
        if code != 1 and output.value is not None:
            raise ValueError("failed ABI call retained stale output")
        return code, json.loads(output.value) if code == 1 else None

    try:
        for dtype in ("f16", "bf16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "bool"):
            for rank in range(4):
                shape = list(range(2, rank + 2))
                for first in itertools.permutations(range(rank)):
                    for second in itertools.permutations(range(rank)):
                        middle = [shape[i] for i in first]
                        nodes = [dict(dtype=dtype, shape=shape, original_id=100),
                                 dict(dtype=dtype, shape=middle, original_id=101, operation="transpose", operands=[0], attrs=dict(permutation=first)),
                                 dict(dtype=dtype, shape=[middle[i] for i in second], original_id=102, operation="transpose", operands=[1], attrs=dict(permutation=second))]
                        request = dict(version=1, site=10118, rule="transpose_compose", nodes=nodes, root=2)
                        code, result = apply(request)
                        if code != 1:
                            raise ValueError(error.value.decode())
                        actual = result["nodes"][result["root"]]
                        expected = [first[i] for i in second]
                        if actual["operation"] != "transpose" or actual["attrs"]["permutation"] != expected:
                            raise ValueError("compiled composition differs from independent oracle")
                        checks += 1
        # Version, graph topology, dtype, root, and absent-rule boundaries.
        request["version"] = 2
        if apply(request)[0] != -1:
            raise ValueError("unsupported version accepted")
        request["version"] = 1
        request["nodes"][1]["operands"] = [2]
        if apply(request)[0] != -1:
            raise ValueError("non-topological graph accepted")
        request["nodes"][1]["operands"] = [0]
        request["nodes"][0]["dtype"] = "unknown"
        if apply(request)[0] != -1:
            raise ValueError("unsupported dtype accepted")
        request["nodes"][0]["dtype"] = "bool"
        request["root"] = 99
        if apply(request)[0] != -1:
            raise ValueError("invalid graph root accepted")
        request["rule"] = "absent_rule"
        if apply(request)[0] != 0:
            raise ValueError("missing rule did not return no-match")
        if library.tepl_branch_apply(None, b"{}", c.byref(c.c_char_p()), error, len(error)) != -1:
            raise ValueError("null context was not rejected")
    finally:
        library.tepl_branch_destroy(context)
    print(json.dumps(dict(permutation_cases=checks, dtypes=13, protocol_rejections=6)))


if __name__ == "__main__":
    main()
