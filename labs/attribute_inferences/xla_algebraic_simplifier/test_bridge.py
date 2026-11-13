"""Independent permutation oracle for the compiled-rule C bridge."""
import argparse
import ctypes as c
import itertools
import json


class List(c.Structure):
    _fields_ = [("data", c.POINTER(c.c_int64)), ("size", c.c_size_t)]


class Node(c.Structure):
    _fields_ = [("op", c.c_int), ("f32", c.c_int), ("original_id", c.c_int64),
                ("shape", List), ("permutation", List), ("fields", List * 4),
                ("operands", c.c_size_t * 2), ("arity", c.c_size_t)]


class Result(c.Structure):
    _fields_ = [("nodes", c.POINTER(Node)), ("count", c.c_size_t),
                ("root", c.c_size_t), ("rule", c.c_size_t)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("program")
    args = parser.parse_args()
    lib = c.CDLL(args.library)
    lib.tepl_algebraic_load.restype = c.c_void_p
    lib.tepl_algebraic_load.argtypes = [c.c_char_p, c.c_char_p, c.c_size_t]
    lib.tepl_algebraic_match.argtypes = [c.c_void_p, c.POINTER(Node), c.c_size_t,
                                       c.c_size_t, c.c_uint64, c.POINTER(Result), c.c_char_p, c.c_size_t]
    lib.tepl_algebraic_destroy.argtypes = [c.c_void_p]
    error = c.create_string_buffer(4096)
    context = lib.tepl_algebraic_load(args.program.encode(), error, len(error))
    if not context:
        raise RuntimeError(error.value.decode())
    checks = 0
    try:
        for rank in range(4):
            for first in itertools.permutations(range(rank)):
                for second in itertools.permutations(range(rank)):
                    keep = []
                    def xs(values):
                        array = (c.c_int64 * len(values))(*values)
                        keep.append(array)
                        return List(array, len(array))
                    dims = list(range(2, rank + 2))
                    middle = [dims[i] for i in first]
                    nodes = (Node * 3)()
                    for i, shape in enumerate([dims, middle, [middle[j] for j in second]]):
                        nodes[i].f32 = 1
                        nodes[i].original_id = i
                        nodes[i].shape = xs(shape)
                        if i:
                            nodes[i].op = 1
                            nodes[i].arity = 1
                            nodes[i].operands[0] = i - 1
                            nodes[i].permutation = xs(first if i == 1 else second)
                    result = Result()
                    code = lib.tepl_algebraic_match(context, nodes, 3, 2, 0, c.byref(result), error, len(error))
                    if code != 1:
                        raise ValueError(error.value.decode())
                    root = result.nodes[result.root]
                    if second == tuple(range(rank)):
                        if root.original_id != 1:
                            raise ValueError("identity rule did not preserve captured HLO identity")
                    elif list(root.permutation.data[:root.permutation.size]) != [first[j] for j in second]:
                        raise ValueError("composed permutation differs from independent oracle")
                    for node in nodes:
                        node.f32 = 0
                    if lib.tepl_algebraic_match(context, nodes, 3, 2, 0, c.byref(Result()), error, len(error)) != 0:
                        raise ValueError("unsupported capture dtype accepted")
                    checks += 1
    finally:
        lib.tepl_algebraic_destroy(context)
    print(json.dumps(dict(permutation_cases=checks, dtype_rejections=checks)))


if __name__ == "__main__":
    main()
