"""Bytes one Mellum decode step reads (phase 8b, D1), from the GGUF's own tensor table: no model process.

    python3 -I tests/mellum_decode_bytes.py MODEL.gguf [CONTEXT ...]      e.g. ... 100 2048 8192 32768

A decode step at context c (the new token is position c - 1) reads:
  - one row of token_embd (the embedding is a row gather);
  - per layer the norms, the Q, K, V and output projections, the F32 router, and the 8 selected experts'
    gate, up and down slices (each 1/64 of its tensor, since the expert tensors are [.., .., 64]);
  - output_norm and the whole output head;
  - the K and V cache rows attention reads, as f16: min(c, 1,024) rows on a sliding layer (the ring) and c rows
    on a full layer, plus the ring's position tags (int32, one per row) on a sliding layer.
It leaves out the activations (a few KB of f32 rows per step) and every write; both are small next to these.
Which layers slide comes from the GGUF's sliding-window pattern if present, else Mellum's (il + 1) % 4 != 0.
"""

import struct
import sys

# ggml type: (block elements, block bytes)
GGML = {0: (1, 4), 1: (1, 2), 8: (32, 34), 30: (1, 2)}
NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 30: "BF16"}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def string(self):
        return self.f.read(self.take("<Q")).decode("utf-8", "replace")

    def value(self, t):
        if t in SCALAR:
            return self.take(SCALAR[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("<I"), self.take("<Q")
            if et in SCALAR and n > 64:  # long numeric arrays (the tokenizer's): skip, keep the length
                self.f.seek(n * struct.calcsize(SCALAR[et]), 1)
                return ("array", n)
            return [self.value(et) for _ in range(n)]
        raise ValueError("unknown GGUF value type %d" % t)


def read_gguf(path):
    with open(path, "rb") as f:
        r = Reader(f)
        assert f.read(4) == b"GGUF", "not a GGUF file"
        version, n_tensors, n_kv = r.take("<I"), r.take("<Q"), r.take("<Q")
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.take("<I"))
        tensors = {}
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.take("<Q") for _ in range(r.take("<I"))]
            ggml_type, _offset = r.take("<I"), r.take("<Q")
            n = 1
            for d in dims:
                n *= d
            be, bb = GGML[ggml_type]
            tensors[name] = (dims, ggml_type, n // be * bb)
    return version, kv, tensors


def find(kv, suffix):
    hits = [v for k, v in kv.items() if k.endswith(suffix)]
    return hits[0] if hits else None


def layout(kv):
    """n_layer, n_expert, n_used, window, bytes of one f16 K or V row, and which layers slide."""
    n_layer = find(kv, ".block_count")
    pattern = find(kv, ".attention.sliding_window_pattern")
    if isinstance(pattern, list) and len(pattern) == n_layer:
        sliding = [bool(x) for x in pattern]
    else:
        sliding = [(il + 1) % 4 != 0 for il in range(n_layer)]
    row = find(kv, ".attention.head_count_kv") * find(kv, ".attention.key_length") * 2
    return (n_layer, find(kv, ".expert_count"), find(kv, ".expert_used_count"),
            find(kv, ".attention.sliding_window"), row, sliding)


def weight_groups(kv, tensors):
    """{group: [bytes a step reads, {types}]}; refuses if any tensor is left uncounted."""
    n_layer, n_expert, n_used, _window, _row, _sliding = layout(kv)
    groups, used = {}, {"token_embd.weight"}

    def add(group, name, frac=1.0):
        used.add(name)
        dims, t, nbytes = tensors[name]
        groups.setdefault(group, [0.0, set()])
        groups[group][0] += nbytes * frac
        groups[group][1].add(NAMES.get(t, str(t)))

    embd_dims, _, embd_bytes = tensors["token_embd.weight"]
    groups["embedding row"] = [embd_bytes / embd_dims[1], {NAMES[tensors["token_embd.weight"][1]]}]
    for il in range(n_layer):
        p = "blk.%d." % il
        for name in ("attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"):
            add("norms", p + name + ".weight")
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            add("attention projections", p + name + ".weight")
        add("router", p + "ffn_gate_inp.weight")
        for name in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            add("experts (%d of %d)" % (n_used, n_expert), p + name + ".weight", n_used / n_expert)
    add("norms", "output_norm.weight")
    add("output head", "output.weight")
    assert used == set(tensors), "tensors a step was not counted against: %s" % sorted(set(tensors) - used)
    return groups


def kv_read(kv, context):
    """Bytes of K/V rows (and ring tags) attention reads at this context."""
    n_layer, _ne, _nu, window, row, sliding = layout(kv)
    total = 0
    for il in range(n_layer):
        rows = min(context, window) if sliding[il] else context
        total += rows * 2 * row + (rows * 4 if sliding[il] else 0)
    return total


def main(path, contexts):
    version, kv, tensors = read_gguf(path)
    n_layer, n_expert, n_used, window, row, sliding = layout(kv)
    print("model: %s (GGUF v%d, %d tensors, %.3f GB on disk)" %
          (path, version, len(tensors), sum(t[2] for t in tensors.values()) / 1e9))
    print("layers %d (%d sliding, window %d), experts %d of %d, kv heads %d x %d" %
          (n_layer, sum(sliding), window, n_used, n_expert, find(kv, ".attention.head_count_kv"),
           find(kv, ".attention.key_length")))
    groups = weight_groups(kv, tensors)
    weights = sum(v[0] for v in groups.values())
    for name, (b, types) in groups.items():
        print("  %-24s %12.3f MB  %s" % (name, b / 1e6, "/".join(sorted(types))))
    print("  %-24s %12.3f MB" % ("weights per step", weights / 1e6))

    print("\ncontext  KV read (MB)  total (MB)  KV share")
    for c in contexts:
        kv_bytes = kv_read(kv, c)
        total = weights + kv_bytes
        print("%7d  %12.2f  %10.2f  %7.1f%%" % (c, kv_bytes / 1e6, total / 1e6, 100 * kv_bytes / total))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or [100, 2048, 8192, 32768])
