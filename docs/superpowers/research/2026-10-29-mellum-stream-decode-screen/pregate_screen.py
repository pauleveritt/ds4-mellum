"""Pre-gating screen: how well does layer L's FFN input, renormalized with layer L+k's ffn_norm weight and put
through layer L+k's router, predict layer L+k's top-8?  Inputs: a mellum_trace prefix (ffn_in per layer at traced
positions) and the GGUF for the routers and norms.
    uv run --with numpy python -I pregate_screen.py MODEL.gguf TRACE_PREFIX
"""
import glob, importlib.util, os, re, sys
import numpy as np
spec = importlib.util.spec_from_file_location('g', os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tools', 'mellum_gguf.py'))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
model, prefix = sys.argv[1], sys.argv[2]
with open(model, 'rb') as f:
    d = g.read_directory(f, os.path.getsize(model))
    t = {x.name: x for x in d.tensors}
    def load(name):
        x = t[name]; assert x.type == 0, (name, x.type)
        f.seek(d.data_start + x.offset); a = np.frombuffer(f.read(x.bytes), dtype=np.float32)
        return a.reshape(tuple(reversed(x.dims)))
    L = sum(1 for n in t if re.fullmatch(r'blk\.\d+\.ffn_gate_inp\.weight', n))
    router = [load(f'blk.{i}.ffn_gate_inp.weight') for i in range(L)]
    norm = [load(f'blk.{i}.ffn_norm.weight') for i in range(L)]
files = sorted(glob.glob(prefix + '.p*.ffn_in.f32'))
E = router[0].shape[1]
def top8(v): return set(np.argsort(-v)[:8].tolist())
def rms(v): return v / np.sqrt(np.mean(v * v) + 1e-6)
variants = {
    'ffn_in[L] as is': lambda x, h, l, k: x[l],
    'rmsnorm(hidden[L]) * ffn_norm[L+k] (after layer L)': lambda x, h, l, k: rms(h[l]) * norm[l + k],
}
for name, fn_pred in variants.items():
    for k in (1, 2):
        hits, full, n = 0, 0, 0
        for fn in files:
            x = np.fromfile(fn, dtype=np.float32).reshape(L, E)
            h = np.fromfile(fn.replace('.ffn_in.', '.hidden.'), dtype=np.float32).reshape(L, E)
            for l in range(L - k):
                actual = top8(router[l + k] @ x[l + k])
                pred = top8(router[l + k] @ fn_pred(x, h, l, k))
                o = len(actual & pred); hits += o; full += o == 8; n += 1
        print('%s, k=%d: %d pairs over %d positions; mean overlap %.2f of 8; all 8 predicted %.1f%%'
              % (name, k, n, len(files), hits / n, 100 * full / n))
