"""Correctness + benchmark harness for fused residual add + RMSNorm + per-token-group fp8 quant (see problem_desc.md).

Impl contract: module exposes `make()` returning
    fn(x[M,H] bf16, residual[M,H] bf16, weight[H] bf16) -> (h[M,H] bf16, q[M,H] fp8e4m3, s[M,H/128] fp32)
"""

import math

import torch

try:
    from common.bench import time_graph  # this repo's layout (Modal harness)
except ImportError:  # standalone copy: bench.py next to this file
    from bench import time_graph

G = 128  # quantization group along H
EPS = 1e-6
FP8_MAX = 448.0
AMAX_MIN = 1e-10
FP8 = torch.float8_e4m3fn
HBM_BW = 3.35e12  # H100 SXM spec, bytes/s

BENCH_SHAPES = [(16384, 7168), (2048, 7168), (128, 7168)]  # first one is the main number
CASES = [  # (M, H, dist, seed)
    (1000, 7168, "normal", 0),  # M not a multiple of anything
    (2048, 7168, "outliers", 1),
    (2048, 7168, "row_scale", 2),
    (128, 4096, "normal", 3),  # another H
    (16384, 7168, "normal", 4),  # the benchmark shape itself
]


def problem_bytes(M, H):
    return M * H * (2 + 2 + 2 + 1) + M * (H // G) * 4 + H * 2


# ----------------------------------------------------------------------------- inputs
def make_inputs(M, H, seed=0, dist="normal"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, H, device="cuda", generator=g)
    r = torch.randn(M, H, device="cuda", generator=g)
    w = 1 + 0.1 * torch.randn(H, device="cuda", generator=g)
    if dist == "normal":
        pass
    elif dist == "outliers":  # per-channel outliers + sparse spikes: neighbouring group scales differ by ~100x
        ch = torch.randperm(H, device="cuda", generator=g)[:8]
        x[:, ch] *= 100
        spikes = torch.rand(M, H, device="cuda", generator=g) < 1e-3
        r = torch.where(spikes, r * 30, r)
    elif dist == "row_scale":  # row magnitudes 2^-24 .. 2^16 (eps dominates tiny rows) + exactly-zero groups
        e = torch.randint(-24, 17, (M, 1), device="cuda", generator=g).float().exp2()
        x, r = x * e, r * e
    else:
        raise ValueError(dist)
    x, r, w = x.to(torch.bfloat16), r.to(torch.bfloat16), w.to(torch.bfloat16)
    if dist == "row_scale":  # 3 groups per row where residual = -x exactly -> h == 0 there, amax == 0
        n_g = H // G
        for j in range(3):
            gi = (torch.arange(M, device="cuda") * (j + 3) + 7 * j) % n_g
            cols = gi[:, None] * G + torch.arange(G, device="cuda")[None, :]
            r.scatter_(1, cols, -x.gather(1, cols))
    return x.contiguous(), r.contiguous(), w.contiguous()


# ----------------------------------------------------------------------------- correctness
def reference64(x, r, w):
    """fp64 reference: h (bf16, exact rule), y, s_ref, t = y / s_ref (fp8 units)."""
    h = (x.float() + r.float()).to(torch.bfloat16)
    hd = h.double()
    rstd = torch.rsqrt(hd.pow(2).mean(dim=1, keepdim=True) + EPS)
    y = hd * rstd * w.double()
    M, H = y.shape
    amax = y.view(M, H // G, G).abs().amax(dim=2).clamp_min(AMAX_MIN)
    s = amax / FP8_MAX
    t = (y.view(M, H // G, G) / s.unsqueeze(2)).view(M, H)
    return h, s, t


def ulp_e4m3(a):
    """Spacing of e4m3 values around |a| (fp64 tensor): 2^(e-3) for normals (e >= -6), 2^-9 below (subnormals)."""
    e = torch.floor(torch.log2(a.clamp_min(2.0 ** -6)))
    return torch.exp2(e - 3)


def check_one(out, x, r, w):
    """Returns (list of error strings, worst q err/bound, worst s rel err, # q != correctly rounded t).
    q err/bound is ~1 for any correctly rounded output (a value can sit right at half an ulp); the mismatch count
    vs round-to-nearest(t) is the informative number (expect 0 or a handful from fp32-vs-fp64 noise at midpoints)."""
    M, H = x.shape
    errs = []
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        return [f"expected (h, q, s), got {type(out).__name__}"], math.nan, math.nan, -1
    h, q, s = out
    for name, tns, shape, dt in (("h", h, (M, H), torch.bfloat16), ("q", q, (M, H), FP8),
                                 ("s", s, (M, H // G), torch.float32)):
        if tuple(tns.shape) != shape or tns.dtype != dt or not tns.is_contiguous():
            errs.append(f"{name}: shape={tuple(tns.shape)} dtype={tns.dtype} contiguous={tns.is_contiguous()}, "
                        f"want {shape} {dt}")
    if errs:
        return errs, math.nan, math.nan, -1

    h_ref, s_ref, t = reference64(x, r, w)
    if not torch.equal(h, h_ref):
        n = (h != h_ref).sum().item()
        i = (h != h_ref).view(-1).nonzero()[0].item()
        errs.append(f"h: {n} elements differ from bf16(x + residual); first at {divmod(i, H)}: "
                    f"{h.view(-1)[i].item()} vs {h_ref.view(-1)[i].item()}")

    s_rel = ((s.double() - s_ref).abs() / s_ref)
    s_worst = s_rel.max().item()
    if not s_worst <= 2.0 ** -16:  # also catches NaN
        i = s_rel.view(-1).nan_to_num(math.inf).argmax().item()
        mm, gg = divmod(i, H // G)
        errs.append(f"s: {(~(s_rel <= 2.0 ** -16)).sum().item()} scales off by > 2^-16 rel; worst ({mm},{gg}) "
                    f"{s[mm, gg].item():.6g} vs ref {s_ref[mm, gg].item():.6g}")

    qd = q.double()
    a = t.abs()
    bound = 0.5 * ulp_e4m3(a) + 2.0 ** -16 * a
    ratio = (qd - t).abs() / bound
    q_worst = ratio.nan_to_num(math.inf).max().item()
    if torch.isnan(qd).any() or (qd.abs() > FP8_MAX).any():
        errs.append(f"q: {torch.isnan(qd).sum().item()} NaN, {(qd.abs() > FP8_MAX).sum().item()} beyond ±448")
    elif q_worst > 1:
        i = ratio.view(-1).argmax().item()
        mm, kk = divmod(i, H)
        errs.append(f"q: {(ratio > 1).sum().item()} elements over bound; worst ({mm},{kk}) q={qd[mm, kk].item():.6g} "
                    f"t={t[mm, kk].item():.6g} (err/bound {q_worst:.3g})")
    q_rn = t.float().clamp(-FP8_MAX, FP8_MAX).to(FP8)  # fp64 -> fp32 -> fp8: double rounding only at exact ties
    n_off = int((q.view(torch.uint8) != q_rn.view(torch.uint8)).sum().item())
    return errs, q_worst, s_worst, n_off


def check(fn, name="impl", cases=CASES) -> tuple[list[str], list[str]]:
    errs, info = [], []
    for M, H, dist, seed in cases:
        x, r, w = make_inputs(M, H, seed, dist)
        x0, r0, w0 = x.clone(), r.clone(), w.clone()
        out = fn(x, r, w)
        torch.cuda.synchronize()
        tag = f"[{M}x{H} {dist}]"
        e, q_worst, s_worst, n_off = check_one(out, x, r, w)
        errs += [f"{tag} {m}" for m in e]
        info.append(f"{name:<24} {tag:<24} q != RN(t): {n_off:<6} q err/bound {q_worst:.3g}   s rel err {s_worst:.2g}")
        if not (torch.equal(x, x0) and torch.equal(r, r0) and torch.equal(w, w0)):
            errs.append(f"{tag} inputs were modified")
        if not e:
            out2 = fn(x, r, w)
            if not all(torch.equal(a.view(torch.uint8) if a.dtype == FP8 else a,
                                   b.view(torch.uint8) if b.dtype == FP8 else b) for a, b in zip(out, out2)):
                errs.append(f"{tag} determinism: two calls differ")
        del out
    return errs, info


# ----------------------------------------------------------------------------- baselines
def torch_ref_fn(x, r, w):
    """The spec, in plain torch (fp32 math). Also torch_ref.make()."""
    M, H = x.shape
    h = (x.float() + r.float()).to(torch.bfloat16)
    hf = h.float()
    y = hf * torch.rsqrt(hf.pow(2).mean(dim=1, keepdim=True) + EPS) * w.float()
    yg = y.view(M, H // G, G)
    s = yg.abs().amax(dim=2).clamp_min(AMAX_MIN) / FP8_MAX
    q = (yg / s.unsqueeze(2)).clamp(-FP8_MAX, FP8_MAX).to(FP8).view(M, H)
    return h, q, s


def baselines():
    b = {"torch eager": torch_ref_fn}
    try:
        # emulate_precision_casts: by default Inductor drops the .to(bf16).float() round trip and normalizes the
        # unrounded fp32 sum (scales off by ~2^-9); the spec normalizes the bf16 h, like vLLM.
        b["torch.compile"] = torch.compile(torch_ref_fn, dynamic=False, options={"emulate_precision_casts": True})
    except Exception as e:  # noqa: BLE001
        print(f"torch.compile unavailable: {type(e).__name__}: {e}")
    return b


# ----------------------------------------------------------------------------- driver
def n_sets(M, H, target=200e6, cap=32):
    return max(2, min(cap, math.ceil(target / problem_bytes(M, H))))


def _fmt(name, us, nbytes):
    gbs = nbytes / (us * 1e-6) / 1e9
    return f"    {name:<28} {us:9.2f} us  {gbs:7.0f} GB/s  {100 * gbs * 1e9 / HBM_BW:5.1f}% of HBM"


def run(impl, mode="bench") -> str:
    torch.manual_seed(0)
    fn = impl.make()
    if mode == "profile":  # main shape: one warm call, then NVTX-tagged calls for ncu
        M, H = BENCH_SHAPES[0]
        sets = [make_inputs(M, H, 100 + i) for i in range(3)]
        fn(*sets[0])
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile")
        for s in sets[1:3]:
            fn(*s)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return ""

    lines = [f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | eps={EPS} | "
             f"floor at {HBM_BW / 1e12:.2f} TB/s: " +
             ", ".join(f"{M}x{H} {problem_bytes(M, H) / HBM_BW * 1e6:.1f} us" for M, H in BENCH_SHAPES)]
    errs, info = check(fn)
    lines.append("correctness: PASS" if not errs else "correctness: FAIL\n  " + "\n  ".join(errs))
    lines += ["  " + s for s in info]

    bl = baselines()
    for name, f in bl.items():  # baselines must pass the same check (one case each, the main shape is the slow one)
        try:
            berrs, binfo = check(f, name, cases=CASES[:3])
        except Exception as e:  # noqa: BLE001
            lines.append(f"  {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            continue
        lines += ["  " + s for s in binfo] + [f"  {name} FAILS: {s}" for s in berrs]

    lines.append("CUDA graph, rotating input sets (> L2), median per call:")
    for M, H in BENCH_SHAPES:
        k = n_sets(M, H)
        sets = [make_inputs(M, H, 100 + i) for i in range(k)]
        nb = problem_bytes(M, H)
        lines.append(f"  {M} x {H}: {nb / 1e6:.1f} MB, floor {nb / HBM_BW * 1e6:.1f} us, {k} input sets")
        results = {"** impl **": time_graph(fn, sets)}
        for name, f in bl.items():
            try:
                results[name] = time_graph(f, sets)
            except Exception as e:  # noqa: BLE001
                lines.append(f"    {name}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        for name, (med, mn) in results.items():
            lines.append(_fmt(name, med, nb) + f"   (min {mn:.2f})")
        med, _ = time_graph(lambda x, r, w: x + r, sets)  # bandwidth probe: its own bytes (read 4, write 2 per elem)
        lines.append(_fmt("x + residual [speed]", med, M * H * 6) + "   <- achievable HBM BW, own bytes")
        del sets
        torch.cuda.empty_cache()
    return "\n".join(lines)


if __name__ == "__main__":  # python harness.py [impl_module]: full correctness check + CUDA-graph benchmark
    import importlib
    import sys

    print(run(importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "torch_ref")))
