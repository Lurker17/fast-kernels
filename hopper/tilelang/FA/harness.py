"""Correctness + benchmark harness for attention prefill forward (see problem_desc.md). bf16 now; fp8 is stage 2.

Impl contract: module exposes `PRECISION = "bf16" | "fp8"` (default bf16) and `make()` returning
    bf16: fn(q[B,N,Hq,D], k[B,N,Hkv,D], v[B,N,Hkv,D], causal: bool)                          -> (o, lse)
    fp8:  fn(q8, k8, v8, q_descale[B,Hkv], k_descale[B,Hkv], v_descale[B,Hkv], causal: bool)   -> (o, lse)
with o [B,N,Hq,D] bf16 and lse [B,Hq,N] fp32 (natural log), both contiguous. GQA: query head h reads KV head h // G.
"""

import math

import torch
import torch.nn.functional as F

try:
    from common.bench import time_graph  # this repo's layout (Modal harness)
except ImportError:  # standalone copy: bench.py next to this file
    from bench import time_graph

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
PEAK = {"bf16": 989.4e12, "fp8": 1978.9e12}  # H100 SXM dense wgmma FLOP/s

# (B, N, Hq, Hkv, D, causal): Llama-3-8B attention (GQA 32/8, D=128), 16k tokens per call. Main number: causal 4x4096.
BENCH_SHAPES = [
    (4, 4096, 32, 8, 128, True),
    (4, 4096, 32, 8, 128, False),
    (16, 1024, 32, 8, 128, True),
    (16, 1024, 32, 8, 128, False),
    (1, 16384, 32, 8, 128, True),
    (1, 16384, 32, 8, 128, False),
]
CASES = [  # (B, N, Hq, Hkv, D, causal, dist, seed)
    (2, 1000, 8, 2, 128, False, "normal", 0),  # N not a multiple of any tile
    (2, 1000, 8, 2, 128, True, "peaky", 1),
    (1, 2048, 8, 8, 128, False, "peaky", 2),  # MHA (G = 1)
    (1, 2048, 8, 2, 128, True, "drift", 3),  # running max grows every block
    (2, 2048, 8, 2, 128, False, "drift", 4),
    (4, 128, 4, 1, 128, True, "normal", 5),  # a single block, all 4 q heads on one kv head
    (4, 4096, 32, 8, 128, True, "normal", 6),  # the main benchmark shapes themselves
    (4, 4096, 32, 8, 128, False, "normal", 7),
]


def flops(B, N, Hq, D, causal):
    return 4 * B * Hq * N * N * D / (2 if causal else 1)


# ----------------------------------------------------------------------------- inputs
def make_inputs(B, N, Hq, Hkv, D, seed=0, dist="normal"):
    """bf16 q [B,N,Hq,D], k, v [B,N,Hkv,D], contiguous."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, N, Hq, D, device="cuda", generator=g)
    k = torch.randn(B, N, Hkv, D, device="cuda", generator=g)
    v = torch.randn(B, N, Hkv, D, device="cuda", generator=g)
    if dist == "normal":
        pass
    elif dist == "peaky":  # logits std ~9: a few keys dominate each row
        q, k = q * 3, k * 3
    elif dist == "drift":  # logit bonus 0 -> ~20 along the key axis: running max grows at every block
        u = torch.randn(D, device="cuda", generator=g)
        u = u / u.norm()
        q = q + 3 * u
        ramp = torch.arange(N, device="cuda").float() / N
        k = k + (20 * math.sqrt(D) / 3) * ramp[None, :, None, None] * u
    else:
        raise ValueError(dist)
    return tuple(t.to(torch.bfloat16).contiguous() for t in (q, k, v))


def quantize_fp8(q, k, v):
    """Per-(batch, kv head) amax/448 quantization (FA3 descale convention). Returns (q8, k8, v8, dq, dk, dv)."""
    B, N, Hq, D = q.shape
    Hkv = k.shape[2]
    qg = q.float().view(B, N, Hkv, Hq // Hkv, D)

    def scale(t, dims):
        return (t.abs().amax(dim=dims) / FP8_MAX).clamp_min(1e-12)  # [B, Hkv]

    dq, dk, dv = scale(qg, (1, 3, 4)), scale(k.float(), (1, 3)), scale(v.float(), (1, 3))
    q8 = (qg / dq[:, None, :, None, None]).view(B, N, Hq, D).to(FP8)
    k8 = (k.float() / dk[:, None, :, None]).to(FP8)
    v8 = (v.float() / dv[:, None, :, None]).to(FP8)
    return q8.contiguous(), k8.contiguous(), v8.contiguous(), dq.contiguous(), dk.contiguous(), dv.contiguous()


def dequant(t8, d, Hkv):
    """fp8 [B,N,H,D] * descale[B,Hkv] -> fp32 (q heads map to kv head h // G)."""
    B, N, H, D = t8.shape
    return (t8.float().view(B, N, Hkv, H // Hkv, D) * d[:, None, :, None, None]).view(B, N, H, D)


def impl_args(precision, q, k, v, causal):
    if precision == "bf16":
        return (q, k, v, causal)
    return (*quantize_fp8(q, k, v), causal)


def real_values(precision, args):
    """The exact real-valued inputs the kernel sees (fp32), for the reference."""
    if precision == "bf16":
        q, k, v, _ = args
        return q.float(), k.float(), v.float()
    q8, k8, v8, dq, dk, dv, _ = args
    Hkv = k8.shape[2]
    return dequant(q8, dq, Hkv), dequant(k8, dk, Hkv), dequant(v8, dv, Hkv)


# ----------------------------------------------------------------------------- reference + emulation
def reference(qf, kf, vf, causal, p_dtype, budget=2e9):
    """fp64 reference (O_ref, LSE_ref) and the full-row low-precision emulation O_emu (fp32 scores, exact row max,
    p rounded to p_dtype before P.V, bf16 out). Inputs fp32 [B,N,H,D]. Loops over (b, kv head, q-head chunk)."""
    B, N, Hq, D = qf.shape
    Hkv = kf.shape[2]
    G = Hq // Hkv
    sc = 1.0 / math.sqrt(D)
    o_ref = torch.empty(B, N, Hq, D, device="cuda", dtype=torch.float64)
    o_emu = torch.empty(B, N, Hq, D, device="cuda", dtype=torch.float32)
    lse_ref = torch.empty(B, Hq, N, device="cuda", dtype=torch.float64)
    mask = torch.ones(N, N, device="cuda", dtype=torch.bool).triu(1) if causal else None  # True = masked
    chunk = max(1, min(G, int(budget // (N * N * 8))))
    for b in range(B):
        for hk in range(Hkv):
            k32, v32 = kf[b, :, hk], vf[b, :, hk]  # [N, D]
            k64, v64 = k32.double(), v32.double()
            for h0 in range(hk * G, (hk + 1) * G, chunk):
                hs = slice(h0, min(h0 + chunk, (hk + 1) * G))
                q32 = qf[b, :, hs].transpose(0, 1)  # [c, N, D]
                s64 = (q32.double() @ k64.T) * sc
                if causal:
                    s64.masked_fill_(mask, -math.inf)
                lse = torch.logsumexp(s64, dim=-1)
                o_ref[b, :, hs] = (torch.exp(s64 - lse[..., None]) @ v64).transpose(0, 1)
                lse_ref[b, hs] = lse
                del s64
                s32 = (q32 @ k32.T) * sc
                if causal:
                    s32.masked_fill_(mask, -math.inf)
                p = torch.exp(s32 - s32.amax(dim=-1, keepdim=True))
                l = p.sum(dim=-1, keepdim=True)
                o = (p.to(p_dtype).float() @ v32) / l
                o_emu[b, :, hs] = o.to(torch.bfloat16).float().transpose(0, 1)
                del s32, p
    return o_ref, lse_ref, o_emu


def online_emu(q, k, v, causal, descales=None, p_dtype=torch.bfloat16, BN=128):
    """Blockwise online-softmax attention in torch, the way a FlashAttention kernel does it: running max m and sum l,
    O rescaled when m grows, p rounded to p_dtype before P.V. Used by torch_ref*.py as the harness smoke test.
    Takes kernel-contract inputs, returns (o bf16 [B,N,Hq,D], lse fp32 [B,Hq,N])."""
    B, N, Hq, D = q.shape
    Hkv = k.shape[2]
    if descales is None:
        qf, kf, vf = q.float(), k.float(), v.float()
    else:
        dq, dk, dv = descales
        qf, kf, vf = dequant(q, dq, Hkv), dequant(k, dk, Hkv), dequant(v, dv, Hkv)
    idx = torch.arange(Hq, device=q.device) // (Hq // Hkv)
    qf = qf.transpose(1, 2)  # [B, Hq, N, D]
    kf = kf.transpose(1, 2).index_select(1, idx)  # [B, Hq, N, D] (GQA expand)
    vf = vf.transpose(1, 2).index_select(1, idx)
    c = 1.0 / math.sqrt(D)
    m = torch.full((B, Hq, N), -math.inf, device=q.device)
    l = torch.zeros(B, Hq, N, device=q.device)
    acc = torch.zeros(B, Hq, N, D, device=q.device)
    rows = torch.arange(N, device=q.device)
    for j0 in range(0, N, BN):
        kb, vb = kf[:, :, j0:j0 + BN], vf[:, :, j0:j0 + BN]
        s = (qf @ kb.transpose(-1, -2)) * c  # [B, Hq, N, bn]
        if causal:
            cols = torch.arange(j0, min(j0 + BN, N), device=q.device)
            s = s.masked_fill(cols[None, :] > rows[:, None], -math.inf)
        m_new = torch.maximum(m, s.amax(dim=-1))
        alpha = torch.exp(m - m_new)
        p = torch.exp(s - m_new[..., None])
        l = l * alpha + p.sum(dim=-1)
        acc = acc * alpha[..., None] + p.to(p_dtype).float() @ vb
        m = m_new
    o = (acc / l[..., None]).transpose(1, 2).to(torch.bfloat16).contiguous()
    return o, (m + torch.log(l)).contiguous()


# ----------------------------------------------------------------------------- correctness
def check_one(out, shape, ref, strict=True):
    """Returns (errors, info string). `ref` = (o_ref, lse_ref, o_emu)."""
    B, N, Hq, D = shape
    o_ref, lse_ref, o_emu = ref
    if isinstance(out, torch.Tensor):
        out = (out, None)
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        return [f"expected (o, lse), got {type(out).__name__}"], ""
    o, lse = out
    errs = []
    if strict:
        for name, t, shp, dt in (("o", o, (B, N, Hq, D), torch.bfloat16), ("lse", lse, (B, Hq, N), torch.float32)):
            if t is None or tuple(t.shape) != shp or t.dtype != dt or not t.is_contiguous():
                desc = "None" if t is None else f"shape={tuple(t.shape)} dtype={t.dtype} contiguous={t.is_contiguous()}"
                errs.append(f"{name}: {desc}, want {shp} {dt} contiguous")
        if errs:
            return errs, ""
    if tuple(o.shape) != (B, N, Hq, D):
        return [f"o: shape {tuple(o.shape)}, want {(B, N, Hq, D)}"], ""

    od = o.double()
    if not torch.isfinite(od).all():
        return [f"o: {(~torch.isfinite(od)).sum().item()} non-finite values"], ""
    e = (od - o_ref).abs()
    e_emu = (o_emu.double() - o_ref).abs()
    mx, mean, mx_emu, mean_emu = e.max().item(), e.mean().item(), e_emu.max().item(), e_emu.mean().item()
    b_max, b_mean = 2 * mx_emu + 2.0 ** -14, 2 * mean_emu + 2.0 ** -16
    if mx > b_max:
        i = e.view(-1).argmax().item()
        bi, ni, hi, di = (int(x) for x in torch.unravel_index(torch.tensor(i), e.shape))
        errs.append(f"o: max err {mx:.3g} > bound {b_max:.3g} (emu {mx_emu:.3g}); worst at b={bi} n={ni} h={hi} d={di}: "
                    f"{od[bi, ni, hi, di].item():.5g} vs {o_ref[bi, ni, hi, di].item():.5g}")
    if mean > b_mean:
        errs.append(f"o: mean err {mean:.3g} > bound {b_mean:.3g} (emu {mean_emu:.3g})")
    info = f"o max {mx:.2e} ({mx / max(mx_emu, 1e-30):4.2f}x emu)  mean {mean:.2e} ({mean / max(mean_emu, 1e-30):4.2f}x emu)"

    if lse is not None and tuple(lse.shape) == (B, Hq, N):
        ld = lse.double()
        el = (ld - lse_ref).abs() / lse_ref.abs().clamp_min(1)
        lw = el.nan_to_num(math.inf).max().item()
        if lw > 2.0 ** -10:
            i = el.view(-1).nan_to_num(math.inf).argmax().item()
            bi, hi, ni = (int(x) for x in torch.unravel_index(torch.tensor(i), el.shape))
            errs.append(f"lse: {(~(el <= 2.0 ** -10)).sum().item()} rows off by > 2^-10 rel; worst b={bi} h={hi} n={ni}: "
                        f"{ld[bi, hi, ni].item():.6g} vs {lse_ref[bi, hi, ni].item():.6g}")
        info += f"  lse rel {lw:.1e}"
    elif strict:
        errs.append("lse: missing or wrong shape")
    else:
        info += "  lse n/a"
    return errs, info


def _bitwise_equal(a, b):
    if isinstance(a, torch.Tensor):
        a, b = (a,), (b,)
    return all(x is None and y is None or
               (x is not None and y is not None and torch.equal(x.view(torch.uint8), y.view(torch.uint8)))
               for x, y in zip(a, b))


def check(fn, precision, baselines=(), cases=CASES):
    """Runs the impl (strict) and each (name, fn) baseline (O-only, non-strict) on every case."""
    errs, info = [], []
    p_dtype = FP8 if precision == "fp8" else torch.bfloat16
    for B, N, Hq, Hkv, D, causal, dist, seed in cases:
        tag = f"[{B}x{N} h{Hq}/{Hkv} {'causal' if causal else 'full'} {dist}]"
        args = impl_args(precision, *make_inputs(B, N, Hq, Hkv, D, seed, dist), causal)
        ref = reference(*real_values(precision, args), causal, p_dtype)
        saved = [a.clone() for a in args if isinstance(a, torch.Tensor)]
        for name, f in (("impl", fn), *baselines):
            strict = name == "impl"
            try:
                out = f(*args)
                torch.cuda.synchronize()
            except Exception as ex:  # noqa: BLE001
                if strict:
                    raise
                info.append(f"{name:<14} {tag:<40} {type(ex).__name__}: {str(ex).splitlines()[0][:100]}")
                continue
            e, s = check_one(out, (B, N, Hq, D), ref, strict)
            info.append(f"{name:<14} {tag:<40} {s}")
            if strict:
                errs += [f"{tag} {m}" for m in e]
                if not e and not _bitwise_equal(out, fn(*args)):
                    errs.append(f"{tag} determinism: two calls differ")
            else:
                info += [f"{'':<14} {tag:<40} FAILS {m}" for m in e]
            del out
        if not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                   for a, b in zip([a for a in args if isinstance(a, torch.Tensor)], saved)):
            errs.append(f"{tag} inputs were modified")
        del args, ref, saved
        torch.cuda.empty_cache()
    return errs, info


# ----------------------------------------------------------------------------- baselines
def _sdpa(backend):
    from torch.nn.attention import sdpa_kernel

    def fn(q, k, v, causal):
        with sdpa_kernel(backend):
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               is_causal=causal, enable_gqa=q.shape[2] != k.shape[2])
        return o.transpose(1, 2), None

    return fn


def _fa3():
    """FlashAttention-3 (Hopper) prebuilt from the HF kernels hub. Returns flash_attn_func or raises."""
    from kernels import get_kernel

    return get_kernel("kernels-community/flash-attn3", revision="main").flash_attn_func


def _fa3_out(r):
    return (r[0], r[1]) if isinstance(r, (tuple, list)) else (r, None)


def baselines():
    """[(name, precision, fn)] with fn taking that precision's args. Failures are printed and skipped."""
    from torch.nn.attention import SDPBackend

    b = [("sdpa cudnn", "bf16", _sdpa(SDPBackend.CUDNN_ATTENTION)),
         ("sdpa flash", "bf16", _sdpa(SDPBackend.FLASH_ATTENTION))]
    try:
        fa3 = _fa3()
        b.append(("FA3 bf16", "bf16", lambda q, k, v, causal: _fa3_out(fa3(q, k, v, causal=causal))))
        b.append(("FA3 fp8", "fp8", lambda q, k, v, dq, dk, dv, causal: _fa3_out(
            fa3(q, k, v, causal=causal, q_descale=dq, k_descale=dk, v_descale=dv))))
    except Exception as e:  # noqa: BLE001
        print(f"FA3 (kernels hub) unavailable: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    return b


# ----------------------------------------------------------------------------- driver
def _fmt(name, us, fl, prec):
    tf = fl / (us * 1e-6) / 1e12
    return f"    {name:<22} {us:9.1f} us  {tf:7.1f} TFLOP/s  {100 * tf * 1e12 / PEAK[prec]:5.1f}% of {prec} peak"


def run(impl, mode="bench") -> str:
    torch.manual_seed(0)
    prec = getattr(impl, "PRECISION", "bf16")
    assert prec in PEAK, prec
    fn = impl.make()

    if mode == "profile":  # main shape: one warm call, then NVTX-tagged calls for ncu
        B, N, Hq, Hkv, D, causal = BENCH_SHAPES[0]
        sets = [impl_args(prec, *make_inputs(B, N, Hq, Hkv, D, 100 + i), causal) for i in range(3)]
        fn(*sets[0])
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile")
        for s in sets[1:3]:
            fn(*s)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return ""

    lines = [f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | impl precision: {prec} | "
             f"peak {PEAK[prec] / 1e12:.0f} TFLOP/s"]
    bl = baselines()
    errs, info = check(fn, prec, [(n, f) for n, p, f in bl if p == prec])
    lines.append("correctness: PASS" if not errs else "correctness: FAIL\n  " + "\n  ".join(errs))
    lines += ["  " + s for s in info]

    lines.append("CUDA graph, 2 rotating input sets, median per call:")
    for B, N, Hq, Hkv, D, causal in BENCH_SHAPES:
        fl = flops(B, N, Hq, D, causal)
        raw = [make_inputs(B, N, Hq, Hkv, D, 100 + i) for i in range(2)]
        sets = {"bf16": [(*r, causal) for r in raw]}
        sets["fp8"] = [impl_args("fp8", *r, causal) for r in raw] if prec == "fp8" or any(
            p == "fp8" for _, p, _ in bl) else None
        lines.append(f"  B={B} N={N} Hq={Hq} Hkv={Hkv} D={D} {'causal' if causal else 'non-causal'}: "
                     f"{fl / 1e12:.3f} TFLOP, floor {fl / PEAK[prec] * 1e6:.0f} us ({prec})")
        rows = [("** impl **", prec, fn)] + bl
        for name, p, f in rows:
            try:
                med, mn = time_graph(f, sets[p])
                lines.append(_fmt(name, med, fl, p) + f"   (min {mn:.1f})")
            except Exception as e:  # noqa: BLE001
                if name == "** impl **":
                    raise
                lines.append(f"    {name:<22} {type(e).__name__}: {str(e).splitlines()[0][:110]}")
        del raw, sets
        torch.cuda.empty_cache()
    return "\n".join(lines)


if __name__ == "__main__":  # python harness.py [impl_module]
    import importlib
    import sys

    print(run(importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "torch_ref")))
