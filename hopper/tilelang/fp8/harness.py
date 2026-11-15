"""Correctness + benchmark harness for the FP8 block-scaled GEMM (see problem_desc.md).

Impl contract: module exposes `make()` returning
    fn(a[M,K] fp8e4m3, b[N,K] fp8e4m3, sa[M,K/128] fp32, sb[N/128,K/128] fp32) -> c[M,N] bf16
"""

import torch
import torch.nn.functional as F

try:
    from common.bench import time_graph  # this repo's layout (Modal harness)
except ImportError:  # standalone copy: bench.py next to this file
    from bench import time_graph

M = N = K = 8192
G = 128  # quantization group along K (and block size along N for B)
FP8_MAX = 448.0
FP8_PEAK = 1979e12  # H100 SXM dense fp8 (spec sheet)
FLOPS = 2 * M * N * K
N_SETS = 4  # each set: A 67 MB + B 67 MB > 50 MB L2
FP8 = torch.float8_e4m3fn


# ----------------------------------------------------------------------------- inputs
def quant_1x128(x):
    """[R, C] float -> fp8 [R, C], scales [R, C/128]. Each 1x128 group's amax maps to 448."""
    r, c = x.shape
    v = x.view(r, c // G, G)
    amax = v.abs().amax(dim=2).clamp_min(1e-4)
    q = (v * (FP8_MAX / amax).unsqueeze(2)).to(FP8).view(r, c)
    return q, (amax / FP8_MAX).float().contiguous()


def quant_128x128(x):
    """[R, C] float -> fp8 [R, C], scales [R/128, C/128]. Each 128x128 block's amax maps to 448."""
    r, c = x.shape
    v = x.view(r // G, G, c // G, G)
    amax = v.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    q = (v * (FP8_MAX / amax)).to(FP8).view(r, c)
    return q, (amax / FP8_MAX).view(r // G, c // G).float().contiguous()


def make_inputs(seed=0, dist="normal"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if dist == "normal":
        a = torch.randn(M, K, device="cuda", generator=g)
        b = torch.randn(N, K, device="cuda", generator=g) * 0.02
    elif dist == "uniform_pos":  # no cancellation: sums grow with K, stresses accumulation precision
        a = torch.rand(M, K, device="cuda", generator=g)
        b = torch.rand(N, K, device="cuda", generator=g)
    elif dist == "scale_stress":  # group magnitudes span 2^16 -> wrong scale indexing is a big error
        a = torch.randn(M, K, device="cuda", generator=g)
        b = torch.randn(N, K, device="cuda", generator=g)
        ea = torch.randint(-8, 9, (M, K // G), device="cuda", generator=g).float().exp2()
        eb = torch.randint(-8, 9, (N // G, K // G), device="cuda", generator=g).float().exp2()
        a = a * ea.repeat_interleave(G, dim=1)
        b = b * eb.repeat_interleave(G, dim=0).repeat_interleave(G, dim=1)
    else:
        raise ValueError(dist)
    qa, sa = quant_1x128(a)
    qb, sb = quant_128x128(b)
    return qa, qb, sa, sb


def dequant64(qa, qb, sa, sb):
    a = qa.double() * sa.double().repeat_interleave(G, dim=1)
    b = qb.double() * sb.double().repeat_interleave(G, dim=0).repeat_interleave(G, dim=1)
    return a, b


# ----------------------------------------------------------------------------- correctness
def accuracy(out, a64, b64):
    """Returns (n_bad, worst err/bound, message for the worst element)."""
    ref = a64 @ b64.T
    bound = 2.0 ** -8 * ref.abs() + 2.0 ** -11 * (a64.abs() @ b64.abs().T)
    err = (out.double() - ref).abs()
    ratio = err / bound.clamp_min(1e-300)
    i = ratio.argmax().item()
    r, c = divmod(i, N)
    msg = (f"worst ({r},{c}) out={out[r, c].item():.6g} ref={ref[r, c].item():.6g} "
           f"err={err[r, c].item():.3g} bound={bound[r, c].item():.3g}")
    return int((ratio > 1).sum().item()), ratio.view(-1)[i].item(), msg


CASES = [(0, "normal"), (1, "uniform_pos"), (2, "scale_stress")]


def check(fn, name="impl") -> tuple[list[str], list[str]]:
    errs, info = [], []
    for seed, dist in CASES:
        qa, qb, sa, sb = make_inputs(seed, dist)
        out = fn(qa, qb, sa, sb)
        torch.cuda.synchronize()
        if out.shape != (M, N) or out.dtype != torch.bfloat16 or not out.is_contiguous():
            return [f"bad output: shape={tuple(out.shape)} dtype={out.dtype} contiguous={out.is_contiguous()}"], info
        a64, b64 = dequant64(qa, qb, sa, sb)
        n_bad, worst, msg = accuracy(out, a64, b64)
        del a64, b64
        info.append(f"{name:<28} [{dist:<12}] max err/bound {worst:.3g}")
        if n_bad:
            errs.append(f"[{dist}/seed{seed}] accuracy: {n_bad} elems over bound; {msg}")
        if not torch.equal(out, fn(qa, qb, sa, sb)):
            errs.append(f"[{dist}/seed{seed}] determinism: two calls differ")
    return errs, info


# ----------------------------------------------------------------------------- baselines
def _blockwise_scaled_mm():
    """cuBLAS 1x128 x 128x128 via torch._scaled_mm, if this torch/cuBLAS supports it. Tries the scale layouts torch
    has used; the accuracy check (not this probe) is what decides whether a variant interprets the scales right."""
    qa, qb, sa, sb = make_inputs(0, "scale_stress")
    a64, b64 = dequant64(qa, qb, sa, sb)
    variants = {
        "sa row-major": lambda sa, sb: (sa, sb),
        "sa col-major": lambda sa, sb: (sa.t().contiguous().t(), sb),
        "sa, sb col-major": lambda sa, sb: (sa.t().contiguous().t(), sb.t().contiguous().t()),
    }
    notes = []
    for vname, lay in variants.items():
        def fn(a, b, sa, sb, lay=lay):
            sa2, sb2 = lay(sa, sb)
            return torch._scaled_mm(a, b.t(), scale_a=sa2, scale_b=sb2, out_dtype=torch.bfloat16)
        try:
            out = fn(qa, qb, sa, sb)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            notes.append(f"{vname}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
            continue
        n_bad, worst, _ = accuracy(out, a64, b64)
        if n_bad == 0:
            return fn, f"blockwise _scaled_mm works ({vname}, scale_stress err/bound {worst:.3g})"
        notes.append(f"{vname}: runs but wrong (scale_stress err/bound {worst:.3g})")
    return None, "blockwise _scaled_mm unavailable: " + " | ".join(notes)


def _deepgemm():
    """DeepGEMM's sm90 1x128 x 128x128 kernel, if installed. Returns ({name: fn}, note).
    'as given'    : row-major SA passed straight in; any scale re-layout DeepGEMM does is inside the timed call (fair).
    'SA pre-laid' : SA converted to DeepGEMM's TMA-aligned M-major layout once, outside the timing ([speed] only:
                    problem_desc forbids that for impls, but it is the kernel's own time)."""
    try:
        import deep_gemm as dg
    except Exception as e:  # noqa: BLE001
        return {}, f"DeepGEMM unavailable: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
    gemm = getattr(dg, "fp8_gemm_nt", None) or getattr(dg, "gemm_fp8_fp8_bf16_nt", None)
    relayout = getattr(dg, "get_mn_major_tma_aligned_tensor", None) or getattr(dg, "get_col_major_tma_aligned_tensor", None)
    if gemm is None:
        return {}, f"DeepGEMM: no fp8 gemm entry point; has {[n for n in dir(dg) if 'gemm' in n][:12]}"

    def as_given(a, b, sa, sb):
        out = torch.empty(a.shape[0], b.shape[0], device="cuda", dtype=torch.bfloat16)
        gemm((a, sa), (b, sb), out)
        return out

    fns = {"DeepGEMM (as given)": as_given}
    if relayout is not None:
        cache = {}

        def pre_laid(a, b, sa, sb):
            key = sa.data_ptr()
            if key not in cache:  # first (warm-up) call per input set: outside graph capture and timing
                cache[key] = relayout(sa)
            out = torch.empty(a.shape[0], b.shape[0], device="cuda", dtype=torch.bfloat16)
            gemm((a, cache[key]), (b, sb), out)
            return out

        fns["DeepGEMM (SA pre-laid) [speed]"] = pre_laid
    return fns, f"DeepGEMM {getattr(dg, '__version__', '?')}: {gemm.__name__}, relayout={getattr(relayout, '__name__', None)}"


def baselines():
    one = torch.ones((), device="cuda")
    ones_m = torch.ones(M, 1, device="cuda")
    ones_n = torch.ones(1, N, device="cuda")
    b = {}
    bw, note = _blockwise_scaled_mm()
    if bw is not None:
        b["cuBLAS fp8 blockwise (TARGET)"] = bw
    b["cuBLAS fp8 per-tensor [speed]"] = lambda a, b_, sa, sb: torch._scaled_mm(
        a, b_.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    b["fp8 rowwise [speed]"] = lambda a, b_, sa, sb: torch._scaled_mm(
        a, b_.t(), scale_a=ones_m, scale_b=ones_n, out_dtype=torch.bfloat16)
    return b, note


# ----------------------------------------------------------------------------- driver
def _fmt(name, us):
    tf = FLOPS / (us * 1e-6) / 1e12
    return f"  {name:<32} {us / 1e3:7.3f} ms  {tf:6.0f} TFLOP/s  {100 * tf * 1e12 / FP8_PEAK:5.1f}% fp8 peak"


def run(impl, mode="bench") -> str:
    torch.manual_seed(0)
    fn = impl.make()
    sets = [make_inputs(100 + i) for i in range(N_SETS)]
    if mode == "profile":  # one warm call, then a few NVTX-tagged calls for ncu
        fn(*sets[0])
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile")
        for s in sets[1:3]:
            fn(*s)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return ""

    lines = [f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | M=N=K={M} | "
             f"{FLOPS / 1e12:.2f} TFLOP, floor {FLOPS / FP8_PEAK * 1e3:.3f} ms at fp8 peak"]
    errs, info = check(fn)
    lines.append("correctness: PASS" if not errs else "correctness: FAIL\n  " + "\n  ".join(errs))
    lines += ["  " + s for s in info]

    bl, note = baselines()
    lines.append(note)
    if "cuBLAS fp8 blockwise (TARGET)" in bl:
        _, binfo = check(bl["cuBLAS fp8 blockwise (TARGET)"], "cuBLAS blockwise")
        lines += ["  " + s for s in binfo]
    dg, dg_note = _deepgemm()
    lines.append(dg_note)
    for name, f in list(dg.items()):
        try:
            derrs, dinfo = check(f, name)
        except Exception as e:  # noqa: BLE001
            lines.append(f"  {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            continue
        lines += ["  " + s for s in dinfo] + ["  " + s for s in derrs]
        bl[name] = f

    lines.append(f"CUDA graph, {N_SETS} rotating input sets, median per call:")
    results = {"** impl **": time_graph(fn, sets)}
    for name, f in bl.items():
        results[name] = time_graph(f, sets)
    bf = [(qa.to(torch.bfloat16), qb.to(torch.bfloat16)) for qa, qb, _, _ in sets]
    results["cuBLAS bf16 F.linear [speed]"] = time_graph(lambda a, b_: F.linear(a, b_), bf)
    for name, (med, mn) in results.items():
        lines.append(_fmt(name, med) + f"   (min {mn / 1e3:.3f})")
    return "\n".join(lines)


if __name__ == "__main__":  # python harness.py [impl_module]: full correctness check + CUDA-graph benchmark
    import importlib
    import sys

    print(run(importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "gemm_ws_tma_async")))
