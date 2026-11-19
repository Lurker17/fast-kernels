"""Naive UNFUSED baseline in tilelang: three separate streaming kernels, y round-trips through HBM in fp32.

    k1 add   : h = bf16(x + residual)                      read 2+2 B, write 2 B per element   (6 B)
    k2 norm  : y = h * rsqrt(mean(h^2) + eps) * weight      read 2 B,   write 4 B (fp32 y)      (6 B)
    k3 quant : s = max(amax_128(|y|), 1e-10) / 448,        read 4 B,   write 1 B + 4 B/128     (5.03 B)
               q = fp8(clamp(y / s))
    total 17 B/element vs 7 B fused  ->  at ~3 TB/s: ~670 us at 16384 x 7168 (fused: ~275 us)

Each kernel: one CTA per row, 128 threads, 128-bit vectors (8 bf16 or 2 x 4 fp32 per thread per step),
H / 1024 steps per thread (7 at H = 7168). Every thread issues ALL its loads for the row first (straight-line,
`T.unroll`), then computes and stores: 7 x 16 B per input in flight per thread instead of 1.
Reductions: warp shuffles (`T.warp_reduce_sum`, `T.shfl_xor`), then 4 warp partials through shared memory (k2);
k3's group amax stays inside 16 lanes (one group of 128 fp32 = 16 lanes x 8), no shared memory.

Run (harness):  python harness.py naive_unfused
Run (per-kernel bandwidth): python naive_unfused.py
"""

import os
import sys

import torch
import tilelang
import tilelang.language as T

G = 128
EPS = 1e-6
FP8_MAX = 448.0
AMAX_MIN = 1e-10
THREADS = 128
VEC = 8  # elements per thread per step: 16 B of bf16 (k1, k2) / 32 B of fp32 (k2 store, k3)
ROW_STEP = THREADS * VEC  # 1024 elements of a row per step


@tilelang.jit
def add_kernel(X, R):
    M, H = T.const("M, H")
    X: T.Tensor((M, H), T.bfloat16)
    R: T.Tensor((M, H), T.bfloat16)
    Ho = T.empty((M, H), T.bfloat16)
    assert H % ROW_STEP == 0, "naive kernels assume H % 1024 == 0"
    ITERS = H // ROW_STEP

    with T.Kernel(M, threads=THREADS) as row:
        tx = T.get_thread_binding()
        xv = T.alloc_local((ITERS, VEC), T.bfloat16)
        rv = T.alloc_local((ITERS, VEC), T.bfloat16)
        ov = T.alloc_local((VEC,), T.bfloat16)
        for i in T.unroll(ITERS):  # all loads first: 2 x ITERS x 16 B in flight per thread
            for v in T.vectorized(VEC):
                xv[i, v] = X[row, (i * THREADS + tx) * VEC + v]
            for v in T.vectorized(VEC):
                rv[i, v] = R[row, (i * THREADS + tx) * VEC + v]
        for i in T.unroll(ITERS):
            for v in T.vectorized(VEC):  # add in fp32, one rounding to bf16 (the spec's h)
                ov[v] = T.cast(T.cast(xv[i, v], T.float32) + T.cast(rv[i, v], T.float32), T.bfloat16)
            for v in T.vectorized(VEC):
                Ho[row, (i * THREADS + tx) * VEC + v] = ov[v]
    return Ho


@tilelang.jit
def norm_kernel(Hin, W):
    M, H = T.const("M, H")
    Hin: T.Tensor((M, H), T.bfloat16)
    W: T.Tensor((H,), T.bfloat16)
    Y = T.empty((M, H), T.float32)
    assert H % ROW_STEP == 0
    ITERS = H // ROW_STEP
    WARPS = THREADS // 32

    with T.Kernel(M, threads=THREADS) as row:
        tx = T.get_thread_binding()
        hv = T.alloc_local((ITERS, VEC), T.bfloat16)
        wv = T.alloc_local((VEC,), T.bfloat16)
        yv = T.alloc_local((VEC,), T.float32)
        red = T.alloc_shared((WARPS,), T.float32)
        ss = T.alloc_var(T.float32)
        rstd = T.alloc_var(T.float32)
        for i in T.unroll(ITERS):  # the whole row's share in registers: ITERS x 16 B in flight
            for v in T.vectorized(VEC):
                hv[i, v] = Hin[row, (i * THREADS + tx) * VEC + v]
        ss = 0.0
        for i in T.unroll(ITERS):
            for v in T.unroll(VEC):
                ss = ss + T.cast(hv[i, v], T.float32) * T.cast(hv[i, v], T.float32)
        ss = T.warp_reduce_sum(ss)  # all 32 lanes hold the warp's sum
        if tx % 32 == 0:
            red[tx // 32] = ss
        T.sync_threads()
        rstd = T.rsqrt((red[0] + red[1] + red[2] + red[3]) / H + EPS)  # fixed order: deterministic
        for i in T.unroll(ITERS):
            for v in T.vectorized(VEC):  # weight is 14 KB, L2/L1-resident after the first rows
                wv[v] = W[(i * THREADS + tx) * VEC + v]
            for v in T.vectorized(VEC):
                yv[v] = T.cast(hv[i, v], T.float32) * rstd * T.cast(wv[v], T.float32)
            for v in T.vectorized(VEC):  # 32 B per thread -> 2 x st.global.v4.f32
                Y[row, (i * THREADS + tx) * VEC + v] = yv[v]
    return Y


@tilelang.jit
def quant_kernel(Y, Q):
    """Q is an OUT-parameter allocated by the caller: T.empty of an fp8 tensor fails in tilelang 0.1.15
    (its output allocator goes through torch's DLPack bridge, which rejects the fp8 dtype: "Unsupported code 10")."""
    M, H = T.const("M, H")
    Y: T.Tensor((M, H), T.float32)
    Q: T.Tensor((M, H), T.float8_e4m3fn)
    S = T.empty((M, H // G), T.float32)
    assert H % ROW_STEP == 0
    ITERS = H // ROW_STEP  # each step: 8 groups of 128 = 1024 elements; 16 lanes per group, 8 fp32 per lane
    LANES = G // VEC  # 16

    with T.Kernel(M, threads=THREADS) as row:
        tx = T.get_thread_binding()
        yv = T.alloc_local((ITERS, VEC), T.float32)
        qv = T.alloc_local((VEC,), T.float8_e4m3fn)
        amax = T.alloc_var(T.float32)
        sc = T.alloc_var(T.float32)
        gi = tx // LANES  # group within the step
        ln = tx % LANES  # lane within the group
        for i in T.unroll(ITERS):  # all loads first: ITERS x 32 B in flight per thread
            for v in T.vectorized(VEC):
                yv[i, v] = Y[row, (i * 8 + gi) * G + ln * VEC + v]
        for i in T.unroll(ITERS):
            amax = 0.0
            for v in T.unroll(VEC):
                amax = T.max(amax, T.abs(yv[i, v]))
            for d in T.unroll(4):  # xor 8, 4, 2, 1: stays inside each 16-lane half of the warp = one group
                amax = T.max(amax, T.shfl_xor(amax, 8 >> d))
            sc = T.max(amax, AMAX_MIN) / FP8_MAX
            for v in T.vectorized(VEC):
                qv[v] = T.cast(T.clamp(yv[i, v] / sc, -FP8_MAX, FP8_MAX), T.float8_e4m3fn)
            for v in T.vectorized(VEC):  # 8 B per thread
                Q[row, (i * 8 + gi) * G + ln * VEC + v] = qv[v]
            if ln == 0:
                S[row, i * 8 + gi] = sc
    return S


def make():
    """Harness contract: fn(x, residual, weight) -> (h, q, s)."""

    def fn(x, r, w):
        h = add_kernel(x, r)
        y = norm_kernel(h, w)
        q = torch.empty(y.shape, device=y.device, dtype=torch.float8_e4m3fn)
        s = quant_kernel(y, q)
        return h, q, s

    return fn


# ----------------------------------------------------------------------------- standalone: per-kernel bandwidth


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from harness import BENCH_SHAPES, HBM_BW, make_inputs, n_sets, problem_bytes  # noqa: E402

    try:
        from common.bench import time_graph
    except ImportError:
        from bench import time_graph

    for M, H in BENCH_SHAPES:
        k = n_sets(M, H)
        sets = [make_inputs(M, H, 100 + i) for i in range(k)]
        hs = [add_kernel(x, r) for x, r, _ in sets]
        ys = [norm_kernel(h, w) for h, (_, _, w) in zip(hs, sets)]
        e = M * H
        rows = [  # name, fn, arg sets, bytes moved by that kernel
            ("k1 add  (r 4 B, w 2 B)", lambda x, r: add_kernel(x, r), [(x, r) for x, r, _ in sets], e * 6),
            ("k2 norm (r 2 B, w 4 B)", lambda h, w: norm_kernel(h, w), [(h, w) for h, (_, _, w) in zip(hs, sets)],
             e * 6 + H * 2),
            ("k3 quant(r 4 B, w 1 B)", lambda y, q: quant_kernel(y, q),
             [(y, torch.empty(y.shape, device="cuda", dtype=torch.float8_e4m3fn)) for y in ys],
             e * 5 + M * (H // G) * 4),
            ("x + residual (torch)", lambda x, r: x + r, [(x, r) for x, r, _ in sets], e * 6),
        ]
        print(f"{M} x {H}  ({k} input sets; fused-problem floor {problem_bytes(M, H) / HBM_BW * 1e6:.1f} us)")
        total = 0.0
        for name, f, args, nb in rows:
            med, mn = time_graph(f, args)
            if name.startswith("k"):
                total += med
            print(f"  {name:<24} {med:9.2f} us  {nb / (med * 1e-6) / 1e9:7.0f} GB/s  "
                  f"{100 * nb / (med * 1e-6) / HBM_BW:5.1f}% of HBM")
        fn = make()
        med, _ = time_graph(fn, sets)
        print(f"  {'all three (make())':<24} {med:9.2f} us  (sum of parts {total:.2f} us); "
              f"vs fused floor: {problem_bytes(M, H) / (med * 1e-6) / 1e9:.0f} GB/s effective")
        del sets, hs, ys
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
