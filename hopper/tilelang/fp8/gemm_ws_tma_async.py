"""FP8 block-scaled GEMM (DeepSeek-V3 / DeepGEMM scaling), hand-written warp specialization.

    C[M, N] = sum_kb  SA[m, kb] * SB[n/128, kb] * ( Aq[m, kb*128:(kb+1)*128] . Bq[n, kb*128:(kb+1)*128] )

The fp8 port of ../gemm_ws_tma_async.py (TMA + mbarriers + threadblock swizzle + explicit
wgmma), with the "2xAcc" promotion from tilelang's examples/deepseek_deepgemm. No overlap between promotion
and MMA beyond what the two consumer warpgroups give for free.

    block = 384 threads = 3 warpgroups, tile 128 x 128, block_K = 128 = one quantization group
        WG0  consumer: tile rows  0..63  -> part0 (64x128 fp32, 64 regs) + acc0 (64 regs)
        WG1  consumer: tile rows 64..127 -> part1 + acc1
        WG2  producer: TMA A (128x128 fp8 = 16 KB) + B (16 KB) per stage, and each of its 128 threads writes
                       one row's combined scale  SA[m, kb] * SB[n_blk, kb]  into scale_s[s, row]

Why two 64-row consumers instead of one 128-row consumer as in the bf16 version: promotion needs a second
accumulator. One WG owning 128x128 would hold 128 (partial) + 128 (acc) = 256 regs/thread > 232.
With m64 per WG it is 64 + 64. This is DeepGEMM's layout too (BLOCK_M=128 = 2 math WGs).

Per k-group, consumer w:
    wait full[s]
    wgmma_gemm(A_s[s, w-rows], B_s[s]) -> part      clear_accum: scale-d = 0 on the first of the 4 k32 wgmmas
    wait_wgmma(0)                                   promotion needs this group's partial finished: no wait<1> here
    acc += part * scale_s[s, row]                   one FMA per element (SA*SB already folded by the producer)
    arrive empty[s]                                 after the scale_s reads, so the producer can't overwrite them

Why the scales go through smem and the producer: SA is [M, K/128] row-major, so a tile's 128 scales for one k-group
are a strided column (stride 256 B). TMA can't load that (inner box must be >= 16 B), and problem_desc forbids
transposing outside the call. So the producer's 128 threads each do one plain ld.global, multiply by the SB scalar,
st.shared, and then the mbarrier arrive they were already doing. mbarrier.arrive has release semantics and
try_wait acquire semantics (CTA scope), so the consumers' reads after the wait see the stores. Same argument
in reverse for empty[s]: consumers read scale_s before they arrive.

Barriers (same shape as the bf16 2-consumer version):
    full[s]   count 128: the producer WG's 128 arrivals (+ 32 KB of TMA tx bytes via tma_copy's expect_tx)
    empty[s]  count 256: all threads of WG0 + WG1

Knobs: stages (smem 32.5 KB each: 4 = 130 KB, 6 = 195 KB), swizzle (threadblock raster panel size, 0 = off).
"""

import os

import torch
import tilelang
import tilelang.language as T

G = 128  # quantization group along K (A and B), block size along N (B)
WG_M = 64  # tile rows per consumer warpgroup (one m64 wgmma)
N_CONSUMERS = 2
CONSUMER_WGS = tuple(range(N_CONSUMERS))  # a Python container: the eager frontend unrolls it at trace time
# (a `for w in range(...)` inside the kernel becomes a T.serial loop -> w is a Var -> T.ws(w) rejects it)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def fp8_gemm_ws(A, B, SA, SB, stages: int = 4, swizzle: int = 8, hoist: bool = True):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float8_e4m3fn)
    B: T.Tensor((N, K), T.float8_e4m3fn)
    SA: T.Tensor((M, K // G), T.float32)
    SB: T.Tensor((N // G, K // G), T.float32)
    C = T.empty((M, N), T.bfloat16)

    S = stages
    block_M = N_CONSUMERS * WG_M  # 128
    block_N = G  # 128: one SB block per tile -> SB is a single scalar per k-group
    block_K = G  # 128: one quantization group per stage -> 4 x wgmma k32 per stage
    n_k = K // block_K
    barrier_counts = [128] * S + [128 * N_CONSUMERS] * S  # [0,S) = full[s], [S,2S) = empty[s]

    with T.Kernel(N // block_N, M // block_M, threads=128 * (N_CONSUMERS + 1)) as (bx, by):
        T.use_swizzle(panel_size=swizzle, enable=swizzle > 0)
        tx = T.get_thread_binding()

        A_s = T.alloc_shared((S, block_M, block_K), T.float8_e4m3fn)  # S x 16 KB
        B_s = T.alloc_shared((S, block_N, block_K), T.float8_e4m3fn)  # S x 16 KB
        scale_s = T.alloc_shared((S, block_M), T.float32)  # S x 512 B: SA[row, kb] * SB[bx, kb]
        part = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]  # wgmma accumulator
        acc = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]  # promoted fp32 sum
        bars = T.alloc_barrier(barrier_counts)

        # ---------------- producer warpgroup ----------------
        with T.ws(N_CONSUMERS):
            T.dec_max_nreg(40)
            row = tx - 128 * N_CONSUMERS  # 0..127: the tile row whose scale this thread fetches
            sc = T.alloc_var(T.float32)
            for ko in T.serial(n_k):
                s = ko % S
                phase = (ko // S) & 1
                if hoist:  # issue the scale loads first: their latency overlaps the empty wait + TMA issue
                    sc = SA[by * block_M + row, ko] * SB[bx, ko]
                T.mbarrier_wait_parity(bars[S + s], phase ^ 1)  # slot s free? (both consumers arrived)
                T.tma_copy(A[by * block_M:(by + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                           A_s[s, :, :], barrier=bars[s])
                T.tma_copy(B[bx * block_N:(bx + 1) * block_N, ko * block_K:(ko + 1) * block_K],
                           B_s[s, :, :], barrier=bars[s])
                if not hoist:  # loads issued after the wait: the producer stalls on them every k-group
                    sc = SA[by * block_M + row, ko] * SB[bx, ko]
                scale_s[s, row] = sc
                T.mbarrier_arrive(bars[s])  # release: orders the scale_s store before the phase flip

        # ---------------- consumers: WG w owns tile rows [w*64, (w+1)*64) ----------------
        for w in CONSUMER_WGS:
            with T.ws(w):
                T.inc_max_nreg(232)
                T.clear(acc[w])
                for ko in T.serial(n_k):
                    s = ko % S
                    phase = (ko // S) & 1
                    T.mbarrier_wait_parity(bars[s], phase)  # A, B and scale_s for stage s landed?
                    T.wgmma_gemm(A_s[s, w * WG_M:(w + 1) * WG_M, :], B_s[s, :, :], part[w],
                                 transpose_B=True, clear_accum=True)
                    T.wait_wgmma(0)
                    for i, j in T.Parallel(WG_M, block_N):
                        acc[w][i, j] += part[w][i, j] * scale_s[s, w * WG_M + i]
                    T.mbarrier_arrive(bars[S + s])  # my 128 of 256 arrivals: A/B/scale_s[s] no longer read
                T.copy(acc[w], C[by * block_M + w * WG_M, bx * block_N])

    return C


def make(stages: int = 4, swizzle: int = 8, hoist: bool = True):
    """Harness contract: fn(a, b, sa, sb) -> c bf16."""

    def fn(a, b, sa, sb):
        return fp8_gemm_ws(a, b, sa, sb, stages=stages, swizzle=swizzle, hoist=hoist)

    return fn


# ----------------------------------------------------------------------------- standalone: correctness + sweep


def bench_interleaved(fns: dict, rounds=7, iters=10):
    """Round-robin timing so clock/power throttling hits every variant equally; returns median ms per call."""
    times = {k: [] for k in fns}
    for f in fns.values():
        f()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for name, f in fns.items():
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(iters):
                f()
            en.record()
            en.synchronize()
            times[name].append(st.elapsed_time(en) / iters)
    return {k: sorted(v)[len(v) // 2] for k, v in times.items()}


def main():
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from harness import M, N, K, make_inputs  # noqa: E402

    a, b, sa, sb = make_inputs(0, "scale_stress")
    configs = [(4, 8, False), (4, 8, True), (6, 8, True), (4, 4, True), (4, 12, True)]  # (stages, swizzle, hoist)
    fns = {f"S={st} P={p}{' hoist' if h else ''}": make(st, p, h) for st, p, h in configs}

    if os.environ.get("PROFILE"):  # under ncu: one launch each, in config order
        for f in fns.values():
            f(a, b, sa, sb)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile")
        for f in fns.values():
            f(a, b, sa, sb)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return

    a32 = a.float() * sa.repeat_interleave(128, dim=1)
    b32 = b.float() * sb.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = a32 @ b32.T
    for name, f in fns.items():
        rel = ((f(a, b, sa, sb).float() - ref).abs().max() / ref.abs().max()).item()
        print(f"  {name:<16} max|err|/max|ref| = {rel:.2e}")

    flops = 2 * M * N * K
    for name, ms in bench_interleaved({k: (lambda f=f: f(a, b, sa, sb)) for k, f in fns.items()}).items():
        print(f"  {name:<16} {ms * 1e3:8.1f} us   {flops / ms / 1e9:7.1f} TFLOP/s")


if __name__ == "__main__":
    main()
