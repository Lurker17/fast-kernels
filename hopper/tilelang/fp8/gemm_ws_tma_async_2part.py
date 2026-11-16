"""FP8 block-scaled GEMM, v1: v0 (gemm_ws_tma_async.py) + double-buffered partial accumulator per consumer WG.

v0's consumer did   wait full -> issue k -> wait<0> -> promote k -> arrive,   so each WG drained its tensor queue
every k-group and promoted with nothing queued behind (ncu: tensor pipe 54% busy, WARPGROUP.DEPBAR ~1/3 of consumer
samples). v1 gives each WG two partial buffers and issues one k-group ahead:

    issue 0 -> pa
    loop:  issue k+1 -> pb ; wait<1> (k done) ; promote pa (k) ; arrive empty[k]      <- k+1 runs during the promotion
           issue k+2 -> pa ; wait<1>          ; promote pb (k+1) ; arrive empty[k+1]
    tail:  issue last -> pb ; wait<1> ; promote pa ; wait<0> ; promote pb

- wgmma accumulators are not scoreboarded: a buffer is rewritten only after its promotion (program order + the
  wgmma.fence tilelang emits before each batch), and read only after wait<1> confirmed its batch retired.
- At most 2 batches in flight per WG, at least 1 during every promotion.
- Registers: acc + pa + pb = 64 + 64 + 64 = 192 / thread (cap 232).
- Each WG now holds stages k and k+1 at once -> the producer's lead is one slot shorter; sweep stages 5/6.
- Producer, barriers, scales, swizzle: unchanged from v0 (see its docstring).
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


@T.macro
def issue(A_s, B_s, bars, part, ko, w, S):
    """Wait for stage ko % S to land, then queue its 4 x k32 wgmmas into `part` (overwrite: clear_accum)."""
    s = ko % S
    T.mbarrier_wait_parity(bars[s], (ko // S) & 1)
    T.wgmma_gemm(A_s[s, w * WG_M:(w + 1) * WG_M, :], B_s[s, :, :], part, transpose_B=True, clear_accum=True)


@T.macro
def promote(acc, part, scale_s, bars, ko, w, S):
    """acc += part * scale for k-group ko, then release its stage. Caller guarantees part's batch has retired."""
    s = ko % S
    for i, j in T.Parallel(WG_M, G):
        acc[i, j] += part[i, j] * scale_s[s, w * WG_M + i]
    T.mbarrier_arrive(bars[S + s])


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def fp8_gemm_ws_2part(A, B, SA, SB, stages: int = 4, swizzle: int = 8, hoist: bool = True, chunk: int = 8):
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
    assert chunk >= 2 and n_k % chunk == 0, "K loop runs in chunks of `chunk` k-groups"
    steps = tuple(range(chunk - 1))  # in-chunk steps, a Python container -> unrolled at trace time
    barrier_counts = [128] * S + [128 * N_CONSUMERS] * S  # [0,S) = full[s], [S,2S) = empty[s]

    with T.Kernel(N // block_N, M // block_M, threads=128 * (N_CONSUMERS + 1)) as (bx, by):
        T.use_swizzle(panel_size=swizzle, enable=swizzle > 0)
        tx = T.get_thread_binding()

        A_s = T.alloc_shared((S, block_M, block_K), T.float8_e4m3fn)  # S x 16 KB
        B_s = T.alloc_shared((S, block_N, block_K), T.float8_e4m3fn)  # S x 16 KB
        scale_s = T.alloc_shared((S, block_M), T.float32)  # S x 512 B: SA[row, kb] * SB[bx, kb]
        part_a = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]  # even k-groups
        part_b = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]  # odd k-groups
        acc = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]  # promoted fp32 sum
        parts = [part_a, part_b]  # parts[k % 2][w]: k-group k's partial buffer for WG w
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
                # ptxas (C7514) serializes EVERY wgmma in the kernel if a batch is still in flight across a loop
                # back-edge while the loop body reads the other accumulator (bisected on K; straight-line code is
                # accepted). So: chunks of `chunk` k-groups, steps inside a chunk unrolled in Python (straight-line),
                # wait<0> at each chunk end -> nothing in flight across the T.serial back-edge.
                for c in T.serial(n_k // chunk):
                    base = c * chunk
                    issue(A_s, B_s, bars, parts[0][w], base, w, S)  # chunk prologue: batch base -> part_a
                    for j in steps:  # Python tuple: unrolled at trace time
                        issue(A_s, B_s, bars, parts[(j + 1) % 2][w], base + j + 1, w, S)  # queue next (2 in flight)
                        T.wait_wgmma(1)  # batch base+j retired
                        promote(acc[w], parts[j % 2][w], scale_s, bars, base + j, w, S)  # overlaps batch base+j+1
                    T.wait_wgmma(0)  # chunk end: drain, so nothing crosses the back-edge
                    promote(acc[w], parts[(chunk - 1) % 2][w], scale_s, bars, base + chunk - 1, w, S)

                T.copy(acc[w], C[by * block_M + w * WG_M, bx * block_N])

    return C


def make(stages: int = 5, swizzle: int = 8, hoist: bool = True, chunk: int = 8):
    """Harness contract: fn(a, b, sa, sb) -> c bf16."""

    def fn(a, b, sa, sb):
        return fp8_gemm_ws_2part(a, b, sa, sb, stages=stages, swizzle=swizzle, hoist=hoist, chunk=chunk)

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
    configs = [(5, 8, 2), (5, 8, 4), (5, 8, 8), (6, 8, 4), (6, 8, 8)]  # (stages, swizzle, chunk)
    fns = {f"S={st} P={p} ch={ch}": make(st, p, chunk=ch) for st, p, ch in configs}
    import gemm_ws_tma_async as v0  # same-run reference
    fns = {"v0 S=4 P=8": v0.make(4, 8, True), **fns}

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
