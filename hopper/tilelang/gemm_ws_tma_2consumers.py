"""Hopper warp-specialized GEMM, COOPERATIVE: 1 producer + 2 consumer warpgroups sharing a 256x128 tile.

    C[M, N] = A[M, K] @ B[N, K]^T        (bf16 in, fp32 accumulate, bf16 out)

Builds on gemm_ws_tma_async.py (TMA + mbarriers + swizzle + async wgmma). What changed:

    block = 384 threads = 3 warpgroups
        WG0  consumer: rows   0..127 of the tile -> its own 128x128 accumulator C0
        WG1  consumer: rows 128..255 of the tile -> its own 128x128 accumulator C1
        WG2  producer: TMA-loads A (256x64) + B (128x64) per stage = 48 KB; 4 stages = 192 KB of smem

Both consumers read the SAME B tile each stage and run the full K loop: no split-K, no reduction.
Why it helps: (1) 256x128 tile -> 85 FLOP per byte loaded into the SM instead of 64 -> 25% less L2->SM traffic;
(2) while one consumer does bookkeeping (barrier wait / arrive / wait_wgmma), the other's wgmmas keep the
tensor cores busy -> fewer bubbles.

Barriers: still ONE full[s] and ONE empty[s] per stage.
    full[s]   count 128: the producer's 128 arrivals + 48 KB of TMA bytes. BOTH consumers wait on it (waits only read).
    empty[s]  count 256: all 128 threads of WG0 + all 128 of WG1 arrive. The phase completes only when both
              consumers are done with slot s (they share B), so the producer's single wait = "both consumers done".

Consumers are WG0/WG1 (contiguous threads 0..255), producer WG2, so each T.ws block is one contiguous thread range.
"""

import os

import torch
import tilelang
import tilelang.language as T

NUM_STAGES = 4
WG_M = 128  # rows of the tile owned by each consumer warpgroup


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def gemm_coop(A, B, block_N: int = 128, block_K: int = 64, swizzle: int = 6):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)
    C = T.empty((M, N), T.bfloat16)

    S = NUM_STAGES
    block_M = 2 * WG_M  # 256: two consumer warpgroups stacked along M
    n_k = T.ceildiv(K, block_K)
    barrier_counts = [128] * S + [256] * S  # [0,S) = full[s] (producer WG), [S,2S) = empty[s] (both consumer WGs)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=384) as (bx, by):
        T.use_swizzle(panel_size=swizzle, enable=swizzle > 0)

        A_s = T.alloc_shared((S, block_M, block_K), T.bfloat16)  # 4 x 32 KB
        B_s = T.alloc_shared((S, block_N, block_K), T.bfloat16)  # 4 x 16 KB
        C0 = T.alloc_fragment((WG_M, block_N), T.float32)  # lives in WG0's registers (128 regs/thread)
        C1 = T.alloc_fragment((WG_M, block_N), T.float32)  # lives in WG1's registers
        bars = T.alloc_barrier(barrier_counts)

        # ---------------- producer warpgroup ----------------
        with T.ws(2):
            T.dec_max_nreg(40)  # producer only issues TMA: give registers back to the consumers
            for ko in T.serial(n_k):
                s = ko % S
                phase = (ko // S) & 1
                T.mbarrier_wait_parity(bars[S + s], phase ^ 1)  # slot s free? (= both consumers arrived)
                # One 256-row A box covers both consumers' halves (TMA box limit is 256 per dim).
                T.tma_copy(A[by * block_M:(by + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                           A_s[s, :, :], barrier=bars[s])
                T.tma_copy(B[bx * block_N:(bx + 1) * block_N, ko * block_K:(ko + 1) * block_K],
                           B_s[s, :, :], barrier=bars[s])
                T.mbarrier_arrive(bars[s])

        # ---------------- consumer 0: tile rows 0..127 ----------------
        with T.ws(0):
            T.inc_max_nreg(232)  # 128*40 + 256*232 = 64.5K <= 64K regs/SM
            T.clear(C0)
            for ko in T.serial(n_k):
                s = ko % S
                phase = (ko // S) & 1
                T.mbarrier_wait_parity(bars[s], phase)  # stage s landed?
                T.wgmma_gemm(A_s[s, 0:WG_M, :], B_s[s, :, :], C0, transpose_B=True)
                T.wait_wgmma(1)  # batch ko-1 done -> my reads of stage ko-1 are finished
                if ko > 0:
                    T.mbarrier_arrive(bars[S + (ko - 1) % S])  # my 128 arrivals of 256 for the previous stage
            T.wait_wgmma(0)
            T.copy(C0, C[by * block_M, bx * block_N])

        # ---------------- consumer 1: tile rows 128..255 ----------------
        with T.ws(1):
            T.inc_max_nreg(232)
            T.clear(C1)
            for ko in T.serial(n_k):
                s = ko % S
                phase = (ko // S) & 1
                T.mbarrier_wait_parity(bars[s], phase)
                T.wgmma_gemm(A_s[s, WG_M:2 * WG_M, :], B_s[s, :, :], C1, transpose_B=True)
                T.wait_wgmma(1)
                if ko > 0:
                    T.mbarrier_arrive(bars[S + (ko - 1) % S])  # the other 128 arrivals
            T.wait_wgmma(0)
            T.copy(C1, C[by * block_M + WG_M, bx * block_N])

    return C


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
    M = N = K = 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # Swizzle sweep: tile-rows are now 256 tall (A panel 4 MB, B panel 2 MB) -> model says optimum near P = sqrt(66) ~ 8.
    PANELS = [4, 6, 8, 12]
    kernels = {f"v6 coop P={p}": gemm_coop.compile(M=M, N=N, K=K, swizzle=p) for p in PANELS}

    if os.environ.get("PROFILE"):  # under ncu: one launch each, order = PANELS..., cuBLAS
        for k in kernels.values():
            k(a, b)
        a @ b.T
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile")
        for k in kernels.values():
            k(a, b)
        a @ b.T
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return

    ref = a @ b.T
    for name, k in kernels.items():
        torch.testing.assert_close(k(a, b).float(), ref.float(), rtol=2e-2, atol=2e-1)
    print(f"correctness: PASS (P = {PANELS})")

    fns = {name: (lambda k=k: k(a, b)) for name, k in kernels.items()}
    fns["cuBLAS"] = lambda: a @ b.T
    flops = 2 * M * N * K
    for name, ms in bench_interleaved(fns).items():
        print(f"  {name:<32} {ms * 1e3:8.1f} us   {flops / ms / 1e9:6.1f} TFLOP/s")

    os.makedirs("/tmp/out", exist_ok=True)
    with open("/tmp/out/gemm_ws_tma_2consumers.generated.cu", "w") as f:
        f.write(kernels["v6 coop P=8"].get_kernel_source())


if __name__ == "__main__":
    main()
