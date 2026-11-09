"""Hopper warp-specialized GEMM, PERSISTENT + COOPERATIVE: explicit loops/barriers; swizzle decode from tilelang.

    C[M, N] = A[M, K] @ B[N, K]^T        (bf16 in, fp32 accumulate, bf16 out)

Builds on gemm_ws_tma_2consumers.py (v6: 1 producer + 2 consumer warpgroups, 256x128 tile, async wgmma).

What "persistent" changes
-------------------------
v6 launches one CTA per output tile (32 x 64 = 2048 CTAs). Only one fits per SM (192 KB smem), so a new CTA starts only
after the previous one exits: every tile pays launch + pipeline fill, and the epilogue sits on the critical path.

v7 launches exactly NUM_SMS CTAs ("workers"). Worker `bid` walks a FIXED list of tiles, known up front, no communication:

    tile_id = w * NUM_SMS + bid        for w = 0, 1, 2, ...   while tile_id < num_tiles
    (tile_m, tile_n) = sched.coord(tile_id)      # tilelang's swizzle decode (pure arithmetic, no state)

At step w the 132 workers hold tile_ids [w*132, (w+1)*132): the same set a normal "wave" would run, so the swizzle's
L2 locality carries over. Which physical SM runs `bid` is up to the hardware and doesn't matter.

The win: the producer is a separate warpgroup, so while the consumers do the epilogue of tile t it is already loading
tile t+1 (throttled only by the empty barriers). When the consumers come back, the stages are full: no fill bubble.
The consumers' own epilogue still blocks their next MMA (program order on their registers); TMA store / ping-pong
shrink / hide that later.

Three things that MUST be right (each one is a deadlock or a race otherwise)
----------------------------------------------------------------------------
1. Producer and consumers walk the identical tile sequence with the identical guard. If one role does one more tile,
   barrier counts drift and everyone hangs.
2. Barrier stage + phase come from a GLOBAL step counter `it = w * n_k + ko` that keeps counting across tiles, not
   from `ko` (which restarts per tile). A barrier's phase only flips when it is used, so the counter must count uses.
   (The guard only skips tiles in the LAST step w, after which nothing runs, so `w * n_k` still counts real uses.)
3. The consumer frees stage it-1 inside the loop (async wgmma), so the LAST stage of each tile is freed after
   wait_wgmma(0). In v6 that arrive was missing and harmless (kernel exits); here the producer would wait on it forever.
"""

import os
import sys

import torch
import tilelang
import tilelang.language as T

WG_M = 128  # rows per consumer warpgroup
EPI_BAR = (1, 2)  # named-barrier ids for consumer 0 / 1's epilogue (id 0 is __syncthreads, don't touch it)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def gemm_persistent(A, B, block_N: int = 128, block_K: int = 64, swizzle: int = 6, num_sms: int = 132,
                    num_stages: int = 4, tma_store: bool = False):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)
    C = T.empty((M, N), T.bfloat16)

    # smem budget (227 KB max): stages 48 KB each; TMA-store staging = 2 x 128x128 bf16 = 64 KB.
    #   tma_store=False: 4 stages = 192 KB.   tma_store=True: 3 stages + 64 KB = 208 KB (4 + 64 = 256 doesn't fit).
    S = num_stages
    block_M = 2 * WG_M  # 256
    n_k = K // block_K  # k-steps per tile
    grid_m = M // block_M  # 32 tile-rows
    grid_n = N // block_N  # 64 tile-cols
    num_tiles = grid_m * grid_n  # 2048
    num_steps = (num_tiles + num_sms - 1) // num_sms  # tiles per worker, rounded up (16 here)
    barrier_counts = [128] * S + [256] * S  # full[s]: producer WG; empty[s]: both consumer WGs

    # Grid = one CTA per SM. No T.use_swizzle: blockIdx is a worker id now, so the swizzle moves into the tile decode.
    with T.Kernel(num_sms, threads=384) as bid:
        # Used ONLY as a decoder: coord(tile_id) -> (m, n). stateful=False = no internal loop/counters; our loops own that.
        # column_major=False: panels of `swizzle` tile-rows, swept column by column (like T.use_swizzle, minus serpentine).
        sched = T.PersistentTileScheduler(grid_m, grid_n, num_workers=num_sms, swizzle_size=swizzle,
                                          column_major=False, stateful=False)
        A_s = T.alloc_shared((S, block_M, block_K), T.bfloat16)
        B_s = T.alloc_shared((S, block_N, block_K), T.bfloat16)
        C0 = T.alloc_fragment((WG_M, block_N), T.float32)
        C1 = T.alloc_fragment((WG_M, block_N), T.float32)
        if tma_store:
            # Epilogue staging: regs -> smem -> (TMA) -> global. ONE buffer, each consumer owns a slice.
            # Two separate buffers get ALIASED by tilelang's smem planner (it sees the T.ws branches as sequential, so
            # it reuses one buffer's memory for the other) -> both consumers would write the same smem concurrently.
            C_s = T.alloc_shared((2, WG_M, block_N), T.bfloat16)  # [consumer, rows, cols], like A_s[stage]
        bars = T.alloc_barrier(barrier_counts)

        # ======================= producer warpgroup (WG2) =======================
        with T.ws(2):
            T.dec_max_nreg(40)
            tm = T.alloc_var(T.int32)  # this tile's (row, col): per-thread scalar registers
            tn = T.alloc_var(T.int32)
            for w in T.serial(num_steps):  # loop 1: over this worker's tiles
                tile_id = w * num_sms + bid
                if tile_id < num_tiles:  # same guard as the consumers
                    tm_d, tn_d = sched.coord(tile_id)
                    tm = tm_d  # alloc_var is a scalar: plain assignment = store into the register. Computed once
                    tn = tn_d  # per tile; without it the decode gets inlined into every TMA coord / epilogue store
                    T.assume(0 <= tm and tm < grid_m)  # lets the compiler drop per-store bounds checks in the epilogue
                    T.assume(0 <= tn and tn < grid_n)
                    for ko in T.serial(n_k):  # loop 2: over K
                        it = w * n_k + ko  # global step -> stage + phase keep counting across tiles
                        s = it % S
                        phase = (it // S) & 1
                        T.mbarrier_wait_parity(bars[S + s], phase ^ 1)  # slot free (both consumers arrived)?
                        T.tma_copy(A[tm * block_M:(tm + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                                   A_s[s, :, :], barrier=bars[s])
                        T.tma_copy(B[tn * block_N:(tn + 1) * block_N, ko * block_K:(ko + 1) * block_K],
                                   B_s[s, :, :], barrier=bars[s])
                        T.mbarrier_arrive(bars[s])

        # ======================= consumer 0 (WG0): tile rows 0..127 =======================
        with T.ws(0):
            T.inc_max_nreg(232)
            tm = T.alloc_var(T.int32)  # this tile's (row, col): per-thread scalar registers
            tn = T.alloc_var(T.int32)
            for w in T.serial(num_steps):
                tile_id = w * num_sms + bid
                if tile_id < num_tiles:
                    tm_d, tn_d = sched.coord(tile_id)
                    tm = tm_d  # alloc_var is a scalar: plain assignment = store into the register. Computed once
                    tn = tn_d  # per tile; without it the decode gets inlined into every TMA coord / epilogue store
                    T.assume(0 <= tm and tm < grid_m)  # lets the compiler drop per-store bounds checks in the epilogue
                    T.assume(0 <= tn and tn < grid_n)
                    T.clear(C0)  # previous tile's epilogue already read C0 (program order, same threads)
                    for ko in T.serial(n_k):
                        it = w * n_k + ko
                        s = it % S
                        phase = (it // S) & 1
                        T.mbarrier_wait_parity(bars[s], phase)
                        T.wgmma_gemm(A_s[s, 0:WG_M, :], B_s[s, :, :], C0, transpose_B=True)
                        T.wait_wgmma(1)  # batch it-1 done
                        if ko > 0:
                            T.mbarrier_arrive(bars[S + (it - 1) % S])  # free stage of step it-1
                    T.wait_wgmma(0)  # last batch of this tile done
                    T.mbarrier_arrive(bars[S + (w * n_k + n_k - 1) % S])  # free the tile's LAST stage (rule 3)
                    # Epilogue. While this runs, the producer is already loading tile w+1.
                    if tma_store:
                        # (a) my half of C_s may still be being READ by the TMA store of the previous tile. Bulk-store groups are
                        #     per-thread: only the thread that issued it has one in flight, everyone else returns at once.
                        T.tma_store_wait(0)
                        T.sync_threads(EPI_BAR[0], 128)  # ...then the whole warpgroup learns its half is free
                        # (b) regs -> smem: plain st.shared from all 128 threads ("generic proxy" writes)
                        T.copy(C0, C_s[0, :, :])
                        # (c) TMA reads smem through the "async proxy", which doesn't see generic writes by default:
                        #     each thread fences its own writes, then a barrier makes sure ALL 128 have fenced
                        T.fence_proxy_async()
                        T.sync_threads(EPI_BAR[0], 128)
                        # (d) one elected thread issues the bulk store + commit; nobody waits here -> straight to next tile
                        T.tma_copy(C_s[0, :, :], C[tm * block_M:tm * block_M + WG_M, tn * block_N:(tn + 1) * block_N])
                    else:
                        T.copy(C0, C[tm * block_M, tn * block_N])  # regs -> global, 4-byte st.global per element pair
            if tma_store:
                T.tma_store_wait(0)  # last tile's store must finish reading C_s before the CTA exits

        # ======================= consumer 1 (WG1): tile rows 128..255 =======================
        with T.ws(1):
            T.inc_max_nreg(232)
            tm = T.alloc_var(T.int32)  # this tile's (row, col): per-thread scalar registers
            tn = T.alloc_var(T.int32)
            for w in T.serial(num_steps):
                tile_id = w * num_sms + bid
                if tile_id < num_tiles:
                    tm_d, tn_d = sched.coord(tile_id)
                    tm = tm_d  # alloc_var is a scalar: plain assignment = store into the register. Computed once
                    tn = tn_d  # per tile; without it the decode gets inlined into every TMA coord / epilogue store
                    T.assume(0 <= tm and tm < grid_m)  # lets the compiler drop per-store bounds checks in the epilogue
                    T.assume(0 <= tn and tn < grid_n)
                    T.clear(C1)
                    for ko in T.serial(n_k):
                        it = w * n_k + ko
                        s = it % S
                        phase = (it // S) & 1
                        T.mbarrier_wait_parity(bars[s], phase)
                        T.wgmma_gemm(A_s[s, WG_M:2 * WG_M, :], B_s[s, :, :], C1, transpose_B=True)
                        T.wait_wgmma(1)
                        if ko > 0:
                            T.mbarrier_arrive(bars[S + (it - 1) % S])
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(bars[S + (w * n_k + n_k - 1) % S])
                    if tma_store:  # same 4 steps as consumer 0, own buffer + own named barrier
                        T.tma_store_wait(0)
                        T.sync_threads(EPI_BAR[1], 128)
                        T.copy(C1, C_s[1, :, :])
                        T.fence_proxy_async()
                        T.sync_threads(EPI_BAR[1], 128)
                        T.tma_copy(C_s[1, :, :], C[tm * block_M + WG_M:(tm + 1) * block_M, tn * block_N:(tn + 1) * block_N])
                    else:
                        T.copy(C1, C[tm * block_M + WG_M, tn * block_N])
            if tma_store:
                T.tma_store_wait(0)

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
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gemm_ws_tma_2consumers import gemm_coop  # v6, for an interleaved A/B

    M = N = K = 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    common = dict(M=M, N=N, K=K, swizzle=6, num_sms=num_sms)
    kernels = {"v6 coop": gemm_coop.compile(M=M, N=N, K=K, swizzle=6),
               "v7 persistent S=4, reg epi": gemm_persistent.compile(**common, num_stages=4),
               "v7 persistent S=3, reg epi": gemm_persistent.compile(**common, num_stages=3),  # isolates the stage cost
               "v7 persistent S=3, TMA epi": gemm_persistent.compile(**common, num_stages=3, tma_store=True)}

    if os.environ.get("PROFILE"):  # under ncu: one launch each, in dict order, then cuBLAS
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
    print(f"correctness: PASS ({', '.join(kernels)})")

    fns = {name: (lambda k=k: k(a, b)) for name, k in kernels.items()}
    fns["cuBLAS"] = lambda: a @ b.T
    flops = 2 * M * N * K
    for name, ms in bench_interleaved(fns).items():
        print(f"  {name:<32} {ms * 1e3:8.1f} us   {flops / ms / 1e9:6.1f} TFLOP/s")

    os.makedirs("/tmp/out", exist_ok=True)
    with open("/tmp/out/gemm_ws_persistent.generated.cu", "w") as f:
        f.write(kernels["v7 persistent S=3, TMA epi"].get_kernel_source())


if __name__ == "__main__":
    main()
