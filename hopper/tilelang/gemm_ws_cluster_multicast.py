"""Hopper warp-specialized GEMM, PERSISTENT + COOPERATIVE + 2-CTA CLUSTERS + TMA MULTICAST of A (v8.2, step 2).

    C[M, N] = A[M, K] @ B[N, K]^T        (bf16 in, fp32 accumulate, bf16 out)

Builds on gemm_ws_persistent.py (v7: 132 workers, 1 producer + 2 consumer warpgroups, 256x128 tile, 4 stages,
register epilogue).

Why clusters
------------
v7's ncu profile: L2 throughput ~80% of peak (cuBLAS: 58%), consumers wait on `full` ~33% of their samples while the
producer waits on `empty` ~87% -> the stage ring is full of loads still in flight. Goal: fewer L2->SM bytes per FLOP.
Two CTAs that work on tiles (tm, tn) and (tm, tn+1) need the SAME A rows. With TMA multicast, one L2 read of A can
land in both CTAs' smem: per CTA per k-step 32 KB from L2 instead of 48 KB (-33%).

Step 1 (THIS FILE): only the launch + tile decode change. No multicast, no cross-CTA barriers yet.
    - launch 132 CTAs as 66 clusters of 2 (`cluster_dims=2`): the two CTAs of a cluster run on SMs of the same GPC.
    - the scheduler runs at PAIR granularity: the 32 x 64 tile grid becomes 32 x 32 pairs of horizontally adjacent tiles.
          pair_id  = w * num_clusters + cluster_id
          (tm, tp) = sched.coord(pair_id)        # swizzled decode over the 32 x 32 pair grid
          tn       = 2 * tp + rank               # rank = CTA's position in its cluster (0 or 1)
      -> at every step w, both CTAs of a cluster have the same tm (same A rows) and adjacent tn (different B).
    - Expected: same speed as v7 (only the tile order changed). Correctness + speed check before adding multicast.

Why not the scheduler's own `cluster_size=2`: it clusters along M (tiles (tm, tn), (tm+1, tn)), i.e. the pair would
share B (128 rows), not A (256 rows). Sharing A saves more: 256*128/(128+128) = 128 FLOP/B vs 256*128/(256+64) = 102.

Step 2 (THIS FILE, on top of step 1 = gemm_ws_cluster.py): A is multicast, B stays per-CTA. Tile size unchanged.
    - `T.tma_copy(A..., cluster_mask=0b11)`: in tilelang 0.1.15 the LOWEST rank in the mask (rank 0) issues one
      multicast TMA of the whole 256x64 A slice into BOTH CTAs' A_s[s]; rank 1 issues nothing for A. Every CTA still
      does expect_tx(48 KB) on its own full[s]; the multicast completes tx on the barrier at the same smem offset in
      each destination CTA. (True half/half issue would need the raw tma_load_multicast intrinsic; L2 bytes are the
      same either way.)
    - empty[s] counts 256 local arrives + 2 remote ones (one elected thread per PEER consumer WG,
      T.mbarrier_arrive(bar, cta_id=peer)) = 258. Rank 0's producer may only multicast into slot s once rank 1's
      consumers released it too. Symmetric for simplicity (rank 1 strictly doesn't need to hear from rank 0).
    - T.cluster_sync() after barrier init (the peer must not arrive on my barriers / multicast into my smem before
      they're initialized) and before exit (I must not exit while the peer may still arrive on my barriers).
Requires cluster=True, pair_order=True (both CTAs of a cluster must walk the same tm / same steps).
"""

import os

import torch
import tilelang
import tilelang.language as T

WG_M = 128  # rows per consumer warpgroup
CLUSTER = 2  # CTAs per cluster, paired along N
TRACE_PRELUDE = r"""
__device__ __forceinline__ long long tl_gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ long long tl_smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return (long long)s; }
"""
# One-time grid-wide barrier: all CTAs of the (persistent, fully resident) grid meet before tile 0. ctr must be 0 at
# launch (host zeroes it). Deadlocks if any CTA is not resident: fine here (132 CTAs, 1/SM, 66/66 clusters active).
BARRIER_PRELUDE = r"""
__device__ __forceinline__ void tl_grid_barrier(int* ctr, int n) {
  if (threadIdx.x == 0) { atomicAdd(ctr, 1); while (atomicAdd(ctr, 0) < n) {} }
  __syncthreads();
}
"""


def make_prelude(trace, start_barrier):
    """Plain Python on purpose: inside the kernel body the eager builder rewrites `or`/`and` into TIR ops, and the
    prelude must reach codegen as a plain string (pragma_import_c needs a StringImm)."""
    text = (TRACE_PRELUDE if trace else "") + (BARRIER_PRELUDE if start_barrier else "")
    return text if text else None


def decode(w, pair_order, sched, bid, cid, rank, num_clusters, num_pairs, num_sms, num_tiles):
    """Tile of step w -> (valid, tm, tn). Plain Python building expressions (the eager builder rejects nested defs).
    Identical in every role and, for pair_order, in both CTAs of a cluster (same pair_id)."""
    if pair_order:
        pair_id = w * num_clusters + cid
        tm_d, tp_d = sched.coord(pair_id)
        return pair_id < num_pairs, tm_d, tp_d * CLUSTER + rank
    tile_id = w * num_sms + bid
    tm_d, tn_d = sched.coord(tile_id)
    return tile_id < num_tiles, tm_d, tn_d


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def gemm_mcast(A, B, G, block_N: int = 128, block_K: int = 64, swizzle: int = 6, num_sms: int = 132,
                 num_stages: int = 4, cluster: bool = True, pair_order: bool = True, trace: bool = False,
                 start_barrier: bool = False, cid_stride: int = 1):
    """cluster: launch as 2-CTA clusters. pair_order: tile decode pairs (tm, tn), (tm, tn+1) on CTAs 2c, 2c+1
    (False = v7's decode tile = w*num_sms + bid). trace: one thread of consumer 0 records, per tile,
    [smid, tm*grid_n+tn, t_start, t_mid(ko=n_k/2), t_end] (globaltimer ns) into Tr[bid, w, :].
    start_barrier: grid-wide barrier once before tile 0 (G = zeroed int32[1] counter; unused otherwise).
    cid_stride: pair_order only; cluster c works on slot (c * cid_stride) % num_clusters of each step (coprime to
    num_clusters -> a permutation). Same tiles per step, different tile -> cluster (-> GPC) assignment."""
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)
    G: T.Tensor((1,), T.int32)
    C = T.empty((M, N), T.bfloat16)
    # outputs must be declared before any other statement (the eager builder turns if/assert into frames).
    # Same step count for both decodes whenever num_sms is even: ceil(G/2 / (n/2)) == ceil(G / n).
    Tr = T.empty((num_sms, ((M // (2 * WG_M)) * (N // block_N) + num_sms - 1) // num_sms, 5), T.int64)

    S = num_stages
    block_M = 2 * WG_M  # 256
    n_k = K // block_K  # k-steps per tile
    grid_m = M // block_M  # 32 tile-rows
    grid_n = N // block_N  # 64 tile-cols
    assert grid_n % CLUSTER == 0 and num_sms % CLUSTER == 0
    assert cluster and pair_order, "multicast needs 2-CTA clusters walking the same tm"
    pair_n = grid_n // CLUSTER  # 32 pair-cols: pair p covers tile-cols 2p, 2p+1
    num_pairs = grid_m * pair_n  # 1024
    num_clusters = num_sms // CLUSTER  # 66 workers, each = one cluster of 2 CTAs
    num_steps = (num_pairs + num_clusters - 1) // num_clusters  # 16 (15.5 rounded up)
    num_tiles = grid_m * grid_n
    if not pair_order:
        num_steps = (num_tiles + num_sms - 1) // num_sms

    with T.Kernel(num_sms, threads=384, cluster_dims=CLUSTER if cluster else None,
                  prelude=make_prelude(trace, start_barrier)) as bid:
        tx = T.get_thread_binding()
        # 1-D clusters are contiguous blockIdx ranges: CTAs 2c, 2c+1 form cluster c. So cid = bid // 2 and
        # rank == bid % 2; block_rank_in_cluster() reads %cluster_ctarank (what multicast masks are relative to).
        cid = ((bid // CLUSTER) * cid_stride) % num_clusters if cid_stride != 1 else bid // CLUSTER
        rank = T.block_rank_in_cluster() if cluster else bid % CLUSTER
        # Decoder only (stateful=False). pair_order: same swizzle as v7, but over the pair grid (panels of `swizzle`
        # tile-rows, swept pair-column by pair-column). Otherwise exactly v7's decode.
        if pair_order:
            sched = T.PersistentTileScheduler(grid_m, pair_n, num_workers=num_clusters, swizzle_size=swizzle,
                                              column_major=False, stateful=False)
        else:
            sched = T.PersistentTileScheduler(grid_m, grid_n, num_workers=num_sms, swizzle_size=swizzle,
                                              column_major=False, stateful=False)

        dec = dict(pair_order=pair_order, sched=sched, bid=bid, cid=cid, rank=rank, num_clusters=num_clusters,
                   num_pairs=num_pairs, num_sms=num_sms, num_tiles=num_tiles)
        A_s = T.alloc_shared((S, block_M, block_K), T.bfloat16)
        B_s = T.alloc_shared((S, block_N, block_K), T.bfloat16)
        C0 = T.alloc_fragment((WG_M, block_N), T.float32)
        C1 = T.alloc_fragment((WG_M, block_N), T.float32)
        # full[s]: producer WG arrive (128) + expect_tx(48 KB). PLAIN (CTA-local) barrier on purpose: with a
        #   cluster-scope barrier, tilelang 0.1.15's tma_copy switches to a leader protocol (only rank 0 does
        #   expect_tx, of bytes * cluster_size) -> rank 1 would never wait for its data. Here each CTA expects its own
        #   32 KB A + 16 KB B; rank 0's multicast completes tx on full[s] at the same smem offset in both CTAs.
        # empty[s]: my 2 consumer WGs (256) + 1 elected thread per PEER consumer WG (2) = 258. Cluster scope: needed
        #   for the remote arrive (cta_id=peer -> mapa + mbarrier.arrive.shared::cluster).
        full = T.alloc_barrier([128] * S)
        empty = T.alloc_cluster_barrier([256 + 2] * S)
        peer = 1 - rank
        T.cluster_sync()  # peer must not arrive on my barriers / multicast into my smem before they're initialized
        if start_barrier:
            T.call_extern("void", "tl_grid_barrier", T.address_of(G[0]), num_sms)

        # ======================= producer warpgroup (WG2) =======================
        with T.ws(2):
            T.dec_max_nreg(40)
            tm = T.alloc_var(T.int32)
            tn = T.alloc_var(T.int32)
            for w in T.serial(num_steps):
                valid, tm_d, tn_d = decode(w, **dec)
                if valid:
                    tm = tm_d
                    tn = tn_d
                    T.assume(0 <= tm and tm < grid_m)
                    T.assume(0 <= tn and tn < grid_n)
                    for ko in T.serial(n_k):
                        it = w * n_k + ko
                        s = it % S
                        phase = (it // S) & 1
                        T.mbarrier_wait_parity(empty[s], phase ^ 1) # both CTAs have released the slot s
                        T.tma_copy(A[tm * block_M:(tm + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                                   A_s[s, :, :], barrier=full[s], cluster_mask=0b11) # rank 0 issues for both, rank 1 only expects the transaction to happen
                        T.tma_copy(B[tn * block_N:(tn + 1) * block_N, ko * block_K:(ko + 1) * block_K],
                                   B_s[s, :, :], barrier=full[s])
                        T.mbarrier_arrive(full[s])

        # ======================= consumer 0 (WG0): tile rows 0..127 =======================
        with T.ws(0):
            T.inc_max_nreg(232)
            tm = T.alloc_var(T.int32)
            tn = T.alloc_var(T.int32)
            for w in T.serial(num_steps):
                valid, tm_d, tn_d = decode(w, **dec)
                if valid:
                    tm = tm_d
                    tn = tn_d
                    T.assume(0 <= tm and tm < grid_m)
                    T.assume(0 <= tn and tn < grid_n)
                    if trace:
                        if tx == 0:
                            Tr[bid, w, 0] = T.call_extern("int64", "tl_smid")
                            Tr[bid, w, 1] = T.Cast("int64", tm * grid_n + tn)
                            Tr[bid, w, 2] = T.call_extern("int64", "tl_gtimer")
                    T.clear(C0)
                    for ko in T.serial(n_k):
                        it = w * n_k + ko
                        s = it % S
                        phase = (it // S) & 1
                        T.mbarrier_wait_parity(full[s], phase)
                        if trace:
                            if tx == 0 and ko == n_k // 2:
                                Tr[bid, w, 3] = T.call_extern("int64", "tl_gtimer")
                        T.wgmma_gemm(A_s[s, 0:WG_M, :], B_s[s, :, :], C0, transpose_B=True)
                        T.wait_wgmma(1)
                        if ko > 0:
                            T.mbarrier_arrive(empty[(it - 1) % S])
                            if tx % 128 == 0:
                                T.mbarrier_arrive(empty[(it - 1) % S], cta_id=peer)      # remote: 1 per WG
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(empty[(w * n_k + n_k - 1) % S])
                    if tx % 128 == 0:
                        T.mbarrier_arrive(empty[(w * n_k + n_k - 1) % S], cta_id=peer)  # last stage, remote
                    T.copy(C0, C[tm * block_M, tn * block_N])
                    if trace:
                        if tx == 0:
                            Tr[bid, w, 4] = T.call_extern("int64", "tl_gtimer")

        # ======================= consumer 1 (WG1): tile rows 128..255 =======================
        with T.ws(1):
            T.inc_max_nreg(232)
            tm = T.alloc_var(T.int32)
            tn = T.alloc_var(T.int32)
            for w in T.serial(num_steps):
                valid, tm_d, tn_d = decode(w, **dec)
                if valid:
                    tm = tm_d
                    tn = tn_d
                    T.assume(0 <= tm and tm < grid_m)
                    T.assume(0 <= tn and tn < grid_n)
                    T.clear(C1)
                    for ko in T.serial(n_k):
                        it = w * n_k + ko
                        s = it % S
                        phase = (it // S) & 1
                        T.mbarrier_wait_parity(full[s], phase)
                        T.wgmma_gemm(A_s[s, WG_M:2 * WG_M, :], B_s[s, :, :], C1, transpose_B=True)
                        T.wait_wgmma(1)
                        if ko > 0:
                            T.mbarrier_arrive(empty[(it - 1) % S])
                            if tx % 128 == 0:
                                T.mbarrier_arrive(empty[(it - 1) % S], cta_id=peer)      # remote: 1 per WG
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(empty[(w * n_k + n_k - 1) % S])
                    if tx % 128 == 0:
                        T.mbarrier_arrive(empty[(w * n_k + n_k - 1) % S], cta_id=peer)  # last stage, remote
                    T.copy(C1, C[tm * block_M + WG_M, tn * block_N])

        # all 384 threads: don't exit while the peer may still arrive on my empty barriers (its last tile's release)
        T.cluster_sync()

    return C, Tr


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


# v7 (no cluster) | v8.1 (cluster, no multicast: isolates the cluster cost) | v8.2 (this file: + multicast of A)


def summarize_trace(name, tr, num_sms, pair_order, grid_n):
    """tr: [num_sms, steps, 5] int64 = smid, tile, t0, tmid, tend (ns). Prints launch skew, drift growth, per-SM
    systematic slowness and sharer spreads, all in us."""
    import numpy as np
    steps = tr.shape[1]
    w = np.arange(steps)[None, :]
    b = np.arange(num_sms)[:, None]
    valid = (w * (num_sms // 2) + b // 2 < 1024) if pair_order else (w * num_sms + b < 2048)
    t0, tmid, tend = (tr[..., i].astype(np.float64) / 1e3 for i in (2, 3, 4))
    dur = np.where(valid, tend - t0, np.nan)
    smid = tr[:, 0, 0]
    tm, tn = tr[..., 1] // grid_n, tr[..., 1] % grid_n
    print(f"\n[trace] {name}")
    print(f"  launch skew (tile-0 start spread): {t0[:, 0].max() - t0[:, 0].min():.1f} us")
    print("  start-time spread per step w:   " + "  ".join(
        f"w{ww}:{t0[valid[:, ww], ww].max() - t0[valid[:, ww], ww].min():.0f}" for ww in (0, 2, 4, 8, 12, 15)))
    per_sm = np.nanmean(dur, axis=1)
    within = np.nanmean(np.nanstd(dur, axis=1))
    print(f"  tile us: median {np.nanmedian(dur):.1f}   per-SM mean: min {per_sm.min():.1f} max {per_sm.max():.1f} "
          f"std {per_sm.std():.2f}  (jitter within an SM: {within:.2f})")
    order = np.argsort(per_sm)
    print(f"  slowest 8 SMs (smid:us): {[(int(smid[k]), round(float(per_sm[k]), 1)) for k in order[-8:][::-1]]}")
    print(f"  fastest 8 SMs (smid:us): {[(int(smid[k]), round(float(per_sm[k]), 1)) for k in order[:8]]}")
    a_spread, b_spread = [], []
    for ww in range(steps):
        v = valid[:, ww]
        for key, out in ((tm[v, ww], a_spread), (tn[v, ww], b_spread)):
            for k in np.unique(key):
                g = tmid[v, ww][key == k]
                if len(g) > 1:
                    out.append(g.max() - g.min())
    end = np.nanmax(np.where(valid, tend, np.nan), axis=1)
    print(f"  mid-K spread among A-sharers {np.median(a_spread):.1f}  B-sharers {np.median(b_spread):.1f}   "
          f"worker finish spread {end.max() - end.min():.0f}")


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gemm_ws_persistent import gemm_persistent  # v7
    from gemm_ws_cluster import gemm_cluster  # v8.1: cluster launch + pair order, no multicast

    M = N = K = 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros(1, device="cuda", dtype=torch.int32)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    trace = bool(os.environ.get("TRACE"))  # off by default: bench numbers comparable to the v7 / v8.1 runs
    common = dict(M=M, N=N, K=K, num_sms=num_sms, num_stages=4, swizzle=6)
    v7 = gemm_persistent.compile(**common)
    v81 = gemm_cluster.compile(**common, cluster=True, pair_order=True, trace=trace)
    v82 = gemm_mcast.compile(**common, trace=trace)
    calls = {  # ncu table rows are in this order, then cuBLAS
        "v7 persistent": lambda: (v7(a, b), None),
        "v8.1 cluster, no mcast": lambda: v81(a, b, g),
        "v8.2 cluster + mcast A": lambda: v82(a, b, g),
    }

    if os.environ.get("PROFILE"):  # under ncu: one launch each, in dict order, then cuBLAS
        for f in calls.values():
            f()
        a @ b.T
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile")
        for f in calls.values():
            f()
        a @ b.T
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return

    ref = a @ b.T
    traces = {}
    for name, f in calls.items():
        c, tr = f()
        torch.cuda.synchronize()
        torch.testing.assert_close(c.float(), ref.float(), rtol=2e-2, atol=2e-1)
        if trace and tr is not None:
            traces[name] = tr.cpu().numpy()
    print(f"correctness: PASS ({', '.join(calls)})")
    print("ncu table rows below are in this order: " + " | ".join(calls) + " | cuBLAS")

    fns = dict(calls)
    fns["cuBLAS"] = lambda: a @ b.T
    flops = 2 * M * N * K
    for name, ms in bench_interleaved(fns).items():
        print(f"  {name:<32} {ms * 1e3:8.1f} us   {flops / ms / 1e9:6.1f} TFLOP/s")

    for name, tr in traces.items():
        summarize_trace(name, tr, num_sms, True, N // 128)


if __name__ == "__main__":
    main()
