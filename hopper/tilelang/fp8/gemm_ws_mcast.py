"""FP8 block-scaled GEMM, v2: v1 (gemm_ws_tma_async_2part.py) + 2-CTA clusters with TMA multicast of A.

Why not a bigger tile: promotion keeps acc + part in fp32 registers, 2 * BM * BN * 4 B per CTA. 256x128 needs the
whole 256 KB register file even with ONE part buffer (4 WGs x m64: ~85 K regs > 64 K; 2 WGs x m128 or m64n256:
128 + 128 = 256 regs/thread > 232). A 2-CTA cluster sharing A gives the same L2 saving as a 128x256 tile at no
register cost:
    per CTA per k-group from L2:  16 KB A + 16 KB B = 32 KB  ->  8 KB A (half of one multicast) + 16 KB B = 24 KB
    total L2 -> SM:               8.6 GB  ->  6.4 GB  (0.75x)

Cluster protocol = bf16 v8.2 (../gemm_ws_cluster_multicast.py):
- 1-D grid of 4096 CTAs, cluster_dims=2: CTAs 2p, 2p+1 = pair p. Both decode the same (tm, tp); tn = 2*tp + rank.
  -> same A rows (multicast), adjacent B columns (own loads, own SB scalar).
- Tile order: pairs grouped in panels of `swizzle` tile-rows along M (our own decode; T.use_swizzle would remap
  blockIdx without knowing about clusters).
- full[s]  = LOCAL barrier, count 128. Each CTA expects its own 16 KB A + 16 KB B. With cluster_mask=0b11 rank 0
  issues ONE multicast of A into both CTAs' A_s[s]; its complete_tx lands on full[s] at the same smem offset in each.
  (A cluster-scope full would switch tilelang to a leader expect_tx protocol -> hang; see bf16 §15 gotcha.)
- empty[s] = CLUSTER barrier, count 256 local + 2 remote (one elected thread per PEER consumer WG). Rank 0 may only
  multicast into slot s when the peer's consumers released it too. Symmetric for simplicity.
- cluster_sync after barrier init (peer must not arrive on / multicast into uninitialized barriers) and before exit
  (don't exit while the peer may still arrive on my empty barriers).
- Scales: each CTA's producer still fills its own scale_s (same SA rows in both CTAs, different SB scalar).

Consumer: v1's chunked double-buffer (`chunk` k-groups straight-line, wait<0> at chunk end; ptxas C7514 otherwise).
chunk=1 = v0-style (one part buffer, wait<0> every k-group).
"""

import os

import torch
import tilelang
import tilelang.language as T

G = 128  # quantization group along K (A and B), block size along N (B)
WG_M = 64  # tile rows per consumer warpgroup (one m64 wgmma)
N_CONSUMERS = 2
CONSUMER_WGS = tuple(range(N_CONSUMERS))  # Python container: unrolled at trace time (range() would become T.serial)
CLUSTER = 2  # CTAs per cluster, paired along N (they share A)


@T.macro
def issue(A_s, B_s, full, part, ko, w, S):
    """Wait for stage ko % S to land, then queue its 4 x k32 wgmmas into `part` (overwrite: clear_accum)."""
    s = ko % S
    T.mbarrier_wait_parity(full[s], (ko // S) & 1)
    T.wgmma_gemm(A_s[s, w * WG_M:(w + 1) * WG_M, :], B_s[s, :, :], part, transpose_B=True, clear_accum=True)


@T.macro
def promote(acc, part, scale_s, empty, ko, w, S, tx, peer):
    """acc += part * scale for k-group ko, then release its stage locally (128) and to the peer CTA (1 per WG)."""
    s = ko % S
    for i, j in T.Parallel(WG_M, G):
        acc[i, j] += part[i, j] * scale_s[s, w * WG_M + i]
    T.mbarrier_arrive(empty[s])
    if tx % 128 == 0:
        T.mbarrier_arrive(empty[s], cta_id=peer)  # remote: mapa + mbarrier.arrive.shared::cluster


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def fp8_gemm_mcast(A, B, SA, SB, stages: int = 5, swizzle: int = 8, chunk: int = 8):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float8_e4m3fn)
    B: T.Tensor((N, K), T.float8_e4m3fn)
    SA: T.Tensor((M, K // G), T.float32)
    SB: T.Tensor((N // G, K // G), T.float32)
    C = T.empty((M, N), T.bfloat16)

    S = stages
    block_M = N_CONSUMERS * WG_M  # 128
    block_N = G  # 128
    block_K = G  # 128
    n_k = K // block_K
    grid_m = M // block_M  # 64 tile-rows
    grid_p = N // (CLUSTER * block_N)  # 32 pair-columns
    P = swizzle
    assert chunk >= 1 and n_k % chunk == 0, "K loop runs in chunks of `chunk` k-groups"
    assert P >= 1 and grid_m % P == 0, "panel size must divide the tile-rows"
    steps = tuple(range(chunk - 1))  # in-chunk steps (empty for chunk=1), unrolled at trace time

    with T.Kernel(grid_m * grid_p * CLUSTER, threads=128 * (N_CONSUMERS + 1), cluster_dims=CLUSTER) as bid:
        tx = T.get_thread_binding()
        rank = T.block_rank_in_cluster()  # %cluster_ctarank: what multicast masks / cta_id are relative to
        peer = 1 - rank

        # pair decode, materialized once (an inline expression would be re-emitted at every use)
        pair = T.alloc_var(T.int32)
        tm = T.alloc_var(T.int32)
        tn = T.alloc_var(T.int32)
        pair = T.truncdiv(bid, CLUSTER)
        r = T.truncmod(pair, P * grid_p)  # position inside the panel (P tile-rows x all pair-columns)
        tm = T.truncdiv(pair, P * grid_p) * P + T.truncmod(r, P)  # walk down M inside the panel...
        tn = T.truncdiv(r, P) * CLUSTER + rank  # ...then to the next pair-column; rank picks the half of the pair
        T.assume(0 <= tm and tm < grid_m)
        T.assume(0 <= tn and tn < N // block_N)

        A_s = T.alloc_shared((S, block_M, block_K), T.float8_e4m3fn)  # S x 16 KB (filled by rank 0's multicast)
        B_s = T.alloc_shared((S, block_N, block_K), T.float8_e4m3fn)  # S x 16 KB (own)
        scale_s = T.alloc_shared((S, block_M), T.float32)  # S x 512 B: SA[row, kb] * SB[tn, kb]
        part_a = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]
        part_b = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]
        acc = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]
        parts = [part_a, part_b]  # parts[k % 2][w]
        full = T.alloc_barrier([128] * S)  # LOCAL: per-CTA expect_tx (see docstring)
        empty = T.alloc_cluster_barrier([128 * N_CONSUMERS + N_CONSUMERS] * S)  # 256 local + 2 remote
        T.cluster_sync()  # peer must not arrive on my barriers / multicast into my smem before they're initialized

        # ---------------- producer warpgroup ----------------
        with T.ws(N_CONSUMERS):
            T.dec_max_nreg(40)
            row = tx - 128 * N_CONSUMERS
            sc = T.alloc_var(T.float32)
            for ko in T.serial(n_k):
                s = ko % S
                phase = (ko // S) & 1
                sc = SA[tm * block_M + row, ko] * SB[tn, ko]  # hoisted: overlaps the empty wait
                T.mbarrier_wait_parity(empty[s], phase ^ 1)  # slot s free in BOTH CTAs
                T.tma_copy(A[tm * block_M:(tm + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                           A_s[s, :, :], barrier=full[s], cluster_mask=0b11)  # rank 0 multicasts; rank 1 expects
                T.tma_copy(B[tn * block_N:(tn + 1) * block_N, ko * block_K:(ko + 1) * block_K],
                           B_s[s, :, :], barrier=full[s])
                scale_s[s, row] = sc
                T.mbarrier_arrive(full[s])

        # ---------------- consumers: WG w owns tile rows [w*64, (w+1)*64) ----------------
        for w in CONSUMER_WGS:
            with T.ws(w):
                T.inc_max_nreg(232)
                T.clear(acc[w])
                for c in T.serial(n_k // chunk):
                    base = c * chunk
                    issue(A_s, B_s, full, parts[0][w], base, w, S)
                    for j in steps:  # straight-line inside a chunk (ptxas C7514 otherwise)
                        issue(A_s, B_s, full, parts[(j + 1) % 2][w], base + j + 1, w, S)
                        T.wait_wgmma(1)
                        promote(acc[w], parts[j % 2][w], scale_s, empty, base + j, w, S, tx, peer)
                    T.wait_wgmma(0)  # chunk end: nothing in flight across the back-edge
                    promote(acc[w], parts[(chunk - 1) % 2][w], scale_s, empty, base + chunk - 1, w, S, tx, peer)

                T.copy(acc[w], C[tm * block_M + w * WG_M, tn * block_N])

        T.cluster_sync()  # don't exit while the peer may still arrive on my empty barriers

    return C


def make(stages: int = 5, swizzle: int = 8, chunk: int = 8):
    """Harness contract: fn(a, b, sa, sb) -> c bf16."""

    def fn(a, b, sa, sb):
        return fp8_gemm_mcast(a, b, sa, sb, stages=stages, swizzle=swizzle, chunk=chunk)

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
    import gemm_ws_tma_async_2part as v1  # same-run reference

    a, b, sa, sb = make_inputs(0, "scale_stress")
    fns = {"v1 S=5 P=8 ch=8": v1.make(5, 8, chunk=8)}
    for st, p, ch in [(5, 8, 1), (5, 8, 8), (6, 8, 1), (5, 4, 8), (5, 16, 8)]:  # (stages, swizzle, chunk)
        fns[f"mc S={st} P={p} ch={ch}"] = make(st, p, ch)

    if os.environ.get("PROFILE"):  # under ncu: one launch each, in order
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
        print(f"  {name:<18} max|err|/max|ref| = {rel:.2e}")

    flops = 2 * M * N * K
    for name, ms in bench_interleaved({k: (lambda f=f: f(a, b, sa, sb)) for k, f in fns.items()}).items():
        print(f"  {name:<18} {ms * 1e3:8.1f} us   {flops / ms / 1e9:7.1f} TFLOP/s")


if __name__ == "__main__":
    main()
