"""Attention prefill forward, bf16. 1 producer + 2 consumer warpgroups, PING-PONG scheduled.

Same CTA layout as fa_ws_2consumer.py (one CTA = one 128-row Q block of one head; WG0 rows 0..63, WG1 rows 64..127,
WG2 = TMA producer). What changes is WHEN each consumer uses the tensor cores:

    WG0:  [QK0]  sm0        [QK1 PV0]  sm1        [QK2 PV1]  sm2  ...
    WG1:      wait  [QK0]  sm0        [QK1 PV0]  sm1        [QK2 PV1] ...

Two named barriers (bar.sync / bar.arrive, ids 1 and 2; id 0 is __syncthreads) hand a "turn" back and forth:
    T.sync_threads(me, 256)              wait for my turn: needs my 128 threads + the other WG's 128 arrivals
    ... issue wgmmas only ...
    T.named_barrier_arrive(other, 256)   give the turn away (non-blocking)
Everything else (softmax, O rescale) runs while the other warpgroup owns the tensor cores.

KV blocks are visited in REVERSE (kb = n_iter-1 .. 0), like FA3. Online softmax doesn't care about order, and this way
the only block that needs a mask (causal diagonal / ragged tail) is the FIRST one -> masked prologue, unmasked loop,
no peeled last iteration. It also means m is finite after block 0 for every row.

Per consumer, pipeline step t (stage s = t % S), block kb = n_iter-1-t:
    t = 0         : QK_0 only, masked
    t = 1..n-1    : one turn issues QK_t then PV_{t-1}.  wait_wgmma(1) -> S_t ready (groups retire in order) ->
                    softmax stats run while PV_{t-1} is still on the tensor cores.  wait_wgmma(0) -> O *= alpha, P_t.
    tail          : PV_{n-1}
"""

import math

import torch
import tilelang
import tilelang.language as T

PRECISION = "bf16"
LOG2E = 1.4426950408889634
WG_M = 64  # Q rows per consumer warpgroup (one m64 wgmma)
N_CONSUMERS = 2
CONSUMER_WGS = tuple(range(N_CONSUMERS))  # Python tuple: unrolled at trace time (T.ws needs a constant)
SCHED_BAR = (1, 2)  # named barrier per consumer: WG w waits on SCHED_BAR[w] for its turn


# Softmax statistics for one S tile (one warpgroup). `masked` is a trace-time bool: no runtime branch writes the
# accumulator (a runtime `if` around accumulator writes makes ptxas serialize every wgmma, C7515).
@T.macro
def softmax_stats(acc_s, m, m_prev, l, rowsum, alpha, kb, r0, N, masked, causal, BN, c):
    NEG_INF = -T.infinity(T.float32)
    k0 = kb * BN
    if masked:  # keys >= N (ragged tail) and, if causal, keys after the query row
        if causal:
            for i, jj in T.Parallel(WG_M, BN):
                acc_s[i, jj] = T.if_then_else(k0 + jj >= N or k0 + jj > r0 + i, NEG_INF, acc_s[i, jj])
        else:
            for i, jj in T.Parallel(WG_M, BN):
                acc_s[i, jj] = T.if_then_else(k0 + jj >= N, NEG_INF, acc_s[i, jj])
    T.copy(m, m_prev)
    T.reduce_max(acc_s, m, dim=1, clear=False)  # m = max(m_prev, rowmax S)
    for i in T.Parallel(WG_M):
        alpha[i] = T.exp2((m_prev[i] - m[i]) * c)  # m_prev = -inf on block 0 -> 0
    for i, jj in T.Parallel(WG_M, BN):
        acc_s[i, jj] = T.exp2(acc_s[i, jj] * c - m[i] * c)
    T.reduce_sum(acc_s, rowsum, dim=1)
    for i in T.Parallel(WG_M):
        l[i] = l[i] * alpha[i] + rowsum[i]


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def fa_fwd(Q, K, V, causal: bool = False, block_N: int = 128, stages: int = 2, head_dim: int = 128):
    BATCH, N, HQ, HKV = T.dynamic("BATCH, N, HQ, HKV")
    D = head_dim
    Q: T.Tensor((BATCH, N, HQ, D), T.bfloat16)
    K: T.Tensor((BATCH, N, HKV, D), T.bfloat16)
    V: T.Tensor((BATCH, N, HKV, D), T.bfloat16)
    O = T.empty((BATCH, N, HQ, D), T.bfloat16)
    LSE = T.empty((BATCH, HQ, N), T.float32)

    S = stages
    BM = N_CONSUMERS * WG_M  # 128
    BN = block_N
    assert BN == BM, "only the first (diagonal / tail) block is masked: needs BM == BN"
    G = HQ // HKV
    n_qb = T.ceildiv(N, BM)
    n_kb = T.ceildiv(N, BN)
    sm_scale = 1.0 / math.sqrt(D)
    c = sm_scale * LOG2E  # exp(x*scale) = exp2(x*c)
    NEG_INF = -T.infinity(T.float32)

    # barrier layout: [q_full | k_full[S] | v_full[S] | k_empty[S] | v_empty[S]]
    Q_FULL, K_FULL, V_FULL, K_EMPTY, V_EMPTY = 0, 1, 1 + S, 1 + 2 * S, 1 + 3 * S
    barrier_counts = [128] + [128] * S + [128] * S + [128 * N_CONSUMERS] * S + [128 * N_CONSUMERS] * S

    with T.Kernel(n_qb, HQ, BATCH, threads=128 * (N_CONSUMERS + 1)) as (bx, h, b):
        qb = (n_qb - 1 - bx) if causal else bx  # causal: longest Q blocks first
        q0 = qb * BM
        hk = h // G
        n_iter = T.min(T.ceildiv(q0 + BM, BN), n_kb) if causal else n_kb

        Q_s = T.alloc_shared((BM, D), T.bfloat16)
        K_s = T.alloc_shared((S, BN, D), T.bfloat16)
        V_s = T.alloc_shared((S, BN, D), T.bfloat16)
        bars = T.alloc_barrier(barrier_counts)

        acc_s = [T.alloc_fragment((WG_M, BN), T.float32) for _ in CONSUMER_WGS]
        acc_p = [T.alloc_fragment((WG_M, BN), T.bfloat16) for _ in CONSUMER_WGS]
        acc_o = [T.alloc_fragment((WG_M, D), T.float32) for _ in CONSUMER_WGS]
        m = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]
        m_prev = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]
        l = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]
        rowsum = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]
        alpha = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]

        # ---------------- producer warpgroup: K/V blocks in reverse order ----------------
        with T.ws(N_CONSUMERS):
            T.dec_max_nreg(24)
            T.tma_copy(Q[b, q0:q0 + BM, h, :], Q_s, barrier=bars[Q_FULL])
            T.mbarrier_arrive(bars[Q_FULL])
            for t in T.serial(n_iter):
                s = t % S
                ph = (t // S) & 1
                kb = n_iter - 1 - t
                T.mbarrier_wait_parity(bars[K_EMPTY + s], ph ^ 1)
                T.tma_copy(K[b, kb * BN:(kb + 1) * BN, hk, :], K_s[s, :, :], barrier=bars[K_FULL + s])
                T.mbarrier_arrive(bars[K_FULL + s])
                T.mbarrier_wait_parity(bars[V_EMPTY + s], ph ^ 1)
                T.tma_copy(V[b, kb * BN:(kb + 1) * BN, hk, :], V_s[s, :, :], barrier=bars[V_FULL + s])
                T.mbarrier_arrive(bars[V_FULL + s])

        # ---------------- consumers ----------------
        for w in CONSUMER_WGS:
            with T.ws(w):
                T.inc_max_nreg(240)
                me, other = SCHED_BAR[w], SCHED_BAR[1 - w]
                r0 = q0 + w * WG_M
                T.fill(m[w], NEG_INF)
                T.fill(l[w], 0)
                T.clear(acc_o[w])
                T.mbarrier_wait_parity(bars[Q_FULL], 0)
                if w == 1:
                    T.named_barrier_arrive(SCHED_BAR[0], 256)  # pre-arm: WG0 takes the first turn

                # ---- t = 0: diagonal / tail block. QK only, masked ----
                T.mbarrier_wait_parity(bars[K_FULL + 0], 0)
                T.sync_threads(me, 256)  # my turn on the tensor cores
                T.wgmma_gemm(Q_s[w * WG_M:(w + 1) * WG_M, :], K_s[0, :, :], acc_s[w], transpose_B=True, clear_accum=True)
                T.named_barrier_arrive(other, 256)  # your turn
                T.wait_wgmma(0)
                T.mbarrier_arrive(bars[K_EMPTY + 0])
                softmax_stats(acc_s[w], m[w], m_prev[w], l[w], rowsum[w], alpha[w],
                              n_iter - 1, r0, N, True, causal, BN, c)
                T.copy(acc_s[w], acc_p[w])  # P_0 (O is still 0: no rescale)

                # ---- t = 1 .. n_iter-1: QK of this block + PV of the previous one, in one turn ----
                for tt in T.serial(n_iter - 1):
                    t = tt + 1
                    s = t % S
                    sp = (t - 1) % S
                    T.mbarrier_wait_parity(bars[K_FULL + s], (t // S) & 1)
                    T.mbarrier_wait_parity(bars[V_FULL + sp], ((t - 1) // S) & 1)
                    T.sync_threads(me, 256)
                    T.wgmma_gemm(Q_s[w * WG_M:(w + 1) * WG_M, :], K_s[s, :, :], acc_s[w], transpose_B=True, clear_accum=True)  # S_t
                    T.wgmma_gemm(acc_p[w], V_s[sp, :, :], acc_o[w])  # O += P_{t-1} V_{t-1}
                    T.named_barrier_arrive(other, 256)

                    T.wait_wgmma(1)  # S_t ready; PV_{t-1} may still be running
                    T.mbarrier_arrive(bars[K_EMPTY + s])
                    softmax_stats(acc_s[w], m[w], m_prev[w], l[w], rowsum[w], alpha[w],
                                  n_iter - 1 - t, r0, N, False, causal, BN, c)

                    T.wait_wgmma(0)  # PV_{t-1} done: O and P are free
                    T.mbarrier_arrive(bars[V_EMPTY + sp])
                    for i, d in T.Parallel(WG_M, D):
                        acc_o[w][i, d] *= alpha[w][i]
                    T.copy(acc_s[w], acc_p[w])  # P_t

                # ---- tail: PV of the last block ----
                sp = (n_iter - 1) % S
                T.mbarrier_wait_parity(bars[V_FULL + sp], ((n_iter - 1) // S) & 1)
                T.sync_threads(me, 256)
                T.wgmma_gemm(acc_p[w], V_s[sp, :, :], acc_o[w])
                if w == 0:
                    T.named_barrier_arrive(other, 256)  # WG1's final arrive would have no matching sync
                T.wait_wgmma(0)
                T.mbarrier_arrive(bars[V_EMPTY + sp])

                # epilogue: O / l, LSE = m*scale + ln(l); rows >= N are not stored
                for i, d in T.Parallel(WG_M, D):
                    if r0 + i < N:
                        O[b, r0 + i, h, d] = acc_o[w][i, d] / l[w][i]
                for i in T.Parallel(WG_M):
                    if r0 + i < N:
                        LSE[b, h, r0 + i] = m[w][i] * sm_scale + T.log(l[w][i])

    return O, LSE


def make(block_N: int = 128, stages: int = 2):
    """Harness contract: fn(q, k, v, causal) -> (o, lse)."""

    def fn(q, k, v, causal):
        return fa_fwd(q, k, v, causal=causal, block_N=block_N, stages=stages, head_dim=q.shape[-1])

    return fn
