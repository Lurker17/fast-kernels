"""Attention prefill forward, bf16. Hand-written warp specialization: 1 producer + 2 consumer warpgroups.

    block = 384 threads = 3 warpgroups, one CTA per (Q block of BM = 128 rows, query head h, batch b)
        WG0  consumer: Q rows  0..63  of the block -> the whole FlashAttention loop for those rows
        WG1  consumer: Q rows 64..127 of the block -> same, independently
        WG2  producer: TMA-loads the Q block once, then K_j and V_j blocks [BN, D] through an S-stage ring

Each consumer does the full thing on its own 64 rows (one m64 wgmma per GEMM), per KV block j:
    wait K_full[s]
    S  = Q_w . K_j^T                    wgmma SS (both operands from smem), fresh accumulator
    wait_wgmma(0)                       S is needed in registers now; this also retires PV_{j-1} (see below)
    arrive K_empty[s] ; arrive V_empty[s-1]
    mask (diagonal / ragged-tail blocks only)
    m_new = max(m, rowmax S)            alpha = exp2((m - m_new)*c)        c = scale * log2(e)
    P  = exp2(S*c - m_new*c)            l = alpha*l + rowsum P
    O *= alpha                          plain FP32 multiplies on the accumulator registers (no wgmma "scale")
    wait V_full[s]
    O += P . V_j                        wgmma RS: P (bf16) straight from registers, V_j from smem. NOT waited here.

wgmma is async and the hardware does NOT scoreboard its accumulator / A-operand registers: touching O or P while a
wgmma that uses them is in flight is a race. The only ordering is wgmma.wait_group. Here PV_j is left in flight
while the consumer waits for K_{j+1} and issues QK_{j+1}. The wait_wgmma(0) after QK_{j+1} retires both (groups
complete in order), and only after it do we rescale O (needs PV_j done) and overwrite P (PV_j reads it).

Barriers (all mbarriers; full = producer's 128 arrivals + TMA bytes, empty = both consumers' 256 arrivals):
    q_full           once
    k_full[s]  v_full[s]    separate, so QK^T can start before V lands
    k_empty[s] v_empty[s]   separate, so K is released right after QK^T, V only after PV

Causal: Q block qb (rows [qb*BM, qb*BM+BM)) only loops over KV blocks whose first key <= its last row. Blocks whose
last key > its first row get the element-wise mask (with BM = BN = 128: exactly the diagonal block). Heavy Q blocks
(high qb) are launched first. Ragged N: TMA zero-fills OOB rows; OOB keys are masked to -inf (a zero key scores 0,
not -inf) and OOB query rows are not stored.

Not yet: ping-pong between the two consumers (named barriers), intra-warpgroup QK_{j+1}/softmax_j overlap,
smem + TMA-store epilogue, persistent scheduling.
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


# ---------------- consumer: one KV block step (one warpgroup) ----------------
# `masked` is a trace-time bool, so the loop body never branches on the mask. A runtime `if` that writes the S
# accumulator makes ptxas serialize every wgmma in the kernel (C7515), so the masked block is peeled into its own call.
@T.macro
def kv_step(Q_w, K_s, V_s, bars, acc_s, acc_p, acc_o, m, m_prev, l, rowsum, alpha,
            j, r0, N, masked, causal, S, BN, D, c, bar_idx):
    K_FULL, V_FULL, K_EMPTY, V_EMPTY = bar_idx
    NEG_INF = -T.infinity(T.float32)
    s = j % S
    phase = (j // S) & 1
    k0 = j * BN

    # S = Q_w . K_j^T
    T.mbarrier_wait_parity(bars[K_FULL + s], phase)
    T.wgmma_gemm(Q_w, K_s[s, :, :], acc_s, transpose_B=True, clear_accum=True)
    T.wait_wgmma(0)  # S ready, and PV_{j-1} retired (O and P free to touch)
    T.mbarrier_arrive(bars[K_EMPTY + s])
    if j > 0:
        T.mbarrier_arrive(bars[V_EMPTY + (j - 1) % S])

    if masked:  # keys >= N (ragged tail) and, if causal, keys after the query row
        if causal:
            for i, jj in T.Parallel(WG_M, BN):
                acc_s[i, jj] = T.if_then_else(k0 + jj >= N or k0 + jj > r0 + i, NEG_INF, acc_s[i, jj])
        else:
            for i, jj in T.Parallel(WG_M, BN):
                acc_s[i, jj] = T.if_then_else(k0 + jj >= N, NEG_INF, acc_s[i, jj])

    # online softmax
    T.copy(m, m_prev)
    T.reduce_max(acc_s, m, dim=1, clear=False)  # m = max(m_prev, rowmax S)
    for i in T.Parallel(WG_M):
        alpha[i] = T.exp2((m_prev[i] - m[i]) * c)  # m_prev = -inf on block 0 -> 0
    for i, jj in T.Parallel(WG_M, BN):
        acc_s[i, jj] = T.exp2(acc_s[i, jj] * c - m[i] * c)
    T.reduce_sum(acc_s, rowsum, dim=1)
    for i in T.Parallel(WG_M):
        l[i] = l[i] * alpha[i] + rowsum[i]
    for i, d in T.Parallel(WG_M, D):
        acc_o[i, d] *= alpha[i]
    T.copy(acc_s, acc_p)  # fp32 -> bf16, stays in registers

    # O += P . V_j   (left in flight; retired by the next step's wait)
    T.mbarrier_wait_parity(bars[V_FULL + s], phase)
    T.wgmma_gemm(acc_p, V_s[s, :, :], acc_o)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def fa_fwd(Q, K, V, causal: bool = False, block_N: int = 128, stages: int = 2, head_dim: int = 128):
    BATCH, N, HQ, HKV = T.dynamic("BATCH, N, HQ, HKV")  # one compile per (causal, block_N, stages)
    D = head_dim  # Python int: the softmax scale below is a literal, not a per-element rsqrtf
    Q: T.Tensor((BATCH, N, HQ, D), T.bfloat16)
    K: T.Tensor((BATCH, N, HKV, D), T.bfloat16)
    V: T.Tensor((BATCH, N, HKV, D), T.bfloat16)
    O = T.empty((BATCH, N, HQ, D), T.bfloat16)
    LSE = T.empty((BATCH, HQ, N), T.float32)

    S = stages
    BM = N_CONSUMERS * WG_M  # 128
    BN = block_N
    assert BN == BM, "the peeled-mask structure assumes only the last KV block needs a mask (BM == BN)"
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
        # KV blocks this Q block needs: causal stops at the block holding key (q0 + BM - 1)
        n_iter = T.min(T.ceildiv(q0 + BM, BN), n_kb) if causal else n_kb

        Q_s = T.alloc_shared((BM, D), T.bfloat16)  # 32 KB, resident for the whole CTA
        K_s = T.alloc_shared((S, BN, D), T.bfloat16)  # S x 32 KB
        V_s = T.alloc_shared((S, BN, D), T.bfloat16)  # S x 32 KB
        bars = T.alloc_barrier(barrier_counts)

        # per-consumer register state (each list entry lives in one warpgroup's registers)
        acc_s = [T.alloc_fragment((WG_M, BN), T.float32) for _ in CONSUMER_WGS]  # S tile
        acc_p = [T.alloc_fragment((WG_M, BN), T.bfloat16) for _ in CONSUMER_WGS]  # P tile = wgmma A operand
        acc_o = [T.alloc_fragment((WG_M, D), T.float32) for _ in CONSUMER_WGS]  # O accumulator
        m = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]  # running max (raw score units)
        m_prev = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]
        l = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]  # running sum
        rowsum = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]
        alpha = [T.alloc_fragment((WG_M,), T.float32) for _ in CONSUMER_WGS]

        # ---------------- producer warpgroup ----------------
        with T.ws(N_CONSUMERS):
            T.dec_max_nreg(24)
            T.tma_copy(Q[b, q0:q0 + BM, h, :], Q_s, barrier=bars[Q_FULL])
            T.mbarrier_arrive(bars[Q_FULL])
            for j in T.serial(n_iter):
                s = j % S
                phase = (j // S) & 1
                T.mbarrier_wait_parity(bars[K_EMPTY + s], phase ^ 1)
                T.tma_copy(K[b, j * BN:(j + 1) * BN, hk, :], K_s[s, :, :], barrier=bars[K_FULL + s])
                T.mbarrier_arrive(bars[K_FULL + s])
                T.mbarrier_wait_parity(bars[V_EMPTY + s], phase ^ 1)
                T.tma_copy(V[b, j * BN:(j + 1) * BN, hk, :], V_s[s, :, :], barrier=bars[V_FULL + s])
                T.mbarrier_arrive(bars[V_FULL + s])

        # ---------------- consumers: WG w owns Q rows [q0 + w*64, q0 + (w+1)*64) ----------------
        # Only the LAST block ever needs a mask (BM == BN): causal -> it's the diagonal block; non-causal -> it's
        # the ragged tail (and for N % BN == 0 the mask is a no-op on it).
        for w in CONSUMER_WGS:
            with T.ws(w):
                T.inc_max_nreg(240)
                r0 = q0 + w * WG_M
                T.fill(m[w], NEG_INF)
                T.fill(l[w], 0)
                T.clear(acc_o[w])
                T.mbarrier_wait_parity(bars[Q_FULL], 0)

                args = (Q_s[w * WG_M:(w + 1) * WG_M, :], K_s, V_s, bars, acc_s[w], acc_p[w], acc_o[w],
                        m[w], m_prev[w], l[w], rowsum[w], alpha[w])
                for j in T.serial(n_iter - 1):
                    kv_step(*args, j, r0, N, False, causal, S, BN, D, c, (K_FULL, V_FULL, K_EMPTY, V_EMPTY))
                kv_step(*args, n_iter - 1, r0, N, True, causal, S, BN, D, c, (K_FULL, V_FULL, K_EMPTY, V_EMPTY))

                T.wait_wgmma(0)
                T.mbarrier_arrive(bars[V_EMPTY + (n_iter - 1) % S])  # keeps the barrier phases consistent

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
        o, lse = fa_fwd(q, k, v, causal=causal, block_N=block_N, stages=stages, head_dim=q.shape[-1])
        return o, lse

    return fn


if __name__ == "__main__":  # quick local check: python fa_ws_2consumer.py (on a GPU box)
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import harness

    print(harness.run(sys.modules[__name__]))
