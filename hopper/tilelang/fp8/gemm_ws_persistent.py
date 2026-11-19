"""FP8 block-scaled GEMM, v3: PERSISTENT version of v1/v2 (chunked double-buffer consumer, optional 2-CTA multicast).

Why persistent: v2 (gemm_ws_mcast.py, non-persistent) launched 2048 clusters of 2 one after another. A cluster can
only start when two SMs of the same GPC are free at once, so with 1 CTA/SM SMs idle waiting for a partner; v2 lost
~13% to v1. Here 132 CTAs (66 clusters) launch ONCE and loop over tiles, like bf16 v7/v8.2.

    block = 384 threads (WG0, WG1 consumers m64 each, WG2 producer), tile 128 x 128 x 128, S stages
    step t (0 .. num_steps-1): worker -> unit  t * num_workers + worker   (unit = tile, or pair of tiles if clustered)
    the k-stage ring runs ACROSS tiles: global k index  it = t * n_k + k  -> stage it % S, phase (it // S) & 1
    -> the producer loads tile t+1 while the consumers promote/store tile t (epilogue overlap for free).

mode (one knob, so one run gives the ladder):
    "none"    persistent only. 132 workers, PersistentTileScheduler over the 64 x 64 tile grid.
    "cluster" 2-CTA clusters (66 workers), scheduler over the 64 x 32 PAIR grid, tn = 2*tp + rank. No multicast.
    "mcast"   "cluster" + rank 0 multicasts A into both CTAs (cluster_mask=0b11); empty is a cluster barrier with
              256 local + 2 remote arrivals (one elected thread per peer consumer WG). Protocol = bf16 v8.2 (§15).
Tile order comes from tilelang's PersistentTileScheduler (swizzle_size = panel of tile-rows), at pair granularity
when clustered, so both CTAs of a cluster always share tm (same A rows).

Consumer = v1's chunked double buffer: `chunk` k-groups straight-line, wait<0> at chunk end (ptxas C7514 serializes
every wgmma if a batch is in flight across a loop back-edge while the body reads the other accumulator).

Knobs added after v3 (make() defaults = current best: S=4, chunk=4, P=8, pf=4, tma_epi=True):
    tma_epi  epilogue acc -> bf16 C_s -> TMA store; the WG waits only before reusing C_s next tile (+9%)
    inc      carried (stage, phase) instead of it % S / it // S: correct but slower (division was never the cost)
    serp     boustrophedon panel order: noise
    early    release the stage right after wait_wgmma(1), promote from registers: no gain
    dbg      also return D[sm, it, 3] = clock64 stamps [TMA issue, consumer starts waiting, full fires]
             (probe_latency.py): the stall is a per-SM delivery-rate limit, extra stages only queue
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
MODES = ("none", "cluster", "mcast")
RS = 2  # scale-ring slots (ring mode): one block of `ring` k-groups per slot


def decode(t, clustered, sched, bid, cid, rank, num_workers, num_units, serp, sched_n, swizzle):
    """Step t -> (valid, tm, tn). Plain Python building expressions (the eager builder rejects nested defs).
    Same result in every role and, when clustered, in both CTAs of a cluster (same pair index).
    serp: boustrophedon. tilelang's coord() sweeps every panel (swizzle tile-rows x all columns) left -> right;
    serp reverses the column sweep on odd panels so a panel starts on the B columns the previous one just used."""
    unit = t * num_workers + (cid if clustered else bid)
    tm_d, tc_d = sched.coord(unit)
    if serp:
        tc_d = T.if_then_else((unit // (sched_n * swizzle)) % 2 == 1, sched_n - 1 - tc_d, tc_d)
    if clustered:
        return unit < num_units, tm_d, tc_d * CLUSTER + rank
    return unit < num_units, tm_d, tc_d


@T.macro
def issue(A_s, B_s, full, part, s, ph, w, D, dbg, bid, it, tx):
    """Wait for stage s (phase ph) to land, then queue its 4 x k32 wgmmas into `part` (overwrite: clear_accum).
    dbg: WG0's thread 0 stamps clock64 before (t_need) and after (t_ready) the full wait into D[bid, it, 1 / 2]."""
    if dbg and w == 0:
        if tx == 0:
            D[bid, it, 1] = T.call_extern("int64", "clock64")
    T.mbarrier_wait_parity(full[s], ph)
    if dbg and w == 0:
        if tx == 0:
            D[bid, it, 2] = T.call_extern("int64", "clock64")
    T.wgmma_gemm(A_s[s, w * WG_M:(w + 1) * WG_M, :], B_s[s, :, :], part, transpose_B=True, clear_accum=True)


ARRIVE_MODES = {"all": 0, "warp": 1, "wg": 2}


def stage_of(base, j, sv, pv, S, inc):
    """(stage, phase) of global k index base + j. Plain Python building expressions (like decode).
    inc=False: it % S, (it // S) & 1  -> div/mod per step (free when S | chunk and S is a power of 2: folds to constants).
    inc=True : read from sv/pv, filled per block by step_stages from the carried (s0, p0): no division anywhere."""
    if inc:  # min/max: 2 IMNMX that tell the simplifier 0 <= s < S, else the smem address keeps a generic
        return T.max(T.min(sv[j], S - 1), 0), pv[j]  # (s % 640) / 5 ... form (seen in the lowered CUDA)
    it = base + j
    return it % S, (it // S) & 1


@T.macro
def step_stages(sv, pv, s0, p0, S, js):
    """inc: stages/phases of the block's steps j in js from the block start (s0, p0). len(js) <= S -> at most one wrap,
    so each step is add + compare + 2 selects. Phase flips exactly when the stage index wraps."""
    for j in js:
        sv[j] = T.if_then_else(s0 + j >= S, s0 + j - S, s0 + j)
        pv[j] = T.if_then_else(s0 + j >= S, p0 ^ 1, p0)

# Advancing the carried (s0, p0) by a block is written INLINE in the kernel, not as a macro: macro args are passed by
# value (lowered `int s0_3 = s0;`), so `s0 = ...` inside a macro never reaches the caller's var (first try: the
# stage stayed 0 -> ring desync -> Xid 13). `it` is consecutive across chunks AND tiles (the ring spans tiles), so
# starting at (0, 0) and adding the block size each block replaces it % S / (it // S) & 1 entirely.


@T.macro
def group_arrive(bars, idx, tx, amode, bar_id):
    """One 128-thread warpgroup arrives on bars[idx] (array + index: a macro arg `bars[i]` would arrive as a Var). Each arrive is a RELEASE of the arriving thread's earlier smem
    accesses (producer: its scale_s store; consumer: its scale_s reads), so a lone arrive needs the group synced first.
      0 all : every thread arrives (count 128 per WG)
      1 warp: __syncwarp (orders the warp's accesses), lane 0 arrives -> cumulative release (count 4 per WG)
      2 wg  : named barrier bar.sync(bar_id, 128) over the WG, one thread arrives (count 1 per WG)"""
    if amode == 0:
        T.mbarrier_arrive(bars[idx])
    elif amode == 1:
        T.sync_warp()
        if tx % 32 == 0:
            T.mbarrier_arrive(bars[idx])
    else:
        T.sync_threads(bar_id, 128)
        if tx % 128 == 0:
            T.mbarrier_arrive(bars[idx])


@T.macro
def load_ab(A, B, A_s, B_s, full, empty, tm, tn, ko, s, ph, mcast):
    """Producer: wait for slot s to be free (phase ph ^ 1), then TMA A and B for k-group ko into it."""
    T.mbarrier_wait_parity(empty[s], ph ^ 1)
    if mcast:  # rank 0 multicasts into both CTAs; rank 1 only expects the bytes
        T.tma_copy(A[tm * 128:(tm + 1) * 128, ko * G:(ko + 1) * G], A_s[s, :, :], barrier=full[s], cluster_mask=0b11)
    else:
        T.tma_copy(A[tm * 128:(tm + 1) * 128, ko * G:(ko + 1) * G], A_s[s, :, :], barrier=full[s])
    T.tma_copy(B[tn * G:(tn + 1) * G, ko * G:(ko + 1) * G], B_s[s, :, :], barrier=full[s])


@T.macro
def promote(acc, part, scale_s, sa_s, sb_s, sa_r, sb_r, empty, it, s, w, tx, peer, remote, scale_tma, ring, amode,
            scf, early):
    """acc += part * scale for global k index it, then release its stage (locally; + 1 remote arrive per WG).
    scale_tma: scales come as TMA boxes SA[rows, 4*(k//4) : +4] / SB[tn, same] -> column k % 4 (= it % 4, since
    n_k % 4 == 0); the SA*SB product is formed here (2 rows per thread). Else: producer-folded scale_s[s, row].
    s = it % S, computed by the caller (div/mod or incremental, see stage_of).
    early (scale_s path): the caller's wait_wgmma already proved the wgmma that read stage s is done, so the only thing
    still tying the promote to the stage is scale_s[s]. Copy my rows' scales to registers (scf), release the stage,
    THEN do the FFMAs: the slot is held for one ld.shared instead of the whole promote."""
    if early and ring == 0 and not scale_tma:
        for i in T.Parallel(WG_M):
            scf[i] = scale_s[s, w * WG_M + i]
        group_arrive(empty, s, tx, amode, 1 + w)
        if remote:
            if tx % 128 == 0:
                T.mbarrier_arrive(empty[s], cta_id=peer)
        for i, j in T.Parallel(WG_M, G):
            acc[i, j] += part[i, j] * scf[i]
    elif ring > 0:  # TMA scale ring: block it // ring in slot (it // ring) % RS, column it % ring
        rslot = (it // ring) % RS
        rc = it % ring
        for i, j in T.Parallel(WG_M, G):
            acc[i, j] += part[i, j] * (sa_r[rslot, w * WG_M + i, rc] * sb_r[rslot, 0, rc])
    elif scale_tma:
        c4 = it % 4
        for i, j in T.Parallel(WG_M, G):
            acc[i, j] += part[i, j] * (sa_s[s, w * WG_M + i, c4] * sb_s[s, 0, c4])
    else:
        for i, j in T.Parallel(WG_M, G):
            acc[i, j] += part[i, j] * scale_s[s, w * WG_M + i]
    if not (early and ring == 0 and not scale_tma):
        group_arrive(empty, s, tx, amode, 1 + w)  # named barrier ids 1, 2 = consumer WGs (0 = __syncthreads)
        if remote:
            if tx % 128 == 0:
                T.mbarrier_arrive(empty[s], cta_id=peer)  # mapa + mbarrier.arrive.shared::cluster


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def fp8_gemm_persistent(A, B, SA, SB, stages: int = 5, swizzle: int = 8, chunk: int = 4, mode: str = "mcast",
                        num_sms: int = 132, prod_regs: int = 40, cons_regs: int = 232,
                        scale_tma: bool = True, pf: int = 0, ring: int = 0, arrive: str = "all", inc: bool = False,
                        serp: bool = False, early: bool = False, tma_epi: bool = False, dbg: bool = False):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float8_e4m3fn)
    B: T.Tensor((N, K), T.float8_e4m3fn)
    SA: T.Tensor((M, K // G), T.float32)
    SB: T.Tensor((N // G, K // G), T.float32)
    C = T.empty((M, N), T.bfloat16)
    # dbg: per-SM clock64 stamps for every global k index it = t * n_k + k: [t_issue, t_need, t_ready]
    D = T.empty((num_sms, ((M // 128) * (N // 128) + num_sms - 1) // num_sms * (K // 128), 3), T.int64) if dbg else None

    assert mode in MODES
    assert 128 * prod_regs + 256 * cons_regs <= 65536, "setmaxnreg budget: 64K regs/SM"
    clustered = mode != "none"
    mcast = mode == "mcast"
    S = stages
    block_M = N_CONSUMERS * WG_M  # 128
    block_N = G  # 128
    block_K = G  # 128
    n_k = K // block_K
    grid_m = M // block_M  # 64
    grid_n = N // block_N  # 64
    assert chunk >= 1 and n_k % chunk == 0, "K loop runs in chunks of `chunk` k-groups"
    assert n_k % 4 == 0, "scale TMA boxes are 4 k-groups wide (aligned)"
    assert pf == 0 or (not scale_tma and n_k % pf == 0), "pf: register prefetch of the ldg scale path"
    assert not dbg or pf > 0, "dbg stamps are only in the pf producer path"
    assert ring == 0 or (not scale_tma and pf == 0 and n_k % ring == 0 and ring % chunk == 0 and ring * 4 >= 16), \
        "ring: TMA scale ring, block of `ring` k-groups (inner box >= 16 B), chunks must not straddle blocks"
    # ring=4 ran but gave WRONG results (max err 1.2e-1 vs 2.8e-3; logs/20261117-fp8-persist-ring-quick.log); ring=8/16
    # are correct. Unexplained (suspects: smem aliasing/alignment of sa_r when shaped like sa_s, or chunk == block).
    assert ring == 0 or ring >= 8, "ring=4 is known-broken (see comment)"
    amode = ARRIVE_MODES[arrive]
    per_wg = (128, 4, 1)[amode]  # arrivals per warpgroup per phase
    pf_steps = tuple(range(pf))  # in-block producer steps, unrolled at trace time (static register indices)
    assert num_sms % CLUSTER == 0 and grid_n % CLUSTER == 0
    steps = tuple(range(chunk - 1))  # in-chunk steps (empty for chunk=1), unrolled at trace time
    all_steps = tuple(range(chunk))
    assert not inc or (chunk <= S and (pf == 0 or pf <= S) and ring == 0 and (pf > 0 or scale_tma)), \
        "inc: one wrap per block (chunk, pf <= S); implemented for the pf / scale_tma producer paths only"
    if clustered:
        num_workers = num_sms // CLUSTER  # 66 clusters
        num_units = grid_m * (grid_n // CLUSTER)  # 2048 pairs
        sched_n = grid_n // CLUSTER
    else:
        num_workers = num_sms  # 132 CTAs
        num_units = grid_m * grid_n  # 4096 tiles
        sched_n = grid_n
    num_steps = (num_units + num_workers - 1) // num_workers  # 32 (31.03 rounded up)

    with T.Kernel(num_sms, threads=128 * (N_CONSUMERS + 1), cluster_dims=CLUSTER if clustered else None) as bid:
        tx = T.get_thread_binding()
        cid = bid // CLUSTER  # 1-D clusters are contiguous blockIdx ranges: CTAs 2c, 2c+1 = cluster c
        rank = T.block_rank_in_cluster() if clustered else 0
        peer = 1 - rank
        sched = T.PersistentTileScheduler(grid_m, sched_n, num_workers=num_workers, swizzle_size=swizzle,
                                          column_major=False, stateful=False)  # decoder only
        dec = dict(clustered=clustered, sched=sched, bid=bid, cid=cid, rank=rank, num_workers=num_workers,
                   num_units=num_units, serp=serp, sched_n=sched_n, swizzle=swizzle)

        A_s = T.alloc_shared((S, block_M, block_K), T.float8_e4m3fn)  # S x 16 KB
        B_s = T.alloc_shared((S, block_N, block_K), T.float8_e4m3fn)  # S x 16 KB
        if tma_epi:  # bf16 staging for the TMA-store epilogue: 2 x 16 KB, one half per consumer WG
            C_s = T.alloc_shared((N_CONSUMERS, WG_M, block_N), T.bfloat16)
        scf = [T.alloc_fragment((WG_M,), T.float32) for _ in range(N_CONSUMERS)]  # early: my rows' scales
        scale_s = T.alloc_shared((S, block_M), T.float32)  # scale_tma=False: S x 512 B, SA[row, kb] * SB[tn, kb]
        # scale_tma=True: TMA boxes, 4 k-groups wide (TMA's inner box must be >= 16 B = 4 floats; 1 column is 4 B)
        sa_s = T.alloc_shared((S, block_M, 4), T.float32)  # S x 2 KB: SA[tm rows, 4*(ko//4) : +4]
        sb_s = T.alloc_shared((S, 1, 4), T.float32)  # S x 16 B:  SB[tn, 4*(ko//4) : +4]
        rr = ring if ring > 0 else 4
        sa_r = T.alloc_shared((RS, block_M, rr), T.float32)  # ring: RS x (128 x ring) floats, SA[tm rows, block]
        sb_r = T.alloc_shared((RS, 1, rr), T.float32)  # ring: RS x ring floats, SB[tn, block]
        sfull = T.alloc_barrier([128] * RS)  # ring: producer WG arrivals + TMA bytes
        sempty = T.alloc_barrier([128 * N_CONSUMERS] * RS)  # ring: both consumer WGs, once per block
        part_a = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]
        part_b = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]
        acc = [T.alloc_fragment((WG_M, block_N), T.float32) for _ in range(N_CONSUMERS)]
        parts = [part_a, part_b]  # parts[k % 2][w]
        full = T.alloc_barrier([per_wg] * S)  # LOCAL even with multicast: per-CTA expect_tx (bf16 §15 gotcha)
        if mcast:
            empty = T.alloc_cluster_barrier([per_wg * N_CONSUMERS + N_CONSUMERS] * S)  # local + 2 remote
        else:
            empty = T.alloc_barrier([per_wg * N_CONSUMERS] * S)
        if clustered:
            T.cluster_sync()  # peer must not arrive on my barriers / multicast into my smem before they're init

        # ---------------- producer warpgroup ----------------
        with T.ws(N_CONSUMERS):
            T.dec_max_nreg(prod_regs)
            row = tx - 128 * N_CONSUMERS
            tm = T.alloc_var(T.int32)
            tn = T.alloc_var(T.int32)
            sc = T.alloc_var(T.float32)
            psv = T.alloc_local((max(pf, 1),), T.int32)  # inc: this block's stages / phases (unused otherwise)
            ppv = T.alloc_local((max(pf, 1),), T.int32)
            if inc:
                s0 = T.alloc_var(T.int32)
                p0 = T.alloc_var(T.int32)
                s0 = 0
                p0 = 0
            if pf > 0:
                # register prefetch: this block's pf scales in cur_*, the next block's loads in flight in nxt_*.
                # A load doesn't block; its first USE does. Issued one block (pf k-groups) before use -> landed.
                cur_sa = T.alloc_local((pf,), T.float32)
                cur_sb = T.alloc_local((pf,), T.float32)
                nxt_sa = T.alloc_local((pf,), T.float32)
                nxt_sb = T.alloc_local((pf,), T.float32)
            for t in T.serial(num_steps):
                valid, tm_d, tn_d = decode(t, **dec)
                if valid:
                    tm = tm_d
                    tn = tn_d
                    T.assume(0 <= tm and tm < grid_m)
                    T.assume(0 <= tn and tn < grid_n)
                    if pf > 0:
                        for v in T.vectorized(pf):  # tile's first block: the one exposed load per tile
                            cur_sa[v] = SA[tm * block_M + row, v]
                        for v in T.vectorized(pf):
                            cur_sb[v] = SB[tn, v]
                        for kb in T.serial(n_k // pf):
                            k0n = T.min((kb + 1) * pf, n_k - pf)  # next block (clamped: last block reloads itself)
                            for v in T.vectorized(pf):  # SA row is contiguous along K: one 16/32 B load
                                nxt_sa[v] = SA[tm * block_M + row, k0n + v]
                            for v in T.vectorized(pf):  # same address for all threads: broadcast
                                nxt_sb[v] = SB[tn, k0n + v]
                            if inc:
                                step_stages(psv, ppv, s0, p0, S, pf_steps)
                            for j in pf_steps:
                                ko = kb * pf + j
                                s, ph = stage_of(t * n_k + kb * pf, j, psv, ppv, S, inc)
                                load_ab(A, B, A_s, B_s, full, empty, tm, tn, ko, s, ph, mcast)
                                if dbg:  # TMA just issued (load_ab doesn't wait after it)
                                    if row == 0:
                                        D[bid, t * n_k + ko, 0] = T.call_extern("int64", "clock64")
                                scale_s[s, row] = cur_sa[j] * cur_sb[j]  # registers loaded a block ago
                                group_arrive(full, s, tx, amode, 3)  # id 3 = producer WG
                            if inc:  # (s0, p0) += pf (pf <= S: one wrap at most)
                                if s0 + pf >= S:
                                    s0 = s0 + pf - S
                                    p0 = p0 ^ 1
                                else:
                                    s0 = s0 + pf
                            for j in pf_steps:
                                cur_sa[j] = nxt_sa[j]
                                cur_sb[j] = nxt_sb[j]
                    else:
                        for ko in T.serial(n_k):
                            it = t * n_k + ko
                            s = it % S
                            if ring > 0:
                                if ko % ring == 0:  # block start: TMA the next `ring` k-groups of scales
                                    sbi = it // ring  # global block index (n_k % ring == 0)
                                    T.mbarrier_wait_parity(sempty[sbi % RS], ((sbi // RS) & 1) ^ 1)
                                    T.tma_copy(SA[tm * block_M:(tm + 1) * block_M, ko:ko + ring],
                                               sa_r[sbi % RS, :, :], barrier=sfull[sbi % RS])
                                    T.tma_copy(SB[tn:tn + 1, ko:ko + ring], sb_r[sbi % RS, :, :],
                                               barrier=sfull[sbi % RS])
                                    T.mbarrier_arrive(sfull[sbi % RS])
                            elif not scale_tma:
                                sc = SA[tm * block_M + row, ko] * SB[tn, ko]  # hoisted above the empty wait
                            load_ab(A, B, A_s, B_s, full, empty, tm, tn, ko, s, (it // S) & 1, mcast)
                            if scale_tma:  # async: no thread waits on these; full[s] counts their bytes too
                                k4 = (ko // 4) * 4  # aligned box -> never out of bounds; consumer reads column ko % 4
                                T.tma_copy(SA[tm * block_M:(tm + 1) * block_M, k4:k4 + 4], sa_s[s, :, :],
                                           barrier=full[s])
                                T.tma_copy(SB[tn:tn + 1, k4:k4 + 4], sb_s[s, :, :], barrier=full[s])
                            elif ring == 0:
                                scale_s[s, row] = sc
                            group_arrive(full, s, tx, amode, 3)

        # ---------------- consumers: WG w owns tile rows [w*64, (w+1)*64) ----------------
        for w in CONSUMER_WGS:
            with T.ws(w):
                T.inc_max_nreg(cons_regs)
                tm = T.alloc_var(T.int32)
                tn = T.alloc_var(T.int32)
                sv = T.alloc_local((chunk,), T.int32)  # inc: this chunk's stages / phases (registers)
                pv = T.alloc_local((chunk,), T.int32)
                if inc:
                    s0 = T.alloc_var(T.int32)  # carried (stage, phase) of the next chunk's first k-group
                    p0 = T.alloc_var(T.int32)
                    s0 = 0
                    p0 = 0
                for t in T.serial(num_steps):
                    valid, tm_d, tn_d = decode(t, **dec)
                    if valid:
                        tm = tm_d
                        tn = tn_d
                        T.assume(0 <= tm and tm < grid_m)
                        T.assume(0 <= tn and tn < grid_n)
                        T.clear(acc[w])
                        for c in T.serial(n_k // chunk):
                            base = t * n_k + c * chunk  # global k index of the chunk's first k-group
                            if ring > 0:  # chunk boundary: no wgmma in flight here (C7514-safe)
                                if (c * chunk) % ring == 0:
                                    sbi = base // ring
                                    T.mbarrier_wait_parity(sfull[sbi % RS], (sbi // RS) & 1)
                            if inc:
                                step_stages(sv, pv, s0, p0, S, all_steps)
                            s, ph = stage_of(base, 0, sv, pv, S, inc)
                            issue(A_s, B_s, full, parts[0][w], s, ph, w, D, dbg, bid, base, tx)
                            for j in steps:  # straight-line inside a chunk
                                s, ph = stage_of(base, j + 1, sv, pv, S, inc)
                                issue(A_s, B_s, full, parts[(j + 1) % 2][w], s, ph, w, D, dbg, bid, base + j + 1, tx)
                                T.wait_wgmma(1)
                                s, ph = stage_of(base, j, sv, pv, S, inc)
                                promote(acc[w], parts[j % 2][w], scale_s, sa_s, sb_s, sa_r, sb_r, empty, base + j, s, w,
                                        tx, peer, mcast, scale_tma, ring, amode, scf[w], early)
                            T.wait_wgmma(0)  # chunk end: nothing in flight across the back-edge
                            s, ph = stage_of(base, chunk - 1, sv, pv, S, inc)
                            promote(acc[w], parts[(chunk - 1) % 2][w], scale_s, sa_s, sb_s, sa_r, sb_r, empty,
                                    base + chunk - 1, s, w, tx, peer, mcast, scale_tma, ring, amode, scf[w], early)
                            if inc:  # (s0, p0) += chunk (chunk <= S: one wrap at most)
                                if s0 + chunk >= S:
                                    s0 = s0 + chunk - S
                                    p0 = p0 ^ 1
                                else:
                                    s0 = s0 + chunk
                            if ring > 0:
                                if (c * chunk + chunk) % ring == 0:  # chunk that ends a block: release its slot
                                    T.mbarrier_arrive(sempty[(base // ring) % RS])
                        # register epilogue; the producer is already loading the next tile meanwhile
                        if tma_epi:  # async epilogue (bf16 demo v7 pattern), named barriers 4 + w
                            T.tma_store_wait(0)  # previous tile's store finished READING my half of C_s
                            T.sync_threads(4 + w, 128)  # ...and the whole WG knows it
                            T.copy(acc[w], C_s[w, :, :])
                            T.fence_proxy_async()  # generic st.shared -> visible to the async (TMA) proxy
                            T.sync_threads(4 + w, 128)
                            T.tma_copy(C_s[w, :, :], C[tm * block_M + w * WG_M:tm * block_M + (w + 1) * WG_M,
                                                       tn * block_N:(tn + 1) * block_N])
                        else:
                            T.copy(acc[w], C[tm * block_M + w * WG_M, tn * block_N])
                if tma_epi:
                    T.tma_store_wait(0)  # last store must finish reading C_s before the CTA exits

        if clustered:
            T.cluster_sync()  # don't exit while the peer may still arrive on my empty barriers

    if dbg:
        return C, D
    return C


def make(stages: int = 4, swizzle: int = 8, chunk: int = 4, mode: str = "none", scale_tma: bool = False,
         pf: int = 4, ring: int = 0, arrive: str = "all", inc: bool = False, serp: bool = False,
         early: bool = False, tma_epi: bool = True):
    """Harness contract: fn(a, b, sa, sb) -> c bf16."""

    def fn(a, b, sa, sb):
        return fp8_gemm_persistent(a, b, sa, sb, stages=stages, swizzle=swizzle, chunk=chunk, mode=mode,
                                   scale_tma=scale_tma, pf=pf, ring=ring, arrive=arrive, inc=inc, serp=serp,
                                   early=early, tma_epi=tma_epi)

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
    import gemm_ws_tma_async_2part as v1  # same-run reference (non-persistent, no cluster)

    a, b, sa, sb = make_inputs(0, "scale_stress")
    fns = {}
    # Same-run comparison of the current best against the knobs tried on top of it (all S=4, chunk=4, P=8, pf=4).
    fns["best (TMA epilogue)"] = make()
    fns["register epilogue"] = make(tma_epi=False)
    fns["early stage release"] = make(early=True)
    fns["serpentine"] = make(serp=True)
    fns["mcast P=16"] = make(swizzle=16, mode="mcast")

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
        print(f"  {name:<22} max|err|/max|ref| = {rel:.2e}")

    flops = 2 * M * N * K
    for name, ms in bench_interleaved({k: (lambda f=f: f(a, b, sa, sb)) for k, f in fns.items()}).items():
        print(f"  {name:<22} {ms * 1e3:8.1f} us   {flops / ms / 1e9:7.1f} TFLOP/s")


if __name__ == "__main__":
    main()
