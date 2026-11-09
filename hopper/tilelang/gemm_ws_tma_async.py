"""Hopper warp-specialized GEMM: TMA + mbarriers + threadblock swizzle + ASYNC wgmma.

    C[M, N] = A[M, K] @ B[N, K]^T        (bf16 in, fp32 accumulate, bf16 out)

Builds on gemm_ws_tma_tbswizzle.py. The new part is the consumer loop:

  sync  (T.gemm):        wait full[s] -> issue wgmma(s) -> wait<0> (drain!) -> free s
  async (T.wgmma_gemm):  wait full[s] -> issue wgmma(s) -> wait<1>          -> free s-1

With wait<1> the batch for stage s stays in flight while we do the bookkeeping, so the tensor cores always have the
next batch queued (no drain bubble). Stage s can't be freed yet (its wgmma may still read smem); stage s-1 can,
because wgmma batches complete in issue order and wait<1> just confirmed batch s-1 is done.
Barriers are unchanged; only *when* the consumer waits and arrives moved. The producer is unchanged.
"""

import os

import torch
import tilelang
import tilelang.language as T

NUM_STAGES = 4


# We own all synchronization (mbarriers), so tilelang's automatic __syncthreads() insertion must be off
# (it causes lockstep or deadlock between the warpgroups; see kernels/learnings/hopper_gemm_ws.md).
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def gemm_ws(A, B, block_M: int = 128, block_N: int = 128, block_K: int = 64, swizzle: int = 8,
            async_mma: bool = True):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)
    C = T.empty((M, N), T.bfloat16)

    S = NUM_STAGES
    n_k = T.ceildiv(K, block_K)
    barrier_counts = [128] * S + [128] * S  # [0,S) = full[s], [S,2S) = empty[s]

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        # Threadblock swizzle: panels of `swizzle` tile-rows swept column by column -> better L2 reuse of B.
        T.use_swizzle(panel_size=swizzle, enable=swizzle > 0)

        A_s = T.alloc_shared((S, block_M, block_K), T.bfloat16)
        B_s = T.alloc_shared((S, block_N, block_K), T.bfloat16)
        C_frag = T.alloc_fragment((block_M, block_N), T.float32)
        bars = T.alloc_barrier(barrier_counts)

        with T.ws(0):
            T.clear(C_frag)

        # ---------------- producer warpgroup (unchanged) ----------------
        with T.ws(1):
            for ko in T.serial(n_k):
                s = ko % S
                phase = (ko // S) & 1
                T.mbarrier_wait_parity(bars[S + s], phase ^ 1)  # slot s free?
                T.tma_copy(A[by * block_M:(by + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                           A_s[s, :, :], barrier=bars[s])
                T.tma_copy(B[bx * block_N:(bx + 1) * block_N, ko * block_K:(ko + 1) * block_K],
                           B_s[s, :, :], barrier=bars[s])
                T.mbarrier_arrive(bars[s])

        # ---------------- consumer warpgroup ----------------
        with T.ws(0):
            for ko in T.serial(n_k):
                s = ko % S
                phase = (ko // S) & 1
                T.mbarrier_wait_parity(bars[s], phase)  # stage s landed?
                if async_mma:
                    # issue only: no implicit wait. Batch ko joins the tensor-core queue behind batch ko-1.
                    T.wgmma_gemm(A_s[s, :, :], B_s[s, :, :], C_frag, transpose_B=True)
                    T.wait_wgmma(1)  # <=1 batch in flight -> batch ko-1 is done (in-order completion)
                    if ko > 0:
                        T.mbarrier_arrive(bars[S + (ko - 1) % S])  # free the PREVIOUS stage
                else:
                    T.gemm(A_s[s, :, :], B_s[s, :, :], C_frag, transpose_B=True)  # issue + wait<0> (drain)
                    T.mbarrier_arrive(bars[S + s])
            if async_mma:
                T.wait_wgmma(0)  # last batch must finish writing C_frag before the epilogue reads it

        with T.ws(0):
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return C


def bench_interleaved(fns: dict, rounds=7, iters=10):
    """Round-robin timing so clock/power throttling hits every variant equally; returns median ms per call."""
    times = {k: [] for k in fns}
    for f in fns.values():  # warm-up
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

    common = dict(M=M, N=N, K=K, block_M=128, block_N=128, block_K=64, swizzle=8)
    kernels = {"sync  (T.gemm)": gemm_ws.compile(**common, async_mma=False),
               "async (wgmma_gemm + wait<1>)": gemm_ws.compile(**common, async_mma=True)}

    if os.environ.get("PROFILE"):  # under ncu: one launch each, order = sync, async, cuBLAS
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
    print("correctness: PASS (sync, async)")

    fns = {name: (lambda k=k: k(a, b)) for name, k in kernels.items()}
    fns["cuBLAS"] = lambda: a @ b.T
    flops = 2 * M * N * K
    for name, ms in bench_interleaved(fns).items():
        print(f"  {name:<30} {ms * 1e3:8.1f} us   {flops / ms / 1e9:6.1f} TFLOP/s")

    os.makedirs("/tmp/out", exist_ok=True)
    with open("/tmp/out/gemm_ws_tma_async.generated.cu", "w") as f:
        f.write(kernels["async (wgmma_gemm + wait<1>)"].get_kernel_source())


if __name__ == "__main__":
    main()
