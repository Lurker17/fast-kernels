"""Teaching example: Hopper (sm_90) GEMM in tilelang with everything explicit.

    C[M, N] = A[M, K] @ B[N, K]^T        (bf16 in, fp32 accumulate, bf16 out; B in nn.Linear layout)

Block = 256 threads = 2 warpgroups (a warpgroup = 4 warps = 128 threads, the unit wgmma runs on):

    warpgroup 1  = PRODUCER: issues TMA loads of A/B tiles into a ring of NUM_STAGES shared buffers
    warpgroup 0  = CONSUMER: waits for a stage to land, runs wgmma on it, hands the stage back

Two mbarriers per stage form the producer/consumer handshake:

    full[s]   "stage s has data"   producer's TMA completes it  ->  consumer waits on it
    empty[s]  "stage s is free"    consumer arrives after wgmma ->  producer waits on it

An mbarrier completes a *phase* when all expected arrivals (and, for TMA, all expected bytes) are in;
its phase bit then flips 0 -> 1 -> 0 ... Waiters say which parity they're waiting to see complete.
Stage s is reused every NUM_STAGES iterations, so on iteration ko it is in round ko // NUM_STAGES,
and the parity for that round is (ko // NUM_STAGES) & 1.
"""

import torch
import tilelang
import tilelang.language as T

NUM_STAGES = 4


# We own all synchronization (mbarriers). tilelang's automatic shared-memory sync pass can't see that
# the mbarriers already order producer/consumer accesses, so it inserts __syncthreads() between the
# two roles: inside a shared loop -> lockstep (no pipelining); between separate loops -> deadlock
# (consumer stuck at the barrier while the producer waits for a free stage). So: turn it off.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def gemm_ws(A, B, block_M: int = 128, block_N: int = 128, block_K: int = 64):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)
    C = T.empty((M, N), T.bfloat16)

    S = NUM_STAGES
    # Barrier ids: [0, S) = full[s], [S, 2S) = empty[s]. Each expects 128 thread arrivals (one warpgroup).
    barrier_counts = [128] * S + [128] * S

    # 2-D grid: bx -> blockIdx.x walks N tiles, by -> blockIdx.y walks M tiles.
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        # Shared-memory ring: S stages of A and B tiles (4 * (16 KB + 16 KB) = 128 KB).
        A_s = T.alloc_shared((S, block_M, block_K), T.bfloat16)
        B_s = T.alloc_shared((S, block_N, block_K), T.bfloat16)
        # Accumulator: a 128x128 fp32 tile spread across the consumer warpgroup's registers
        # (wgmma's accumulator layout; the compiler picks which thread holds which element).
        C_frag = T.alloc_fragment((block_M, block_N), T.float32)
        bars = T.alloc_barrier(barrier_counts)

        with T.ws(0):
            T.clear(C_frag)

        # Each role runs its OWN k-loop. If both roles share one loop, tilelang's shared-memory hazard
        # pass inserts __syncthreads() every iteration and the two warpgroups march in lockstep
        # (pipeline never fills: measured 216 vs ~600+ TFLOP/s). Separate loops = the producer runs
        # up to S stages ahead, throttled only by the mbarriers.

        # ---------------- producer warpgroup ----------------
        with T.ws(1):
            for ko in T.serial(T.ceildiv(K, block_K)):
                s = ko % S
                phase = (ko // S) & 1
                # Wait until the consumer has freed stage s. On the first lap (phase 0) we wait on parity 1,
                # i.e. the phase "before" the first one, which counts as already complete, so this falls through.
                T.mbarrier_wait_parity(bars[S + s], phase ^ 1)
                # TMA: one elected thread issues a bulk async copy global->shared and tells full[s]
                # to expect this many bytes (expect_tx). The hardware decrements the byte count as data lands.
                T.tma_copy(A[by * block_M:(by + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                           A_s[s, :, :], barrier=bars[s])
                T.tma_copy(B[bx * block_N:(bx + 1) * block_N, ko * block_K:(ko + 1) * block_K],
                           B_s[s, :, :], barrier=bars[s])
                # All 128 producer threads arrive. full[s] completes when the arrivals are in AND the bytes landed.
                T.mbarrier_arrive(bars[s])

        # ---------------- consumer warpgroup ----------------
        with T.ws(0):
            for ko in T.serial(T.ceildiv(K, block_K)):
                s = ko % S
                phase = (ko // S) & 1
                T.mbarrier_wait_parity(bars[s], phase)  # wait for the tiles of stage s to land
                # wgmma: operands read straight from shared memory (no smem->register copy), acc in registers.
                # T.gemm is synchronous: it emits wgmma.fence / wgmma.mma_async ... / commit / wait<0>.
                T.gemm(A_s[s, :, :], B_s[s, :, :], C_frag, transpose_B=True)
                T.mbarrier_arrive(bars[S + s])  # stage s is free again

        # Epilogue: registers -> global (bf16 cast happens in the copy).
        with T.ws(0):
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return C


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    M = N = K = 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    kernel = gemm_ws.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=64)

    import os
    if os.environ.get("PROFILE"):  # under ncu: one launch each of ours and cuBLAS, inside an NVTX range
        kernel(a, b)
        a @ b.T  # warm-up (JIT, cuBLAS heuristics) outside the profiled range
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile")
        kernel(a, b)
        a @ b.T
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return

    c = kernel(a, b)
    ref = a @ b.T
    torch.testing.assert_close(c.float(), ref.float(), rtol=2e-2, atol=2e-1)
    print("correctness: PASS")

    flops = 2 * M * N * K
    for name, fn in [("tilelang ws+tma+wgmma", lambda: kernel(a, b)), ("cuBLAS (a @ b.T)", lambda: a @ b.T)]:
        ms = bench(fn)
        print(f"  {name:<24} {ms * 1e3:8.1f} us   {flops / ms / 1e9:6.1f} TFLOP/s")

    import os
    os.makedirs("/tmp/out", exist_ok=True)
    with open("/tmp/out/gemm_ws_tma.generated.cu", "w") as f:
        f.write(kernel.get_kernel_source())
    print("wrote generated CUDA")


if __name__ == "__main__":
    main()
