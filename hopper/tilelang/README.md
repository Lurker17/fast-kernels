# Hopper warp-specialized GEMM in tilelang

`C[M, N] = A[M, K] @ B[N, K]^T`, bf16 in, fp32 accumulate, bf16 out, M = N = K = 8192, H100 SXM.
tilelang 0.1.15, CUDA 12.8. Each file is standalone: `python <file>.py` checks correctness against cuBLAS and benchmarks both.

Every kernel writes its synchronization by hand (TMA + mbarriers + `T.ws` warp specialization), so read them in order:

| step | file | change | TFLOP/s |
|---|---|---|---|
| 1 | `gemm_ws_tma.py` | producer/consumer warpgroups, TMA into a 4-slot smem ring, `full`/`empty` mbarriers, sync wgmma; tilelang's auto sync pass turned off | 516 |
| 2 | `gemm_ws_tma_tbswizzle.py` | + threadblock swizzle (better L2 reuse) | — |
| 3 | `gemm_ws_tma_async.py` | + async wgmma (`wait<1>`), slot freed one step late | 610 |
| 4 | `gemm_ws_tma_2consumers.py` | + 2 consumer warpgroups on a 256×128 tile | 654–690 |
| 5 | `gemm_ws_persistent.py` | + persistent kernel (132 CTAs), register epilogue | 682 (cuBLAS 733, same run) |
| 6 | `gemm_ws_cluster.py` | + 2-CTA clusters, no multicast yet (slightly slower) | 640 |
| 7 | `gemm_ws_cluster_multicast.py` | + TMA multicast of A across the cluster | 683 (cuBLAS 703, same run) |

Numbers are back-to-back CUDA-event benchmarks on Modal H100s. Clocks weren't locked and cuBLAS itself varied
677–755 TFLOP/s between runs, so only compare numbers from the same run. Step 2's bench number was unreliable
(run-order bias), its ncu time went 2.03 → 1.84 ms.

## The bug that step 1 fixes

The first working version had both roles inside one k-loop and ran at 216 TFLOP/s. tilelang's shared-memory
sync pass (`ThreadSync("shared")`) can't see that the mbarriers already order producer and consumer, so it put a
`__syncthreads()` in every iteration and the two warpgroups ran in lockstep. Splitting the loops alone deadlocks
(the pass puts one `__syncthreads()` between them). Separate loops plus

```python
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
```

gives the pipelined 516 TFLOP/s kernel. Once the pass is off, every other sync is yours: use named barriers
(`T.sync_threads(barrier_id, arrive_count)`) inside a `T.ws` branch, never a full-block `__syncthreads()`.
