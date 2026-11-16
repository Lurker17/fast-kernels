# FP8 block-scaled GEMM on Hopper (DeepSeek-V3 scaling), tilelang

`C[M, N] = dequant(A) @ dequant(B)^T`, fp8 e4m3 in, fp32 accumulate, bf16 out, M = N = K = 8192, H100 SXM.
A (activations) has one fp32 scale per row per 128 of K (`SA[M, K/128]`), B (weights) one per 128×128 block
(`SB[N/128, K/128]`). The full spec, accuracy bound and benchmark protocol are in [`problem_desc.md`](problem_desc.md).

| file | what |
|---|---|
| `problem_desc.md` | problem, layouts, correctness bound, roofline, design questions |
| `harness.py` | inputs (3 distributions), fp64 reference + accuracy bound, determinism check, baselines, CUDA-graph benchmark |
| `bench.py` | CUDA-graph timing with rotating inputs (cold L2) |
| `torch_ref.py` | dequantize + fp32 matmul; checks the harness itself |
| `gemm_ws_tma_async.py` | kernel 1: the bf16 pipeline (`../gemm_ws_tma_async.py`) ported to fp8 with per-group promotion |
| `gemm_ws_tma_async_2part.py` | kernel 2: two partial accumulators per consumer, promotion overlapped with the next group's wgmmas |
| `ab_stages_2part.py` | same-run A/B of kernels 1 and 2 at 4 and 5 stages |

```
python harness.py                      # correctness + benchmark of gemm_ws_tma_async
python harness.py torch_ref            # harness self-check
python gemm_ws_tma_async.py            # quick config sweep (stages, swizzle, scale-load hoisting)
```

## Kernel 1: `gemm_ws_tma_async.py`

- 128×128 tile, `block_K = 128` = one scale group, 4 stages × 32.5 KB, 384 threads.
- Producer warpgroup: TMA for A and B, and each of its 128 threads loads one row's `SA[m, g] · SB[n/128, g]` into
  `scale_s[s, row]` before arriving on `full[s]`. (SA's per-tile column is strided by 256 B, too narrow for TMA, and
  the spec forbids re-laying it out outside the call.)
- Two 64-row consumer warpgroups, each with a `part` and an `acc` accumulator (64 + 64 registers per thread; one
  128-row consumer would need 256 > 232). Per k-group: 4 wgmmas into `part`, `wait_wgmma(0)`,
  `acc += part * scale_s[row]`, arrive `empty[s]`. Promotion is not yet overlapped with the next group's wgmmas.

**Status (one run, H100 SXM, torch 2.9 / CUDA 12.8):**

| | ms | TFLOP/s |
|---|---|---|
| this kernel | 1.236 | 890 |
| cuBLAS fp8, per-tensor scale (speed only, no block scaling) | 0.872 | 1261 |
| fp8 rowwise `_scaled_mm` (speed only) | 1.010 | 1089 |
| cuBLAS bf16 `F.linear` | 1.455 | 756 |

cuBLAS's 1×128 / 128×128 blockwise mode needs CUDA ≥ 12.9 and DeepGEMM isn't installed in the image yet, so there
is no like-for-like baseline yet.

**Correctness: not passing yet.** `normal` (err/bound 0.38) and `scale_stress` (0.78) pass; `uniform_pos` fails by 2%
(worst err/bound 1.02, 117K of 67M elements). Scale indexing is right (scale_stress would fail loudly otherwise); the
excess is accumulation error on all-positive sums. Note the bound's rounding term (2⁻⁸·|ref|) is only ½ ulp at the
bottom of a bf16 binade, not the 2× margin `problem_desc.md` claims, so the bound is tighter than intended there.

## Kernel 2: `gemm_ws_tma_async_2part.py`

Kernel 1's consumer waits for its own batch (`wait_wgmma(0)`) and then promotes with nothing queued on the tensor
cores. Kernel 2 gives each consumer two partial buffers (`pa`, `pb`): issue group k+1 into the other buffer,
`wait_wgmma(1)` (k done), promote k while k+1 runs. Registers: acc + pa + pb = 64 + 64 + 64 = 192 per thread.

**Chunked K loop.** In a plain loop over K a wgmma batch is still running when the loop jumps back to the top.
ptxas can't track it across the jump, so it puts a wait after every wgmma in the kernel (warning C7514; in SASS every
`QGMMA` is followed by `WARPGROUP.DEPBAR.LE gsb0, 0x0`), which turns the overlap off. So the K loop runs in chunks:
an outer `T.serial` over chunks, steps inside a chunk unrolled in Python (straight-line code), and `wait_wgmma(0)` at
each chunk end so nothing is in flight at the jump. With chunks the SASS shows the intended pattern (two batches
queued, `DEPBAR.LE gsb0, 0x1`, promotion FFMAs). Chunks of 2, 4 and 8 groups run within 0.5% of each other.

**Same-run A/B (`ab_stages_2part.py`, H100 SXM, one container):**

| config | bench TFLOP/s | ncu time | tensor busy |
|---|---|---|---|
| kernel 1, 4 stages | 876 | 1.131 ms | 54.8% |
| kernel 1, 5 stages | 806 | 1.150 ms | 53.8% |
| kernel 2, 4 stages | 925 | 1.085 ms | 57.1% |
| kernel 2, 5 stages (default) | 931 | 1.069 ms | 58.5% |

The overlap is worth ~4% of kernel time at equal stages: the second consumer warpgroup was already covering most of
the promotion gap. The interleaved bench moved ~6% between runs for the same config, so use ncu time for small
differences. Output is bitwise identical to kernel 1, so the accuracy status above is unchanged.
