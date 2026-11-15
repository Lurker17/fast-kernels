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
| `gemm_ws_tma_async.py` | first kernel: the bf16 pipeline (`../gemm_ws_tma_async.py`) ported to fp8 with per-group promotion |

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
