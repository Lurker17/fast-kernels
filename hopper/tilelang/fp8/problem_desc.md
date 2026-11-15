# FP8 block-scaled GEMM (DeepSeek-V3 style, Hopper)

```
C[M, N] = dequant(A)[M, K] @ dequant(B)[N, K]^T        (bf16 out, fp32 accumulate)

dequant(A)[m, k] = A[m, k] * SA[m, k // 128]                 1x128 groups: per row, per 128 of K
dequant(B)[n, k] = B[n, k] * SB[n // 128, k // 128]          128x128 blocks
```

This is the quantization used by DeepSeek-V3 training/inference, DeepGEMM, and the FP8 paths of vLLM/SGLang on H100.
Activations (A) get a fine 1x128 scale because their outliers are per token; weights (B) get a coarse 128x128 block scale.

## Why this scheme (and not per-tensor / per-row)

| scheme | where the scale is applied | what's new vs the bf16 demo |
|---|---|---|
| per-tensor (1 scalar each) | epilogue: `C *= sa * sb` | only the fp8 wgmma; the kernel is the bf16 one with a multiply |
| per-row / per-channel | epilogue: `C[m, n] *= sa[m] * sb[n]` | same, with an outer product in the epilogue |
| **1x128 x 128x128 (this)** | **mainloop**: the scale changes every 128 of K | scales flow through the pipeline, a second accumulator, promotion math overlapped with MMA |

Per-tensor and per-row are good warm-ups for bringing up fp8 wgmma (v0), but the scale never enters the K loop.
Block scaling does, and it also fixes a real Hopper problem: the fp8 wgmma accumulates with **reduced precision**
(DeepSeek-V3 report, §3.3.2: ~14 bits), so accumulating all of K inside the tensor core loses accuracy. Promoting into
an fp32 register accumulator every 128 of K is required for accuracy, not just for applying the scales.
(MXFP8 / hardware block scaling is Blackwell-only; on sm_90 the scaling is done in software.)

## Inputs

| Name | Shape | Dtype | Layout | Notes |
|---|---|---|---|---|
| `A` | `[M, K]` = `[8192, 8192]` | `float8_e4m3fn` | row-major (K-major) | activations |
| `B` | `[N, K]` = `[8192, 8192]` | `float8_e4m3fn` | row-major (K-major) | weights, `nn.Linear` layout |
| `SA` | `[M, K/128]` = `[8192, 64]` | fp32 | row-major, contiguous | one scale per row per 128 of K |
| `SB` | `[N/128, K/128]` = `[64, 64]` | fp32 | row-major, contiguous | one scale per 128x128 block |

Layouts match tilelang's `examples/deepseek_deepgemm`. An implementation may not pre-transpose or re-layout inputs
outside the timed call (if it wants SA K-group-major, for TMA, it has to pay for that inside the call).
M, N, K are multiples of 128; no tails.

Both fp8 operands are K-major. Hopper's fp8 wgmma **requires** that for both A and B in shared memory (no transpose
like bf16 allows), so this layout is the one the hardware wants.

## Output

| Name | Shape | Dtype | Layout |
|---|---|---|---|
| `C` | `[M, N]` | bf16 | row-major, contiguous |

## Correctness requirements

The reference is exact: `(A.double() * SA_expanded) @ (B.double() * SB_expanded).T` in fp64. Inputs are already
quantized, so quantization error is not part of the check: the kernel is judged on GEMM error only.

- **Accuracy:** per element, `|C − ref| ≤ 2⁻⁸·|ref| + 2⁻¹¹·Σ_k |dequant(A)[m,k]·dequant(B)[n,k]|`.
  The first term is bf16 output rounding (½ ulp = 2⁻⁹, ×2 margin). The second is accumulation error; it is loose on
  purpose. The harness also prints the measured `max err / bound` for the impl and the baselines, so you can see how
  much headroom you have; tighten if the baselines sit far below 1.
- **Scale-stress inputs:** one test makes scales vary by up to 2¹⁶ across K-groups and blocks. Using the wrong
  group index, one scale per tile, or applying scales after the full-K sum fails this test loudly.
- **Determinism:** two identical calls return bitwise-identical output (rules out atomic split-K).

## Target and roofline

- GPU: H100 SXM (sm_90), 132 SMs. FP8 dense peak **1979 TFLOP/s** (spec sheet; bf16 is 989), HBM ≈ 3.35 TB/s, L2 50 MB.
- FLOPs: 2·8192³ = **1.10 TFLOP** → floor **0.556 ms** at peak.
- Bytes: A 67 MB + B 67 MB + C 134 MB + SA 2.1 MB + SB 16 KB ≈ 270 MB → 81 µs. ~4000 FLOP/B, compute-bound.
- Compare: the bf16 demo at the same shape took ~1.47 ms (v8.2) / ~1.40 ms (cuBLAS). FP8 should roughly halve that.

## Baselines (harness tries each; some may be unavailable on torch 2.9 + CUDA 12.8)

| baseline | what it is | use |
|---|---|---|
| `torch._scaled_mm`, 1x128 / 128x128 blockwise | cuBLAS blockwise (needs cuBLAS ≥ 12.9) | **the real target**, if available |
| `torch._scaled_mm`, per-tensor | cuBLAS fp8, scales only in epilogue | speed ceiling for fp8 without block scaling |
| `torch._scaled_mm`, rowwise | per-row / per-col scales | ditto |
| `F.linear` bf16 | cuBLAS bf16 on the same shape | what FP8 has to beat |
| DeepGEMM (not installed yet) | the reference implementation of this exact scheme | add to the image if blockwise cuBLAS is unavailable |

Per-tensor and rowwise baselines run on the same fp8 bytes with different scales: **speed only**, their accuracy
numbers are not comparable.

## Benchmark protocol

- CUDA graph, 4 rotating input sets (each set is 136 MB > L2), median per call.
- Report ms, TFLOP/s, % of fp8 peak, and the same for every baseline in the same run.

## Things to figure out (design questions, not answers)

1. **Register budget.** v7/v8 gave each consumer warpgroup a 128x128 fp32 accumulator = 128 regs/thread out of 232.
   Promotion needs a second accumulator of the same size. What tile/warpgroup shape fits?
2. **Getting scales to the right threads.** Each k-group, a consumer needs `SA` for the rows it owns and one `SB`
   scalar. Through the TMA pipeline with A/B, or plain loads? When must they arrive?
3. **Overlap.** Promotion is ~2 FMAs per accumulator element on CUDA cores every 128 of K. Can it run while the
   next k-group's wgmmas are in flight (it needs the *previous* group's accumulator to be finished)?
4. **block_K = 128 fp8 = 128 bytes**, exactly one 128B-swizzle row, and one wgmma is k=32, so 4 wgmmas per k-group.
