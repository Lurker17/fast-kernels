# Fused residual add + RMSNorm + per-token-group FP8 quant (DeepSeek-V3 style, Hopper)

```
h[m, :]  = bf16( x[m, :] + residual[m, :] )                                  new residual stream, written out (bf16)
y[m, :]  = h[m, :] * rsqrt( mean_k(h[m, k]^2) + eps ) * weight[:]              fp32, never rounded to bf16
s[m, g]  = max( amax_{k in group g} |y[m, k]|, 1e-10 ) / 448                   one fp32 scale per row per 128 channels
q[m, k]  = fp8_e4m3( y[m, k] / s[m, k // 128] )                               round to nearest, |q| <= 448
```

Every transformer block of an fp8 model runs this before its first GEMM: add the sublayer output to the residual
stream, normalize, and quantize the result for the next fp8 matmul. `q` and `s` are exactly the `A` and `SA` operands of
[`../fp8`](../fp8) (1x128 groups, `SA[M, K/128]` row-major), so this kernel feeds that one. In vLLM / SGLang this is
`fused_add_rms_norm` followed by `per_token_group_quant_fp8`: two kernels, `y` written to and read back from HBM.
Here it is one pass.

Unlike the GEMMs, this is **memory-bound**: ~7 bytes of HBM traffic per element and ~10 FLOPs. The roofline is HBM
bandwidth, and the work is in reductions, vectorized access, and keeping enough bytes in flight.

## Inputs

| Name | Shape | Dtype | Layout | Notes |
|---|---|---|---|---|
| `x` | `[M, H]` | bf16 | row-major, contiguous | sublayer output (attention / MLP) |
| `residual` | `[M, H]` | bf16 | row-major, contiguous | residual stream |
| `weight` | `[H]` | bf16 | contiguous | RMSNorm gain |

`eps = 1e-6` (DeepSeek-V3's `rms_norm_eps`), a compile-time constant. `H` is a multiple of 128. `M` is anything ≥ 1
(no padding: the correctness shapes include `M = 1000`).

**Shapes:** `H = 7168` (DeepSeek-V3 hidden size) for timing. `M = 16384` is the main number (a prefill / training
micro-batch of tokens); `M = 2048` and `M = 128` (decode-sized batch) are reported too. Correctness also runs `H = 4096`.

## Outputs (a tuple, all freshly allocated)

| Name | Shape | Dtype | Layout |
|---|---|---|---|
| `h` | `[M, H]` | bf16 | row-major, contiguous |
| `q` | `[M, H]` | `float8_e4m3fn` | row-major, contiguous |
| `s` | `[M, H/128]` | fp32 | row-major, contiguous (the fp8 GEMM's `SA` layout) |

Inputs must not be modified (production kernels update `residual` in place; here it's out-of-place so the benchmark can
replay the same inputs; the bytes moved are the same).

## Correctness requirements

The reference is computed in fp64 from the bf16 `h`.

- **`h` is bit-exact:** `h == (x.float() + residual.float()).to(bf16)`. One rounding of the fp32 sum. The norm is
  computed on this rounded `h`, as vLLM's `fused_add_rms_norm` does.
- **Scales:** `|s − s_ref| ≤ 2⁻¹⁶ · s_ref` per element. (fp32 `rsqrt`/`amax` vs fp64; loose enough for any order of
  summation, tight enough to catch a wrong group or a missing `eps`.)
- **Quantized values:** with `t = y_ref / s_ref` (the exact value in fp8 units, `|t| ≤ 448`),
  `|q − t| ≤ ½·ulp_e4m3(t) + 2⁻¹⁶·|t|` per element: correct rounding up to fp32 noise near a rounding midpoint. Rounding
  `y` to bf16 before quantizing breaks this (bf16's 2⁻⁹ relative error flips fp8 roundings), so keep `y` in fp32.
  No NaN, `|q| ≤ 448`; each group's largest element lands on ±448.
- **Determinism:** two identical calls return bitwise-identical outputs. Inputs unchanged after the call.

Test inputs (`harness.CASES`):

| dist | what it catches |
|---|---|
| `normal` | x, residual ~ N(0, 1), weight ~ 1 + 0.1·N(0, 1) |
| `outliers` | 8 fixed channels ×100 and ~0.1% random elements ×30 (LLM activation outliers): neighbouring groups' scales differ by 100×, so a wrong group index fails loudly |
| `row_scale` | each row scaled by 2^U[−24, 16]: rows whose mean square is far below `eps` (tests where `eps` goes), plus 3 groups per row that are exactly zero after the add (`residual = −x`), so `amax = 0` hits the `1e-10` clamp |

## Target and roofline

- GPU: H100 SXM (sm_90), 132 SMs, HBM ≈ **3.35 TB/s** (spec; a plain read+write copy reaches ~90% of that), L2 50 MB.
- Bytes per element: read `x` 2 + `residual` 2, write `h` 2 + `q` 1 = **7 B**, plus `s` (4 B per 128 elements = 0.03 B)
  and `weight` (14 KB, once).
- FLOPs: ~10 per element (add, square, accumulate, two multiplies, abs/max, divide, convert): < 1.5 FLOP/byte.

| M × H | bytes | floor at 3.35 TB/s |
|---|---|---|
| **16384 × 7168** | **826 MB** | **247 µs** |
| 2048 × 7168 | 103 MB | 31 µs |
| 128 × 7168 | 6.5 MB | 1.9 µs (fits in L2, and 128 rows < 132 SMs: latency- and occupancy-bound, not HBM) |

## Benchmark protocol

- CUDA graph, rotating input sets so the working set exceeds L2 (2 sets at M=16384, more for smaller M), median
  per call.
- Report µs, achieved GB/s (problem bytes / time) and % of 3.35 TB/s, per shape.

## Baselines

| baseline | what it is | use |
|---|---|---|
| torch eager | the fp32 reference code (`torch_ref.py`): ~10 kernels, `y` round-trips through HBM | the floor to beat by a lot |
| `torch.compile` of the same code | Inductor/Triton fuses it into a few kernels | **the real target** |
| `x + residual` [speed] | read 4 B, write 2 B per element | achievable HBM bandwidth on this GPU, today (its own bytes) |

vLLM / SGLang's fused CUDA kernels are the production reference; not installed (heavy image). Add later if
`torch.compile` turns out to be a weak baseline.

**Measured (2026-11-19, H100 SXM, `logs/20261119-rmsnorm-harness-torchref.log`):**

| M × H | floor | torch eager | **torch.compile** | `x + residual` probe (own bytes) |
|---|---|---|---|---|
| 16384 × 7168 | 246.5 µs | 4477 µs (5.5% HBM) | **366.8 µs (2251 GB/s, 67%)** | 232.9 µs (3026 GB/s, 90%) |
| 2048 × 7168 | 30.8 µs | 604 µs | **50.3 µs (61%)** | 32.8 µs (80%) |
| 128 × 7168 | 1.9 µs | 61.9 µs | **6.1 µs (32%)** | 3.0 µs (55%) |

The probe says ~3.0 TB/s is reachable: at the main shape that's **~275 µs** for this problem's 826 MB, 1.33× faster
than `torch.compile`.

Notes from validating the harness:
- `torch.compile`'s output fails the scale tolerance (up to 2.5e-3 relative on `outliers`, 1.2e-4 on `normal`) even
  with Inductor's `emulate_precision_casts`; its `h` is bit-exact. Not investigated: treat it as the speed target only.
- About 1 in 10⁴ `q` values differ from the fp64 round-to-nearest (`q != RN(t)` in the harness output), the same for
  torch eager: fp32-vs-fp64 noise at rounding midpoints, all within the bound. Expect the same order from a kernel.

## Things to figure out (design questions, not answers)

1. **Who owns a row?** One row is 7168 bf16 = 14 KB per input. One warp per row (224 elements per lane = 28 × 16 B
   vectors per input), one warpgroup, or one CTA? The choice decides how the row sum of squares is reduced (shuffles
   only, or shuffles + shared memory) and how many rows are in flight per SM.
2. **One pass or two?** `rstd` needs the whole row before any `y` exists, and `s` needs the whole group of `y`. Can
   the group `amax` be computed in the same pass as the sum of squares? (Hint: `rstd > 0`.) Then: keep the row in
   registers, re-read it from shared memory, or re-read it from L2?
3. **Bytes in flight.** Little's law: 3.35 TB/s × ~1 µs of loaded HBM latency ≈ 3 MB in flight across the GPU,
   ~25 KB per SM. Where do those bytes live: registers (128-bit `ld.global`, unrolled), `cp.async`, or TMA into
   shared memory with a ring (the producer/consumer pattern from the GEMMs, now for a memory-bound kernel)?
4. **Writes.** `h` (2 B) and `q` (1 B) per element: are your stores 128-bit and coalesced? `s` is 4 B per 128 elements:
   who writes it?
5. **M = 128.** Fewer rows than SMs. Split a row across a thread-block **cluster** and combine the partial sums of
   squares and group maxima through **distributed shared memory**? Or accept it, since it's L2-resident and ~2 µs?
6. **Scale layout for the consumer.** DeepGEMM wants `SA` M-major and TMA-aligned; this spec asks for row-major (what
   our tilelang GEMM reads). What would writing the transposed layout directly cost?

**Naive unfused tilelang (`naive_unfused.py`, 2026-11-19, `logs/20261119-rmsnorm-naive-unfused-{bench,bw}.log`):** three
streaming kernels with 128-bit loads/stores (all of a thread's loads issued before any math), `y` round-trips in fp32.
Passes the check. 16384 × 7168: **720 µs** (2.0× `torch.compile`): 17 B/element moved instead of 7.

| kernel (16384 × 7168) | bytes/elem | µs | GB/s (own bytes) |
|---|---|---|---|
| k1 add | 6 | 235 | 2995 (89%) |
| k2 norm (bf16 h → fp32 y) | 6 | 299 | 2358 (**70%**) |
| k3 quant (fp32 y → fp8 q + s) | 5 | 196 | 3012 (90%) |

k1 and k3 sit at the copy ceiling; k2 doesn't. Likely cause (unverified): it reads only 2 B/element, so each thread
has 7 × 16 B = 112 B of loads in flight vs 224 B in k1/k3 (Little's law: half the read bytes in flight per thread),
and its block-wide `__syncthreads` reduction sits between the loads and the stores. At M = 128 every kernel is
latency-bound (one CTA per SM, ~3–6 µs each).
