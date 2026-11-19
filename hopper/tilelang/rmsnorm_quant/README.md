# Fused residual add + RMSNorm + per-token-group FP8 quant (Hopper), tilelang

`h = bf16(x + residual)`, `y = h · rsqrt(mean(h²) + eps) · weight` (fp32), then 1×128 per-token-group fp8 quant of `y`
into `q` (fp8 e4m3) and `s` (fp32 scales, `[M, H/128]`). `q` and `s` are exactly the `A` / `SA` operands of the fp8
GEMM in [`../fp8`](../fp8). H = 7168 (DeepSeek-V3), main shape M = 16384. Memory-bound: 7 bytes of HBM traffic per
element, so the roofline is bandwidth. Full spec, accuracy rules and design questions: [`problem_desc.md`](problem_desc.md).

| file | what |
|---|---|
| `problem_desc.md` | problem, accuracy rules, roofline, baselines, design questions |
| `harness.py` | inputs (normal / outlier channels / extreme row scales + all-zero groups), fp64 reference, checks, CUDA-graph benchmark at 3 shapes |
| `bench.py` | CUDA-graph timing with rotating inputs (cold L2) |
| `torch_ref.py` | the spec in plain torch; checks the harness itself |
| `naive_unfused.py` | three separate vectorized tilelang kernels (add, norm, quant), `y` round-trips through HBM in fp32 |

```
python harness.py naive_unfused      # correctness + benchmark
python harness.py torch_ref          # harness self-check
python naive_unfused.py              # bandwidth of each of the three kernels
```

## Baselines (H100 SXM, harness, median per call)

| M × H | floor at 3.35 TB/s | torch eager | torch.compile | `x + residual` probe |
|---|---|---|---|---|
| 16384 × 7168 | 246.5 µs | 4450 µs | **365 µs** (68% of HBM) | 91% of HBM (own bytes) |
| 2048 × 7168 | 30.8 µs | 602 µs | 50 µs | 81% |
| 128 × 7168 | 1.9 µs | 69 µs | 6.8 µs | 50% |

The probe says ~3.0 TB/s is reachable, so about **275 µs** at the main shape. `torch.compile`'s scales miss the 2⁻¹⁶
tolerance (up to 2.5e-3 relative), so it counts as a speed target only.

## Naive unfused (`naive_unfused.py`): 720 µs

One CTA per row, 128 threads, 128-bit loads and stores, and each thread issues all its loads for the row before any
math. Passes the checks. It moves 17 bytes per element instead of 7:

| kernel (16384 × 7168) | bytes/elem | µs | GB/s (own bytes) |
|---|---|---|---|
| add | 6 | 235 | 2995 (89%) |
| norm (bf16 `h` → fp32 `y`) | 6 | 299 | 2358 (70%) |
| quant (fp32 `y` → fp8 `q` + `s`) | 5 | 196 | 3012 (90%) |
| all three | 17 | 720 | 1147 effective (problem bytes) |

Add and quant reach the copy ceiling. Norm doesn't; a likely reason (not verified yet) is that it reads only 2 bytes
per element, so each thread has half as many load bytes in flight as the other two, with a block-wide reduction
between its loads and stores. At M = 128 everything is latency-bound: one CTA per SM.

tilelang gotcha: `T.empty` with an fp8 dtype fails ("Unsupported code 10": tilelang allocates outputs through torch's
DLPack bridge, which doesn't support fp8). Allocate the fp8 output in torch and pass it in as a tensor the kernel writes.
