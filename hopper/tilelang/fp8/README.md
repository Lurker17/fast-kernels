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
| `gemm_ws_mcast.py` | kernel 2 + 2-CTA clusters with TMA multicast of A, non-persistent (slower) |
| `gemm_ws_persistent.py` | kernels 3 and 4 (current best): persistent, register prefetch of the scales, 4 stages, TMA-store epilogue; knobs for cluster/multicast and the experiments below |
| `probe_latency.py` | in-kernel `clock64` stamps (`dbg=True`): load latency vs pipeline lead, per k-step |

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

## Kernel 3: `gemm_ws_persistent.py` (~1060 TFLOP/s)

Each row is one change; both numbers are from the same run (interleaved bench, TFLOP/s). Runs differ by up to ±6%
on unlocked clocks, so compare within a row.

| change | before → after |
|---|---|
| 2-CTA clusters + TMA multicast of A, not persistent (`gemm_ws_mcast.py`) | 964 → 838 |
| persistent: 132 CTAs loop over tiles, the stage ring runs across tiles | 943 → 994 |
| persistent + cluster + multicast (`mode="mcast"`) | 994 → 874 (slower at every panel size tried) |
| register prefetch of the scales (`pf=4`) | 976 → 1032 |
| 5 stages → 4 | 1013 → 1062 |

A bigger tile wasn't built: with promotion each consumer needs two fp32 accumulators, 256 registers per thread at
128 rows (cap 232).

**Scale prefetch.** The producer loaded `SA[row, k]` and `SB[n, k]` and multiplied them right away. A load doesn't
stall a thread; the first use of its result does, so every k-group paid a whole global-load latency before `full[s]`
could flip (profile: producer 40% on that multiply, consumers waiting on `full` 22–27%). Each producer thread owns
one row and SA is row-major, so its scales for 4 consecutive k-groups are 16 contiguous bytes: one 128-bit load,
issued a block (4 k-groups) before use. Producer time on scales 40% → 7%, global load sectors 34.6M → 9.2M.
`pf=8` spills under the producer's 40-register cap; loading the scales with TMA was worse per k-group, and only
matched this with its own barrier ring.

**4 stages.** Every k-step each role needs `it % S` and `(it / S) & 1`. With S = 5 that is a real integer division per
step (68 in the generated code). With S = 4 and chunks of 4, `it % 4` is the step's position in the chunk, known at
compile time. The consumers' address/bookkeeping time went 38% → 17%, but most of it moved into waits on `full`
(~20% → 37%): each consumer holds 2 stages (double-buffered partials), so only ~2 k-groups of loads are in flight.
A separate full ncu profile showed equal cycles for S=4 and S=5, so treat this gain as 0–5%.

## Kernel 4: TMA-store epilogue (current best, 1172 TFLOP/s in the harness)

Same file, `tma_epi=True` (now the default). The register epilogue (`T.copy(acc, C)`) did 64 scattered 4-byte
global stores per thread, and the consumer warpgroup couldn't start the next tile's wgmmas until they were issued.
Now each warpgroup converts `acc` to bf16 into its half of a swizzled 2 × 16 KB `C_s`, fences, and one thread issues
the TMA store; the warpgroup waits for that store only before it reuses `C_s` on the next tile.

| same run (interleaved bench) | TFLOP/s |
|---|---|
| kernel 3 (register epilogue) | 1060 |
| **kernel 4 (TMA-store epilogue)** | **1155** (+9%) |

Profile: cycles 1.564 M → 1.414 M, tensor pipe busy 67% → 75%, consumer time in the epilogue 4–8% → 1%.

**Baselines, finally like-for-like.** DeepGEMM builds on an Ubuntu 24.04 / CUDA 12.9 image (it needs g++ 13).
Harness, same run (CUDA graph, 4 rotating input sets, median):

| | ms | TFLOP/s |
|---|---|---|
| DeepGEMM (`fp8_fp4_gemm_nt`, row-major SA as given) | 0.858 | 1281 |
| **this kernel** | **0.938** | **1172** (91.5% of DeepGEMM) |
| cuBLAS fp8, per-tensor scale (speed only) | 0.873 | 1260 |
| fp8 rowwise `_scaled_mm` (speed only) | 1.020 | 1078 |

**The accuracy check: it's the bound, not the kernel.** DeepGEMM fails `uniform_pos` identically: the same 117,189
elements over the bound, the same worst element (out 2048, ref 2057.24, err/bound 1.02), and the same err/bound on the
other two distributions. The miss comes from the fp8 wgmma's limited-precision internal accumulation on
all-positive sums. The bound's accumulation term (2⁻¹¹·|A|·|B|) is too strict for any kernel that promotes every 128 of K.

**Tried on top, no gain (all knobs in the file):**

| knob | idea | result |
|---|---|---|
| `inc` | carry (stage, phase) across steps instead of `it % S`, `it / S`, to make 5 stages cheap | correct but slower (S=5: 998 → 973). nvcc already turns `% 5` into a cheap multiply; S=4 wins because its stage indices are compile-time constants |
| `serp` | serpentine panel order | noise (±1%) |
| `early` | after `wait_wgmma(1)`, read the 2 scale floats into registers, release the stage, then promote | 1155 → 1157 |
| `mode="mcast"` | 2-CTA multicast of A, retested now that the epilogue is async | ties at panel 16 (1132 vs 1129); it used to lose 5–10% |

**Why: measured, not inferred** (`probe_latency.py`). With `dbg=True` the kernel also returns `clock64` stamps per k-step:
when the producer issued the TMA, when a consumer started waiting on `full`, and when `full` fired. Medians, cycles:

| | 4 stages | 5 stages |
|---|---|---|
| k-step | 733 | 753 |
| lead (consumer needs it − TMA issued) | 1252 | 2093 |
| latency (TMA issued → `full` fires) | 1540 | 2350 |
| consumer stall per step | **226** | **233** |

One more stage issued every load 840 cycles earlier, and the data arrived at the same moment. So the consumers'
~30% wait isn't a latency that more stages can hide. Each SM receives its 32 KB per k-step at a fixed rate (~44 B/clk;
× 132 SMs matches ncu's L2 → SM traffic), and extra stages just queue. Multicast cuts L2 reads, but every SM still
ingests the same 32 KB per step. The next lever is fewer bytes per FLOP per SM: a bigger CTA tile (BM = 192 with
three consumer warpgroups, or BN = 256 with a promotion scheme that fits in 232 registers).
