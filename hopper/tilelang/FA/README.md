# Attention prefill forward, bf16 (Hopper, FlashAttention-3 class), tilelang

One attention layer of a Llama-3-8B-style model during prefill: `Hq = 32`, `Hkv = 8` (GQA, 4 query heads per K/V head),
`D = 128`, causal and non-causal. Main shape: causal, `B × N = 4 × 4096`. FLOPs `4·B·Hq·N²·D`, halved for causal:
0.55 TFLOP → **556 µs** at 989 TFLOP/s bf16. Compute-bound (~340 MB of HBM traffic). Full spec, accuracy rules,
roofline and design questions: [`problem_desc.md`](problem_desc.md).

| file | what |
|---|---|
| `problem_desc.md` | problem, interface, accuracy bounds (vs fp64 and a bf16 emulation), roofline, design questions |
| `harness.py` | inputs (normal / peaky / drift, ragged N, MHA), fp64 reference + emulation, CUDA-graph benchmark at 6 configs, baselines (FA3, cuDNN, FA2) |
| `bench.py` | CUDA-graph timing with rotating inputs |
| `torch_ref.py`, `torch_ref_fp8.py` | blockwise online-softmax emulation in torch; checks the harness itself |
| `fa_ws_2consumer.py` | kernel 1: 1 producer + 2 consumer warpgroups, hand-written TMA / mbarriers / wgmma |
| `fa_pingpong.py` | kernel 2: + ping-pong between the consumers (named barriers), QKᵀ and the previous PV in one turn, reverse KV order |
| `fa_pingpong_fastsm.py` | kernel 3: + cheaper softmax (fast-math exp2, shorter row-reduction chains) |

```
python harness.py fa_ws_2consumer      # correctness + benchmark
python harness.py fa_pingpong
python harness.py fa_pingpong_fastsm
python harness.py torch_ref            # harness self-check
```

## Kernel 1: `fa_ws_2consumer.py`

- One CTA per (128 query rows, query head, batch). The producer warpgroup TMA-loads Q once, then K and V blocks
  (128 keys) through a 2-stage ring with separate `full`/`empty` barriers for K and V (QKᵀ can start before V lands;
  K is freed right after QKᵀ).
- Two consumer warpgroups, 64 query rows each, each running the whole FlashAttention loop on its rows: QKᵀ wgmma
  (both operands in smem) → online softmax in registers → PV wgmma with P straight from registers (the QKᵀ
  accumulator layout is the register A-operand layout).
- wgmma doesn't scoreboard its registers: PV_j stays in flight under the next QKᵀ, and the `wait_wgmma(0)` after that
  QKᵀ retires both; only then is O rescaled and P overwritten.
- Causal: blocks above the diagonal are skipped; heavy Q blocks launch first. With `BM = BN = 128` only the last
  block a Q tile visits (the diagonal) needs the mask, so it's peeled into its own call: a runtime `if` that writes
  the S accumulator made ptxas serialize every wgmma in the kernel (C7515). Ragged N: TMA zero-fills OOB keys, which
  score 0, so they are masked to −inf.

**Same run (H100 SXM, CUDA graph, TFLOP/s):**

| B × N | this kernel | FA3 bf16 | cuDNN | FA2 (sdpa) |
|---|---|---|---|---|
| **4 × 4096, causal** | **448** (1228 µs) | 674 (815 µs) | 564 | 318 |
| 4 × 4096, non-causal | 487 | 668 | 599 | 348 |
| 16 × 1024, causal | 202 (as slow as non-causal: not explained yet) | 510 | 480 | 264 |
| 16 × 1024, non-causal | 406 | 641 | 623 | 327 |
| 1 × 16384, causal | 544 | 627 | 592 | 346 |
| 1 × 16384, non-causal | 546 | 663 | 627 | 350 |

All four implementations land at 1.00× the bf16 emulation's error. Roofline (ncu, causal 4 × 4096): the tensor cores
need 586 µs, L2 → SM ~510 µs, instruction issue ~506 µs; measured 1181 µs. Inside a warpgroup QKᵀ → wait → softmax →
PV runs back to back and the two consumers move in step, so the tensor pipe is busy 51% of the time. Next: ping-pong
between the consumers.

## Kernel 2: `fa_pingpong.py`, ping-pong

- Two named barriers (`bar.sync` / `bar.arrive`, ids 1 and 2) pass a "turn" on the tensor cores between the consumers:
  `T.sync_threads(me, 256)` → issue wgmmas only → `T.named_barrier_arrive(other, 256)`. Softmax runs while the other
  warpgroup owns the tensor cores. Named barriers have no phase bit (unlike mbarriers), so the counts must balance:
  each consumer takes n+1 turns, WG1 pre-arms WG0's barrier once and skips its very last arrive.
- One turn issues QKᵀ of block t and PV of block t−1. `wait_wgmma(1)` then returns as soon as S is ready (commit
  groups retire in order), so max/exp/sum overlap this warpgroup's own PV; only `O *= α` and writing P wait for it.
- K/V blocks are visited in reverse (as in FA3): the only block that needs a mask (causal diagonal / ragged tail) is
  the first one, so the loop has a masked prologue and no peeled last iteration.

**Result: 457 TF at causal 4 × 4096, about the same as kernel 1.** ncu: the turn barrier never blocks. Each warpgroup's
softmax (~3000 cycles per block) is ~3× the other's GEMMs (~1000), so softmax overlaps the other softmax, not GEMMs.
Ping-pong only pays once softmax ≲ GEMM time, so the softmax itself had to get cheaper.

## Kernel 3: `fa_pingpong_fastsm.py`, cheaper softmax

Why the softmax was slow (SASS, 679 instructions per block per warp, 10.6 per score; the floor is ~5.5):
- `T.exp2` → `exp2f`, which without fast-math is the denormal-exact sequence `FSETP; @p FMUL; MUFU.EX2; @p FMUL`:
  3 extra instructions per score. Fix: `TL_ENABLE_FAST_MATH` → `--use_fast_math` → a single `MUFU.EX2`.
- `T.reduce_max` / `T.reduce_sum` fold each thread's 32 values per row in one dependent chain (32 × ~4 cycles of
  latency). Fix: fold the 128 columns to 16 partials first (`part[i, j] = max over acc_s[i, j + 16k]`, all in the
  owning thread: 8 independent chains per thread), then reduce the partials + the 4-lane shuffle butterfly. One
  `T.Parallel` may read a buffer with only one index pattern, so this is an inner `T.unroll(8)` loop rather than an
  explicit `max(max(a, b), max(c, d))` tree.

Loop body 676 → 486 SASS instructions per block, whole kernel 8374 → 5736. ncu: tensor pipe 50% → 59%; the softmax is
now mostly waiting on the exp unit (16 exps/clock/SM: half the GEMM time per block at best), which is the part
ping-pong hides.

**Same runs (H100 SXM, CUDA graph, TFLOP/s; FA3 / cuDNN from the kernel 3 run):**

| B × N | kernel 2 | **kernel 3** | FA3 bf16 | cuDNN | FA2 (sdpa) |
|---|---|---|---|---|---|
| **4 × 4096, causal** | 457 (1202 µs) | **539** (1019 µs) | 643 (855 µs) | 587 | 321 |
| 4 × 4096, non-causal | 514 | **572** | 675 | 606 | 348 |
| 16 × 1024, causal | 201 | **304** | 522 | 492 | 267 |
| 16 × 1024, non-causal | 413 | **511** | 640 | 636 | 337 |
| 1 × 16384, causal | 569 | **568** | 627 | 594 | 344 |
| 1 × 16384, non-causal | 560 | **617** | 682 | 633 | 352 |

Correctness unchanged: 1.00× the bf16 emulation's error on every case. Next: overlap a warpgroup's own exps with its
next QKᵀ (FA3's intra-warpgroup pipelining, needs a second S buffer), 3 K/V stages, cheaper epilogue.
