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

```
python harness.py fa_ws_2consumer      # correctness + benchmark
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
