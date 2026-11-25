# Attention prefill forward, bf16 (Hopper, FlashAttention-3 class)

## The problem

One attention layer of a Llama-3-8B-style model during **prefill**. A batch of `B` prompts, each `N` tokens long,
has already been projected to Q, K and V. Compute the attention output for every token.

| parameter | value | meaning |
|---|---|---|
| `Hq` | 32 | query heads |
| `Hkv` | 8 | key/value heads (**GQA**, group size `G = Hq / Hkv = 4`) |
| `D` | 128 | head dim |
| `scale` | `1/sqrt(128)` | softmax scale |
| `B × N` | 16 × 1024, 4 × 4096, 1 × 16384 | 16k tokens per call; all prompts in a batch have the same length |
| `causal` | yes / no | see below |

**GQA: yes.** Every modern LLM uses it (Llama 3, Qwen, Mistral). Query head `h` attends with KV head `h // G`, so 4
query heads share one K/V. To the kernel it's an index mapping (plus a chance to reuse K/V across those 4 heads). MHA
is the special case `G = 1`, and one correctness case runs it.

**Causal vs non-causal.** LLM prefill is **causal**: token `i` attends to tokens `0..i`. Non-causal (every token sees
every token, as in an encoder) is the same kernel without the mask and with uniform work per Q tile. It's the
simpler first milestone and the standard FlashAttention benchmark. Both are measured. **Causal 4 × 4096 is the main
number.**

Out of scope: paged KV, variable-length batches, decode, `Nq ≠ Nkv`, dropout, sliding windows, the backward pass, and
fp8 (stage 2, below).

## Math

For each batch `b`, query head `h`, query position `i` (`hk = h // G`):

```
s_j  = scale · Σ_d Q[b, i, h, d] · K[b, j, hk, d]          for j in 0..N-1   (causal: only j ≤ i)
LSE  = log Σ_j exp(s_j)                                      natural log
O[b, i, h, :] = Σ_j exp(s_j − LSE) · V[b, j, hk, :]
```

The `N × N` score matrix is never written to memory. The kernel streams K/V blocks past a Q tile that stays on chip,
and keeps a running max and sum per row (online softmax).

## Interface

The impl module defines `PRECISION = "bf16"` and `make()`, which returns:

```python
fn(q, k, v, causal: bool) -> (o, lse)
```

| tensor | shape | dtype | layout |
|---|---|---|---|
| `q` (in) | `[B, N, Hq, D]` | bf16 | contiguous (FlashAttention's `bshd`) |
| `k`, `v` (in) | `[B, N, Hkv, D]` | bf16 | contiguous |
| `o` (out) | `[B, N, Hq, D]` | bf16 | contiguous, freshly allocated |
| `lse` (out) | `[B, Hq, N]` | fp32 | contiguous, freshly allocated (the backward and split-KV need it) |

`causal` is a Python bool: compile one kernel per value and cache it. Don't modify the inputs. `N` doesn't have to be a
multiple of your tile size: one correctness case has `N = 1000`.

## Correctness

The reference is fp64, computed from the bf16 inputs. The harness also builds an **emulation** of a careful bf16
kernel: fp32 scores, exact row max, fp32 `exp` and row sum, `p` rounded to bf16 before `P·V` (a real kernel has to do
this, because the PV wgmma takes bf16), and O rounded to bf16. The bounds are:

- **O:** `max|O − O_ref| ≤ 2·max|O_emu − O_ref| + 2⁻¹⁴` and `mean|O − O_ref| ≤ 2·mean|O_emu − O_ref| + 2⁻¹⁶`.
  The factor 2 covers your running max rounding `p` at different points than the exact max does. A missing rescale,
  wrong mask or wrong GQA head is off by orders of magnitude.
- **LSE:** `|LSE − LSE_ref| ≤ 2⁻¹⁰ · max(1, |LSE_ref|)`.
- No NaN/inf, bitwise deterministic across two calls, inputs unchanged.

| case (`harness.CASES`) | what it catches |
|---|---|
| `normal`: q, k, v ~ N(0, 1) | the basics; logits ~ N(0, 1), flat softmax |
| `peaky`: q, k × 3 | logits std ~9: a few keys dominate each row (max subtraction, `l` bookkeeping) |
| `drift`: a shared direction added to queries and, growing with position, to keys | later keys score up to ~20 higher, so the running max grows at **every** block (every rescale exercised) |
| `N = 1000` | ragged last block: zero-filled OOB keys score 0, not −inf, so they still need masking |
| MHA (`Hkv = Hq`), `N = 128` (one block) | GQA mapping, single-block edge case |
| the 4 × 4096 bench shapes | the real thing |

## Roofline (H100 SXM)

FLOPs (FlashAttention convention): `4·B·Hq·N²·D`, halved for causal. At 989 TFLOP/s dense bf16:

| B × N | non-causal FLOPs | floor | causal FLOPs | floor |
|---|---|---|---|---|
| 16 × 1024 | 0.27 T | 278 µs | 0.14 T | 139 µs |
| **4 × 4096** | 1.10 T | 1.11 ms | **0.55 T** | **556 µs** |
| 1 × 16384 | 4.40 T | 4.45 ms | 2.20 T | 2.22 ms |

It's compute-bound: about 340 MB of HBM traffic at 4 × 4096 (~100 µs). There are two compute ceilings:

- **Tensor cores:** 512 FLOPs per score (`2D` for `QKᵀ`, `2D` for `PV`).
- **The exp unit:** 1 `exp2` per score, at 16/clk/SM ≈ **3.9 T/s** across the GPU. Running at tensor-core peak needs
  989/512 ≈ 1.9 T exp/s, about 50% of it. Add the FP32 work per score (max, FMA, row sum, convert, rescale) and the
  softmax is a large fraction of a GEMM's time. Unless it overlaps the wgmmas, you lose that time outright.

Targets: FA3 reports up to ~740 TF (75%) on H100 for bf16, and cuDNN is in the same range. FA3 parity on non-causal,
then causal, is the goal.

## Benchmark

CUDA graph, 2 rotating input sets, median µs per call. Reported as µs, TFLOP/s and % of peak for all 6 configurations.
Baselines in the same run (each is optional: if one fails to load, it's printed and skipped):

| baseline | what |
|---|---|
| `FA3 bf16` | FlashAttention-3 (prebuilt, HF `kernels` hub). **The target** |
| `sdpa cudnn` | `F.scaled_dot_product_attention`, cuDNN backend |
| `sdpa flash` | SDPA's FlashAttention-2 backend (pre-Hopper design: the "FA2" number) |

`torch_ref.py` is a block-by-block online-softmax emulation in torch, used as the harness smoke test (not a baseline).

**Measured (2026-11-24, H100 SXM, `logs/20261124-112725-fa-ws2c-bench-fa3.log`), TFLOP/s (fp8 column: % of fp8 peak):**

| B × N, causal | `fa_ws_2consumer` | FA3 bf16 | sdpa cudnn | FA2 | FA3 fp8 |
|---|---|---|---|---|---|
| **4 × 4096, causal** | **448 (1228 µs)** | **674 (815 µs)** | 564 | 318 | 865 (44%) |
| 4 × 4096, non-causal | 487 | 668 | 599 | 348 | 868 |
| 16 × 1024, causal | 202 (679 µs ≈ non-causal: unexplained) | 510 | 480 | 264 | 549 |
| 16 × 1024, non-causal | 406 | 641 | 623 | 327 | 744 |
| 1 × 16384, causal | 544 | 627 | 592 | 346 | 909 |
| 1 × 16384, non-causal | 546 | 663 | 627 | 350 | 905 |

FA3 comes from the HF kernels hub (`kernels-community/flash-attn3`, `revision="main"`). It returns only O, so its LSE isn't
checked. FA3, cuDNN, FA2 and our kernel all land at 1.00× the emulation's O error: the bounds hold. Run-to-run noise is
about ±5% (cuDNN causal measured 881 µs in the previous run, 974 µs here).

### Roofline of `fa_ws_2consumer`, causal 4 × 4096 (ncu, SM clock 1.79 GHz, `profiles/hopper_flash_attn/`)

Each row is one hardware resource: the work the kernel gives it, and how long that work takes at the resource's peak.
If everything overlapped perfectly, the kernel time would be the largest row.

| resource | work | peak | time at peak |
|---|---|---|---|
| **tensor (HGMMA)** | 0.567 TFLOP executed (diagonal blocks are computed in full: 528 of 1024 blocks, not 512) | 967 TF at 1.79 GHz | **586 µs** |
| **L2 → SM** | 4.56 GB (every Q tile re-reads K/V) | ~8.9 TB/s (42.8% busy over 1.19 ms) | **~510 µs** |
| **issue slots** | 478 M warp instructions (~14 thread-instructions per score) | 4 per clk per SM | **506 µs** |
| ALU (max, convert, compares, int) | ncu: pipe 25.6% busy | | 302 µs |
| exp unit (XU) | 1.11 G `ex2` → 293 µs by count. ncu: XU 30% busy | 16 per clk per SM | 355 µs |
| FMA (scale/sub, sum, rescale) | ncu: pipe 20.2% busy | | 238 µs |
| HBM | 327 MB | 3.35 TB/s | 98 µs |
| **measured** | | | **1181 µs (ncu) / 1228 µs (bench)**. FA3: 815 µs |

What this says:
- **The floor is the tensor cores, at 586 µs.** We take 2.0× that. The gap is serialization: within a warpgroup,
  QKᵀ → wait → softmax → PV runs back to back, so the tensor pipe is busy only 51% of the time. Ping-pong and
  intra-warpgroup overlap are the fix (FA3 is at 72% of this floor).
- **L2 is a near-ceiling.** A K/V block is reused by the `BM` query rows of the CTA. That's `4·BM·BN·D` FLOPs per
  `2·BN·D·2` bytes = **BM FLOPs per L2 byte = 128**. Tensor peak would need 967/128 = 7.6 TB/s from L2, about 85% of
  its peak. Levers: put several GQA heads of one KV head in a CTA (4 heads share K/V), or TMA-multicast K/V across a
  cluster of 2 CTAs, as in the GEMM's v8.
- **Issue slots are a near-ceiling too.** Each SMSP issues one instruction per clock, and a warp of WG0, a warp of WG1
  and a producer warp share it. The softmax instruction stream (~14 instructions per score element per thread) needs
  86% of the tensor floor's time just to issue. Fewer instructions per element (folded FMAs, packed ops, a mask only
  on the diagonal, the producer's spin loop) count directly.
- exp, ALU and FMA are each about 40–60% of the tensor floor. They're free only if they overlap the wgmmas.

## Things to figure out (questions, not answers)

1. **Tiles.** A CTA owns `BM` queries of one head and loops over K/V in `BN`-key blocks. Each consumer warpgroup holds
   an fp32 `S` tile and the fp32 `O` accumulator (64 regs/thread at D = 128) in registers. What `BM`/`BN` fit?
   How many K/V smem stages?
2. **Online softmax.** Running `m`, `l`. Rescale `O` when `m` grows. Fold `scale·log₂e` into one FMA before `exp2`.
   The row max/sum reduce across the 4 threads that share a row in the wgmma accumulator layout.
3. **P stays in registers.** The `QKᵀ` accumulator fragment has the same layout as the register A-operand of the `PV`
   wgmma. Does that hold in tilelang's lowering?
4. **Overlap.** Ping-pong (two consumer warpgroups take turns doing softmax and GEMM, ordered by named barriers), and
   pipelining within one warpgroup (issue `QKᵀ` of block j+1 before the softmax of block j).
5. **Producer.** TMA loads Q once, then K and V through a ring. K is released after `QKᵀ` and V after `PV`, so do they
   need separate barriers?
6. **GQA.** Launch the 4 query heads of a KV head together so K/V hit L2, or put all 4 heads in one CTA's M dimension?
7. **Causal.** Skip blocks above the diagonal, and mask only on diagonal blocks. Q tiles at the end do N/BN blocks while
   those at the start do 1: schedule the long ones first, or go persistent.
8. **Epilogue.** `O / l` → bf16 → smem → TMA store. `LSE = m + log l` (mind log₂ vs ln).

Suggested path: non-causal with 1 consumer warpgroup → producer + 2 consumers → ping-pong → intra-warpgroup overlap →
causal. Compare against tilelang's own auto-pipelined version,
`tilelang/examples/flash_attention/example_gqa_fwd_bshd.py`.

## Stage 2: fp8 (later, same harness)

FA3's e4m3 path. The impl sets `PRECISION = "fp8"`, and the signature becomes
`fn(q8, k8, v8, q_descale, k_descale, v_descale, causal)` with one fp32 descale per `[B, Hkv]`. The harness quantizes,
then checks against the dequantized inputs with `p` rounded to e4m3 in the emulation; `torch_ref_fp8.py` is the smoke
test. The new problems: fp8 wgmma needs V K-major (transpose in kernel), P needs a byte permute in registers, and at
D = 128 the exp unit becomes a co-bottleneck (fp8 peak needs ~100% of it).

## Later: paged attention / decode (separate problem)

1–8 new tokens per sequence against a long KV cache stored in pages. Memory-bound, split along KV (flash-decoding),
gather loads. It reuses this problem's LSE for the split combine.
