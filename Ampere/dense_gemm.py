"""
tensorop_gemm_replication.py — harness for a hand-written Ampere fp16 tensor-core GEMM.

    C[M,N] (fp16) = A[M,K] (fp16) @ B[N,K]^T (fp16),  fp32 accumulate.

Everything that is NOT the kernel is written for you: config validation, tensor setup,
streams, a correctness suite, a benchmark against cuBLAS, and an autotune driver.
The kernel is Section 2, and it is yours.

Target: docs/bundle/03-examples/cutedsl-python/cute/ampere/kernel/dense_gemm/tensorop_gemm.py
(215.8 TFLOP/s at 4096^3 on this A100 — about equal to cuBLAS at 215.9).

SCOPE, deliberately narrowed against the reference so you fight one thing at a time:
  * fp16 in, fp32 accumulate, fp16 out. No fp8/bf16 branch forest.
  * A is (M,K) K-major, B is (N,K) K-major, C is (M,N) N-major. No majorness matrix,
    no C^T = B^T A^T transpose trick.
  * No batch dimension L.
Each of those is a later rung, not a missing feature. Add them once the number is there.


THE LADDER  (measured start: 01_warptiled_mma.py = 99 TF; 01_01 = 128 TF; target 215 TF)

  1. m16n8k16 instead of m16n8k8          verify: smsp__inst_executed_op_hmma.sum halves
  2. register double-buffer the k_block   verify: stall_short_scoreboard drops
     loop (load k+1 before mma k)                tensorop_gemm.py:702-753
  3. STAGES 3 -> 4                        verify: stall_long_scoreboard drops
  4. permutation_mnk widening N           verify: make_tiled_copy_B tiler 16 -> 32
  5. smem epilogue (acc -> smem -> gmem)  verify: store sectors drop; matters most at small K
  6. threadblock rasterization            verify: L2 hit rate
  7. predication (ragged shapes)          verify: the RAGGED shapes below start passing

Do them one at a time and record the number after each. Optimizations interact: last time
double-buffering was net-negative while the kernel was still bank-conflict-bound.

USAGE
    .venv/bin/python tensorop_gemm_replication.py --test
    .venv/bin/python tensorop_gemm_replication.py --bench
    .venv/bin/python tensorop_gemm_replication.py --tune
    .venv/bin/python tensorop_gemm_replication.py --bench --mnk 4096,4096,4096

Confirm any shape you are unsure about in scratchpad.ipynb with ask() before you write it.
"""

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import JitArguments, benchmark

# Fixed for this file. Widening these is rung 8+.
AB_DTYPE = cutlass.Float16
C_DTYPE = cutlass.Float16
ACC_DTYPE = cutlass.Float32
TORCH_AB = torch.float16
TORCH_C = torch.float16

# sm_80 shared memory per block: 48 KiB is free, above that needs the dynamic opt-in,
# and 163840 B is the hard ceiling.
SMEM_DEFAULT_LIMIT = 48 * 1024
SMEM_OPTIN_LIMIT = 163840


# ============================================================================
# SECTION 1 — Config.  WRITTEN FOR YOU.
# ============================================================================
# The single most important structural difference between the reference kernel and
# 01_warptiled_mma.py: every layout constant is DERIVED from the fields below, and
# nothing downstream types a number twice. That is what makes --tune possible at all;
# a kernel with three hardcoded copies of BK cannot be swept.
#
# Add derived properties here as you need them. Keep the invariants as asserts, so a
# bad sweep point fails at construction instead of producing a silently wrong kernel.


@dataclass(frozen=True)
class Config:
    # --- the knobs a sweep turns ---
    BM: int = 128
    BN: int = 128
    BK: int = 32
    STAGES: int = 4
    WARPS: Tuple[int, int, int] = (2, 2, 1)          # warp grid over (M, N, K)
    MMA: Tuple[int, int, int] = (16, 8, 16)          # mma.sync instruction shape
    COPY_BITS: int = 128                             # cp.async width, bits per thread
    # Set once you reach rung 4. An int means the identity layout `N:1`, which only
    # resizes the per-step tile; a cute.Layout also reorders ownership. See the
    # "permutation_mnk" cell of scratchpad.ipynb.
    PERM_N_FACTOR: int = 1

    # --- derived: never type these anywhere else ---
    @property
    def dtype_bits(self) -> int:
        return AB_DTYPE.width

    @property
    def num_threads(self) -> int:
        return self.WARPS[0] * self.WARPS[1] * self.WARPS[2] * 32

    @property
    def copy_elems(self) -> int:
        """Elements moved by one cp.async instruction, per thread."""
        return self.COPY_BITS // self.dtype_bits

    @property
    def g2s_thr_shape(self) -> Tuple[int, int]:
        """(rows, cols) thread grid for the gmem->smem copy, K-major.

        cols threads span one BK row; the rest stack along M. Consecutive threads read
        consecutive gmem, which is what makes the load coalesced.
        """
        cols = self.BK // self.copy_elems
        return (self.num_threads // cols, cols)

    @property
    def major_mode_size(self) -> int:
        """Contiguous extent of the smem layout atom, capped at one 128 B line."""
        return min(self.BK, 128 * 8 // self.dtype_bits)

    @property
    def permutation_mnk(self) -> Optional[Tuple[int, int, int]]:
        if self.PERM_N_FACTOR == 1:
            return None
        return (
            self.WARPS[0] * self.MMA[0],
            self.WARPS[1] * self.MMA[1] * self.PERM_N_FACTOR,
            self.WARPS[2] * self.MMA[2],
        )

    @property
    def smem_ab_bytes(self) -> int:
        return 2 * (self.BM * self.BK) * self.STAGES * self.dtype_bits // 8

    @property
    def smem_c_bytes(self) -> int:
        """Rung 5: the epilogue reuses the A/B buffers rather than adding to them."""
        return self.BM * self.BN * C_DTYPE.width // 8

    @property
    def smem_bytes(self) -> int:
        return self.smem_ab_bytes

    def validate(self) -> None:
        mM, mN, mK = self.MMA
        wM, wN, wK = self.WARPS
        assert wK == 1, "atom layout K > 1 is not supported here"
        assert self.BM % (wM * mM) == 0, f"BM={self.BM} not divisible by warpM*mmaM={wM*mM}"
        assert self.BN % (wN * mN) == 0, f"BN={self.BN} not divisible by warpN*mmaN={wN*mN}"
        assert self.BK % mK == 0, f"BK={self.BK} not divisible by mmaK={mK}"
        assert self.STAGES >= 3, "the mainloop overlap needs at least 3 stages"
        assert self.BK % self.copy_elems == 0, (
            f"BK={self.BK} must be a multiple of copy_elems={self.copy_elems}")
        rows, cols = self.g2s_thr_shape
        assert rows * cols == self.num_threads, (
            f"g2s thread grid {rows}x{cols} != {self.num_threads} threads")
        assert self.BM % rows == 0 and self.BN % rows == 0, (
            f"g2s row count {rows} must divide BM and BN")
        assert self.smem_bytes <= SMEM_OPTIN_LIMIT, (
            f"smem {self.smem_bytes} B exceeds the sm_80 ceiling {SMEM_OPTIN_LIMIT} B")

    def describe(self) -> str:
        rows, cols = self.g2s_thr_shape
        over = " (needs dynamic smem opt-in)" if self.smem_bytes > SMEM_DEFAULT_LIMIT else ""
        return (
            f"tile {self.BM}x{self.BN}x{self.BK}  stages={self.STAGES}  "
            f"warps={self.WARPS}  mma={self.MMA}\n"
            f"    threads={self.num_threads}  copy_elems={self.copy_elems}  "
            f"g2s_thr=({rows},{cols})  perm={self.permutation_mnk}\n"
            f"    smem A+B = {self.smem_bytes/1024:.1f} KiB{over}"
        )


# ============================================================================
# SECTION 2 — THE KERNEL.  YOURS.
# ============================================================================
# Fill these in, in ladder order. Each stub says what it must return and how to check
# the answer. Reference line numbers point into
#   docs/bundle/03-examples/cutedsl-python/cute/ampere/kernel/dense_gemm/tensorop_gemm.py
#
# Confirm every shape with ask() in scratchpad.ipynb before you trust it. Remember the
# AST preprocessor rejects `continue` and `break` inside traced functions.


# ---- structure builders (plain host functions, called at trace time) --------
# Each takes cfg and returns one piece of static structure. Keeping them separate is
# the whole point: a builder is a small, testable answer to one question, and you can
# call any of them straight from scratchpad.ipynb to look at what it produced.


def make_smem_layout_AB(cfg: Config, mn: int):
    row_elems = cfg.major_mode_size                       # contiguous row, capped at 128 B

    # Swizzle in ELEMENT units (we compose into the atom; the reference keeps
    # byte units because it applies the swizzle separately at get_tensor()).
    chunk_elems = cfg.copy_elems                          # 16 B cp.async chunk = 8 fp16
    B = min(int(math.log2(row_elems // chunk_elems)), 3)  # chunks per row beyond the first
    M = int(math.log2(chunk_elems))                       # = 3: never split a chunk
    S = int(math.log2(chunk_elems))                       # = 3: gap up to the row bits
    assert B >= 0, f"row {row_elems} elems is narrower than one copy chunk"

    atom = cute.make_composed_layout(
        cute.make_swizzle(B, M, S), 0,
        cute.make_layout((8, row_elems), stride=(row_elems, 1)))
    layout = cute.tile_to_shape(atom, (mn, cfg.BK, cfg.STAGES), (0, 1, 2))

    assert cute.size_in_bytes(AB_DTYPE, layout) == cfg.smem_ab_bytes // 2
    return layout


def make_gmem_tiled_copy_AB(cfg: Config):
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), AB_DTYPE,
                               num_bits_per_copy=cfg.COPY_BITS)
    rows, cols = cfg.g2s_thr_shape                       # (32, 4)
    thr = cute.make_layout((rows, cols), stride=(cols, 1))   # tid = 4*row + col
    val = cute.make_layout((1, cfg.copy_elems))              # 8 elems along K
    return cute.make_tiled_copy_tv(atom, thr, val)


def make_tiled_mma(cfg: Config):
    """Return the TiledMMA: MmaF16BF16Op(cfg.MMA) over cute.make_layout(cfg.WARPS),
    with permutation_mnk=cfg.permutation_mnk.

    Reference: tensorop_gemm.py:265-290. Rung 4 is where PERM_N_FACTOR becomes 2.

    Check: partition_C(gC).shape is (MMA, MMA_M, MMA_N).
    """
    op = warp.MmaF16BF16Op(AB_DTYPE, ACC_DTYPE, cfg.MMA)
    return cute.make_tiled_mma(op,
                               cute.make_layout(cfg.WARPS),
                               permutation_mnk=cfg.permutation_mnk)

def make_s2r_tiled_copies(cfg: Config, tiled_mma):
    """Return (s2r_A, s2r_B): the ldmatrix copies feeding the mma operands.

    make_tiled_copy_A/B compose the ldmatrix atom with the TiledMMA's fragment layouts,
    so the copy's destination IS the mma's source. Reference: tensorop_gemm.py:624-641.

    Check: the copyB tiler widens from (16,16) to (32,16) once PERM_N_FACTOR is 2 —
    that is the whole observable effect of the permutation, and the reason rung 4 exists.
    """
    ldsm = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), AB_DTYPE) # x4 copy atom. 4 8*8 matrices
    return cute.make_tiled_copy_A(ldsm, tiled_mma), cute.make_tiled_copy_B(ldsm, tiled_mma)


def make_smem_layout_C(cfg: Config):
    """Rung 5 only. Staging layout for the epilogue, (BM, BN), no stages.

    The point is turning scattered per-thread accumulator stores into 128-bit coalesced
    gmem stores. Reference: tensorop_gemm.py:845-876. Allocate it as a union with the
    A/B buffers (max of the two) rather than in addition — tensorop_gemm.py:449-460.
    """
    raise NotImplementedError("rung 5")


# ---- device kernel ----------------------------------------------------------


@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    g2s_A: cute.TiledCopy,
    g2s_B: cute.TiledCopy,
    s2r_A: cute.TiledCopy,
    s2r_B: cute.TiledCopy,
    sA_layout,
    sB_layout,
    cta_tiler: cutlass.Constexpr,
):
    """The mainloop. Reference: tensorop_gemm.py:332-819.

    Every parameter above except the tensors is a compile-time constant. The signature
    is long on purpose: it is the complete list of what this kernel depends on. Add to
    it as you climb the ladder (sC_layout and a tiled_copy_C at rung 5, a raster factor
    at rung 6, predicate tensors at rung 7).

    Shape of the thing you are building:
      1. local_tile gA/gB/gC for this CTA
      2. allocate smem, partition it for the g2s copy and for the mma
      3. prologue: fill STAGES-1 buffers, commit each as its own cp.async group
      4. mainloop over k_tiles:
           - at the last k_block, wait cp_async_wait_group(STAGES-2) + barrier
           - issue the s2r copy for k_block+1 BEFORE the mma for k_block   [rung 2]
           - at k_block 0, issue the gmem fetch for the next tile
           - cute.gemm for k_block
      5. epilogue: accumulators -> gmem (rung 5 routes this via smem)

    The register pipeline in step 4 is the point of rung 2: k and k+1 live in different
    registers, so the mma for k does not wait on the load for k+1. Read the comment at
    tensorop_gemm.py:677-701 first — it explains why you wait on STAGES-2 rather than
    fully draining a buffer.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    ## global layouts for A, B, C
    BM, BN, BK = cta_tiler
    gA = cute.local_tile(mA, (BM, BK), (bidx, None))     # (BM, BK, k_tiles)   k_tiles = K/BK
    gB = cute.local_tile(mB, (BN, BK), (bidy, None))     # (BN, BK, k_tiles)
    gC = cute.local_tile(mC, (BM, BN), (bidx, bidy))     # (BM, BN)

    # input is 16B aligned
    gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
    gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

    smem = utils.SmemAllocator()
    sA = smem.allocate_tensor(AB_DTYPE, sA_layout, byte_alignment=16) # (BM, BK, STAGES)  swizzled
    sB = smem.allocate_tensor(AB_DTYPE, sB_layout, byte_alignment=16) # (BN, BK, STAGES)  swizzled
    thr_copy_A = g2s_A.get_slice(tidx)
    thr_copy_B = g2s_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)                    # (CPY, BM/32, BK/32, k_tiles) = ((8,1), 4, 1, k_tiles)
    tAsA = thr_copy_A.partition_D(sA)                    # (CPY, BM/32, BK/32, STAGES)  = ((8,1), 4, 1, STAGES)
    tBgB = thr_copy_B.partition_S(gB)                    # (CPY, BN/32, BK/32, k_tiles) = ((8,1), 4, 1, k_tiles)
    tBsB = thr_copy_B.partition_D(sB)                    # (CPY, BN/32, BK/32, STAGES)  = ((8,1), 4, 1, STAGES)

    ## Block 1 — mma's view of the output. acc lives in registers for the whole
    ## kernel; its shape ((MMA), MMA_M, MMA_N) IS the warp-tile repeat structure.
    thr_mma = tiled_mma.get_slice(tidx)
    tCgC = thr_mma.partition_C(gC)                       # (MMA_C, BM/(wM*16), BN/(wN*8)) = ((2,2), 4, 8)
    acc = cute.make_fragment(tCgC.shape, ACC_DTYPE)      # same profile as tCgC, fp32: 128 regs/thread
    acc.fill(0.0)

    ## Block 2 — A/B register fragments for ONE stage. sA[None, None, 0] slices
    ## stage 0's (BM, BK) slab: fragments sized for a single stage, not STAGES of
    ## them. make_fragment_B rejects a raw shape — it needs the partitioned tensor.
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0])) # (MMA_A, BM/(wM*16), BK/mK) = ((2,2,2), 4, 2)
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0])) # (MMA_B, BN/(wN*8),  BK/mK) = ((2,2), 8, 2)

    ## Block 3 — s2r plumbing. tXsA partitions ALL stages (read_stage is indexed in
    ## the mainloop); tXrA allocates nothing — it is tCrA's registers re-viewed in
    ## copy shape so both sides of cute.copy agree on the profile.
    thr_s2r_A = s2r_A.get_slice(tidx)
    tXsA = thr_s2r_A.partition_S(sA)                     # (CPY, BM/(wM*16), BK/mK, STAGES) = ((8,1), 4, 2, STAGES)
    tXrA = thr_s2r_A.retile(tCrA)                        # (CPY, BM/(wM*16), BK/mK)         = ((8,1), 4, 2)  = tCrA's regs
    thr_s2r_B = s2r_B.get_slice(tidx)
    tXsB = thr_s2r_B.partition_S(sB)                     # (CPY, BN/(wN*8), BK/mK, STAGES)  = ((8,1), 8, 2, STAGES)
    tXrB = thr_s2r_B.retile(tCrB)                        # (CPY, BN/(wN*8), BK/mK)          = ((8,1), 8, 2)  = tCrB's regs

    num_k_tiles = cute.size(gA, mode=[2])          # runtime, from the tensor
    K_BLOCKS    = cute.size(tCrA, mode=[2])        # = BK/mK = 2, compile-time
    STAGES      = cute.size(tAsA, mode=[3])        # from the smem tensor, compile-time

    ## MILESTONE 2 — cp.async pipeline. STAGES buffers, STAGES-1 kept in flight.
    ##
    ## Invariant at the top of iteration k_tile:
    ##   - groups for tiles k_tile .. k_tile+STAGES-2 have been committed
    ##   - wait_group(STAGES-2) => at most STAGES-2 groups still in flight
    ##     => the group for k_tile has landed => read_stage is readable.
    ## The barrier after the wait is what makes it safe to LATER overwrite stage
    ## (k_tile-1) % STAGES: once every thread passed it, every thread has finished
    ## its ldmatrix reads from iteration k_tile-1.

    # Prologue: fill the first STAGES-1 buffers, one commit group per tile so
    # wait_group can count them individually.
    for s in cutlass.range_constexpr(STAGES - 1):
        cute.copy(g2s_A, tAgA[None, None, None, s], tAsA[None, None, None, s])
        cute.copy(g2s_B, tBgB[None, None, None, s], tBsB[None, None, None, s])
        cute.arch.cp_async_commit_group()

    for k_tile in cutlass.range(num_k_tiles, unroll=1):
        read_stage = k_tile % STAGES
        cute.arch.cp_async_wait_group(STAGES - 2)   # tile k_tile has landed
        cute.arch.barrier()                         # and stage (k_tile-1)%STAGES is dead

        # Issue the fetch for tile k_tile+STAGES-1 BEFORE the math, so the DMA
        # overlaps the mma work below. It writes the stage freed by the barrier.
        # An empty commit when out of tiles keeps the group count aligned.
        fetch_tile = k_tile + STAGES - 1
        if fetch_tile < num_k_tiles:
            write_stage = fetch_tile % STAGES
            cute.copy(g2s_A, tAgA[None, None, None, fetch_tile],
                             tAsA[None, None, None, write_stage])
            cute.copy(g2s_B, tBgB[None, None, None, fetch_tile],
                             tBsB[None, None, None, write_stage])
        cute.arch.cp_async_commit_group()

        # Consume read_stage: smem -> registers, then the mma steps.
        for k_blk in cutlass.range_constexpr(K_BLOCKS):
            cute.copy(s2r_A, tXsA[None, None, k_blk, read_stage], tXrA[None, None, k_blk])
            cute.copy(s2r_B, tXsB[None, None, k_blk, read_stage], tXrB[None, None, k_blk])
        for k_blk in cutlass.range_constexpr(K_BLOCKS):
            cute.gemm(tiled_mma, acc,
                      tCrA[None, None, k_blk],
                      tCrB[None, None, k_blk],
                      acc)

    # epilogue: exact per-thread ownership of C -> plain copy, no coordination.
    # acc is fp32 but C is fp16, and autovec_copy demands equal bit widths — so
    # convert through a same-shaped fp16 fragment first (reference :762-766).
    acc_out = cute.make_fragment_like(acc, C_DTYPE)
    acc_out.store(acc.load().to(C_DTYPE))
    cute.autovec_copy(acc_out, tCgC)







# ---- host entry point -------------------------------------------------------


@cute.jit
def gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream,
         cfg: cutlass.Constexpr):
    """Build the static structure, then launch. Reference: tensorop_gemm.py:159-330.

    `cfg` is Constexpr, so it is CONSUMED at compile time: it does not appear in the
    compiled executor's runtime signature, and every cfg.<field> below folds to a
    literal. That is why build() calls compiled(mA, mB, mC, stream) with no cfg.

        sA_layout = make_smem_layout_AB(cfg, cfg.BM)
        sB_layout = make_smem_layout_AB(cfg, cfg.BN)
        tiled_mma = make_tiled_mma(cfg)
        ...
        gemm_kernel(...).launch(
            grid=[ceil(M/BM), ceil(N/BN), 1],     # rung 6 remaps this for L2 reuse
            block=[cfg.num_throk eads, 1, 1],
            smem=cfg.smem_bytes,
            stream=stream,
        )
    """

    # kernel needs sA_layout, sB_layout, cta_tiler, tiled_mma, g2s_A, g2s_B, s2r_A, s2r_B
    # shared memory has a layout of (BM, BK, stages) for A and (BN, BK, stages) for B
    sA_layout = make_smem_layout_AB(cfg, cfg.BM)
    sB_layout = make_smem_layout_AB(cfg, cfg.BN)

    g2s_A     = make_gmem_tiled_copy_AB(cfg)
    g2s_B     = make_gmem_tiled_copy_AB(cfg)
    tiled_mma = make_tiled_mma(cfg)
    s2r_A, s2r_B = make_s2r_tiled_copies(cfg, tiled_mma)
    M = cute.size(mA, mode=[0])
    N = cute.size(mB, mode=[0])

    gemm_kernel(
        mA, mB, mC,
        tiled_mma,
        g2s_A, g2s_B,                 # one shared pattern serves A and B â or trim the
                                # signature to a single g2s param, your call
        s2r_A, s2r_B,
        sA_layout, sB_layout,
        (cfg.BM, cfg.BN, cfg.BK),
    ).launch(
        grid=[M // cfg.BM, N // cfg.BN, 1],   # exact division â no predication yet
        block=[cfg.num_threads, 1, 1],
        smem=cfg.smem_bytes,
        stream=stream,
    )


# ============================================================================
# SECTION 3 — Tensors and correctness.  WRITTEN FOR YOU.
# ============================================================================

# Multiples of the default 128x128x32 tile. These must pass from rung 0.
SHAPES_ALIGNED = [
    (128, 128, 32),      # one CTA, one k-tile
    (128, 128, 64),      # fewer k-tiles than STAGES — exercises the prologue drain
    (256, 128, 128),
    (512, 384, 256),     # rectangular, several CTAs
    (1024, 512, 4096),   # deep K, accumulation error surface
    (4096, 4096, 4096),  # the north-star shape
]

# These need predication (rung 7) and are expected to fail until then.
SHAPES_RAGGED = [
    (129, 128, 32),
    (128, 130, 32),
    (128, 128, 40),
    (1000, 1000, 1000),
]

BENCH_SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (4096, 4096, 128),   # skinny K: the pipeline barely spins up, epilogue dominates
    (8192, 128, 8192),   # skinny N: few CTAs, low-occupancy regime
]


def make_tensors(M: int, N: int, K: int, int_inputs: bool = True):
    """A (M,K) K-major, B (N,K) K-major, C (M,N) N-major.

    int_inputs draws from {-2..2}, where every fp16 product and the fp32 running sum are
    exactly representable up to 2^24. That makes the reference comparison EXACT, so a
    correctness failure is a real bug and never a tolerance argument. Use --randn for a
    realistic numerics check once the kernel is correct.
    """
    if int_inputs:
        a = torch.randint(-2, 3, (M, K), device="cuda").to(TORCH_AB)
        b = torch.randint(-2, 3, (N, K), device="cuda").to(TORCH_AB)
    else:
        a = torch.randn(M, K, dtype=TORCH_AB, device="cuda")
        b = torch.randn(N, K, dtype=TORCH_AB, device="cuda")
    c = torch.zeros(M, N, dtype=TORCH_C, device="cuda")
    return a, b, c


def to_cute(*tensors):
    # assumed_align=16 is what makes the 128-bit cp.async legal; without it the DSL
    # rejects the copy at trace time.
    return tuple(from_dlpack(t, assumed_align=16) for t in tensors)


def build(cfg: Config, M: int, N: int, K: int, stream, int_inputs: bool = True):
    """Allocate, compile for this exact shape, and return (a, b, c, run).

    Shapes are static at trace time, so each distinct (M,N,K) is its own compiled
    kernel — the same thing template instantiation does in C++ CUTLASS. cute.compile
    hoists that cost out of any loop you put `run` in.

    cfg goes in as a Constexpr and is consumed at compile time, so the returned executor
    takes only (mA, mB, mC, stream).
    """
    cfg.validate()
    a, b, c = make_tensors(M, N, K, int_inputs)
    mA, mB, mC = to_cute(a, b, c)
    compiled = cute.compile(gemm, mA, mB, mC, stream, cfg)
    return a, b, c, (lambda: compiled(mA, mB, mC, stream)), (mA, mB, mC)


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() @ b.float().T).to(TORCH_C)


def test(cfg: Config, stream, include_ragged: bool = False, int_inputs: bool = True,
         shapes=None) -> bool:
    """shapes=None runs the standard suite. Pass a single shape via --mnk while you are
    iterating: one small case compiles in a fraction of the time and fails just as loudly."""
    shapes = shapes or (SHAPES_ALIGNED + (SHAPES_RAGGED if include_ragged else []))
    print(f"== correctness ==\n    {cfg.describe()}\n")
    all_ok = True
    for M, N, K in shapes:
        ragged = (M, N, K) in SHAPES_RAGGED
        if not ragged and (M % cfg.BM or N % cfg.BN or K % cfg.BK):
            print(f"  {M:>5} x {N:>5} x {K:>5}   skipped (not a multiple of "
                  f"{cfg.BM}x{cfg.BN}x{cfg.BK} tile; needs predication)")
            continue
        try:
            a, b, c, run, _ = build(cfg, M, N, K, stream, int_inputs)
            run()
            torch.cuda.synchronize()
            ref = reference(a, b)
            err = (c.float() - ref.float()).abs().max().item()
            ok = (err == 0.0) if int_inputs else bool(
                torch.allclose(c.float(), ref.float(), atol=1e-1, rtol=1e-2))
            note = "  (needs predication)" if ragged and not ok else ""
            print(f"  {M:>5} x {N:>5} x {K:>5}   max|err| = {err:>10.3e}   "
                  f"{'PASS' if ok else 'FAIL'}{note}")
        except NotImplementedError as e:
            print(f"  {M:>5} x {N:>5} x {K:>5}   not implemented ({e})")
            return False
        except Exception as e:
            ok = False
            print(f"  {M:>5} x {N:>5} x {K:>5}   ERROR {type(e).__name__}: {e}")
        all_ok = all_ok and (ok or ragged)
    return all_ok


# ============================================================================
# SECTION 4 — Benchmark.  WRITTEN FOR YOU.
# ============================================================================


def time_ms(fn: Callable, iters: int = 50, warmup: int = 5) -> float:
    """Wall clock via CUDA events. Used for BOTH our kernel and cuBLAS so the
    head-to-head column is apples to apples."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_us_dsl(compiled, jit_args, iters: int = 50, warmup: int = 5) -> float:
    """cutlass.cute.testing.benchmark — the tool --tune uses internally.

    Note: use_cuda_graphs=True fails here with "operation not permitted when stream is
    capturing" because we are on torch's current stream. Pass a dedicated stream if you
    want graphs. workspace_generator + workspace_count give cold-L2 numbers.
    """
    return benchmark(compiled, kernel_arguments=jit_args,
                     warmup_iterations=warmup, iterations=iters)


def bench(cfg: Config, stream, shapes=None) -> None:
    shapes = shapes or BENCH_SHAPES
    print(f"== benchmark vs cuBLAS fp16 ==\n    {cfg.describe()}\n")
    print(f"  {'M':>5} {'N':>6} {'K':>6} | {'ours ms':>8} {'TFLOP/s':>8} | "
          f"{'cuBLAS ms':>9} {'TFLOP/s':>8} | {'%cuBLAS':>7}")
    for M, N, K in shapes:
        try:
            a, b, c, run, _ = build(cfg, M, N, K, stream)
        except NotImplementedError as e:
            print(f"  not implemented ({e})")
            return
        ms = time_ms(run)
        d = torch.empty(M, N, dtype=TORCH_AB, device="cuda")
        ms_ref = time_ms(lambda: torch.mm(a, b.T, out=d))
        flops = 2 * M * N * K
        tf, tf_ref = (flops / (t * 1e-3) / 1e12 for t in (ms, ms_ref))
        print(f"  {M:>5} {N:>6} {K:>6} | {ms:>8.3f} {tf:>8.1f} | "
              f"{ms_ref:>9.3f} {tf_ref:>8.1f} | {100*ms_ref/ms:>6.1f}%")


# ============================================================================
# SECTION 5 — Autotune.  WRITTEN FOR YOU.
# ============================================================================
# This only works because Section 1 derives everything. Add a field to Config, add it to
# the grid, and it sweeps — no other edit. Points that fail validate() are skipped, so
# you can write a generous grid and let the asserts prune it.

TUNE_GRID = {
    "BM": [128],
    "BN": [128],
    "BK": [32, 64],
    "STAGES": [3, 4, 5],
    "WARPS": [(2, 2, 1), (4, 1, 1), (2, 4, 1)],
    "PERM_N_FACTOR": [1, 2],
}


def tune(stream, M: int = 4096, N: int = 4096, K: int = 4096, grid=None) -> None:
    import itertools

    grid = grid or TUNE_GRID
    keys = list(grid)
    results = []
    skipped = 0
    print(f"== autotune at {M}x{N}x{K} ==")
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = Config(**dict(zip(keys, combo)))
        try:
            cfg.validate()
        except AssertionError:
            skipped += 1
            continue
        try:
            _, _, _, run, _ = build(cfg, M, N, K, stream)
            ms = time_ms(run, iters=20, warmup=3)
        except NotImplementedError as e:
            print(f"  not implemented ({e})")
            return
        except Exception as e:
            print(f"  {dict(zip(keys, combo))}  ERROR {type(e).__name__}: {e}")
            continue
        tf = 2 * M * N * K / (ms * 1e-3) / 1e12
        results.append((tf, ms, cfg))
        print(f"  {tf:>7.1f} TF  {ms:>7.3f} ms   " +
              "  ".join(f"{k}={v}" for k, v in zip(keys, combo)))

    # No silent truncation: say what the grid dropped.
    print(f"\n  {len(results)} configs timed, {skipped} rejected by validate()")
    if results:
        results.sort(reverse=True, key=lambda r: r[0])
        tf, ms, cfg = results[0]
        print(f"\n  BEST: {tf:.1f} TFLOP/s ({ms:.3f} ms)\n    {cfg.describe()}")


# ============================================================================
# SECTION 6 — CLI.  WRITTEN FOR YOU.
# ============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--test", action="store_true", help="correctness suite")
    p.add_argument("--bench", action="store_true", help="benchmark vs cuBLAS")
    p.add_argument("--tune", action="store_true", help="sweep TUNE_GRID")
    p.add_argument("--ragged", action="store_true", help="include shapes needing predication")
    p.add_argument("--randn", action="store_true",
                   help="normal inputs instead of exact integers")
    p.add_argument("--mnk", type=str, default=None, help="single shape, e.g. 4096,4096,4096")
    for f in ("BM", "BN", "BK", "STAGES", "PERM_N_FACTOR"):
        p.add_argument(f"--{f}", type=int, default=None)
    p.add_argument("--warps", type=str, default=None, help="e.g. 2,2,1")
    args = p.parse_args()

    overrides = {f: getattr(args, f) for f in ("BM", "BN", "BK", "STAGES", "PERM_N_FACTOR")
                 if getattr(args, f) is not None}
    if args.warps:
        overrides["WARPS"] = tuple(int(x) for x in args.warps.split(","))
    cfg = Config(**overrides)
    # A config typed on the command line must fail loudly. (--tune deliberately uses
    # validate() to PRUNE its grid instead, so a generous grid is fine there.)
    try:
        cfg.validate()
    except AssertionError as e:
        p.error(f"invalid config: {e}")

    shapes = None
    if args.mnk:
        M, N, K = (int(x) for x in args.mnk.split(","))
        shapes = [(M, N, K)]

    if not torch.cuda.is_available():
        raise RuntimeError("this needs a GPU")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}  free {free/2**30:.1f} / "
          f"{total/2**30:.1f} GiB\n")

    stream = cutlass_torch.current_stream()
    if not (args.test or args.bench or args.tune):
        print(cfg.describe())
        return
    if args.test:
        test(cfg, stream, include_ragged=args.ragged, int_inputs=not args.randn,
             shapes=shapes)
        print()
    if args.bench:
        bench(cfg, stream, shapes)
        print()
    if args.tune:
        M, N, K = shapes[0] if shapes else (4096, 4096, 4096)
        tune(stream, M, N, K)


if __name__ == "__main__":
    main()
