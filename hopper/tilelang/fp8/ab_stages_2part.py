"""Same-run A/B: does the 2-part gain come from the overlap or from the 5th stage?

    v0   S=4, v0 S=5  (one part buffer, wait<0>)
    2part S=4, 2part S=5  (two part buffers, wait<1>, chunk 8)

Run: python ab_stages_2part.py
(interleaved bench; PROFILE=1 runs one launch per config for ncu)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemm_ws_tma_async as v0  # noqa: E402
import gemm_ws_tma_async_2part as v1  # noqa: E402
from harness import M, N, K, make_inputs  # noqa: E402

CONFIGS = {
    "v0 S=4": v0.make(4, 8, True),
    "v0 S=5": v0.make(5, 8, True),
    "2part S=4 ch=8": v1.make(4, 8, True, 8),
    "2part S=5 ch=8": v1.make(5, 8, True, 8),
}


def main():
    a, b, sa, sb = make_inputs(0, "scale_stress")
    if os.environ.get("PROFILE"):
        for f in CONFIGS.values():
            f(a, b, sa, sb)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile")
        for f in CONFIGS.values():
            f(a, b, sa, sb)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return
    ref = CONFIGS["v0 S=4"](a, b, sa, sb).float()
    for name, f in CONFIGS.items():
        d = (f(a, b, sa, sb).float() - ref).abs().max().item()
        print(f"  {name:<16} max|out - v0 S=4| = {d:.3g}")
    flops = 2 * M * N * K
    res = v1.bench_interleaved({k: (lambda f=f: f(a, b, sa, sb)) for k, f in CONFIGS.items()})
    for name, ms in res.items():
        print(f"  {name:<16} {ms * 1e3:8.1f} us   {flops / ms / 1e9:7.1f} TFLOP/s")


if __name__ == "__main__":
    main()
