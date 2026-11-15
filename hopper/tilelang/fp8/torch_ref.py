"""Harness smoke test, not a kernel: dequantize to fp32 and matmul in fp32 (TF32 off). Slow (~20 ms) but exact enough
to pass the accuracy bound, so it checks the harness itself before any tilelang kernel exists."""

import torch

G = 128


def make():
    torch.backends.cuda.matmul.allow_tf32 = False

    def fn(a, b, sa, sb):
        a32 = a.float() * sa.repeat_interleave(G, dim=1)
        b32 = b.float() * sb.repeat_interleave(G, dim=0).repeat_interleave(G, dim=1)
        return (a32 @ b32.T).to(torch.bfloat16)

    return fn
