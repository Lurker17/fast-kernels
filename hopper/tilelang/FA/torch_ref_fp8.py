"""Harness smoke test, not a kernel: blockwise online-softmax attention in torch (BN = 128, p rounded to
e4m3 before P.V), the way a FlashAttention kernel computes it. Validates the harness and its error bounds
before any tilelang kernel exists."""

PRECISION = "fp8"

try:
    from harness import FP8, online_emu
except ImportError:  # loaded by modal_app._load under a prefixed module name
    import importlib.util
    import pathlib

    _spec = importlib.util.spec_from_file_location("attn_harness", pathlib.Path(__file__).with_name("harness.py"))
    _h = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_h)
    FP8, online_emu = _h.FP8, _h.online_emu


def make():
    return lambda q, k, v, dq, dk, dv, causal: online_emu(q, k, v, causal, descales=(dq, dk, dv), p_dtype=FP8)
