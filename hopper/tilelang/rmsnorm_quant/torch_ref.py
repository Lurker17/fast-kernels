"""Harness smoke test, not a kernel: the spec in plain torch (fp32 math, ~10 kernels). Checks the harness itself
before any tilelang kernel exists; it is also the "torch eager" baseline."""

try:
    from harness import torch_ref_fn
except ImportError:  # loaded by modal_app._load under a prefixed module name
    import importlib.util
    import pathlib

    _spec = importlib.util.spec_from_file_location("rms_harness", pathlib.Path(__file__).with_name("harness.py"))
    _h = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_h)
    torch_ref_fn = _h.torch_ref_fn


def make():
    return torch_ref_fn
