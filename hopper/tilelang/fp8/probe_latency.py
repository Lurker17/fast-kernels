"""Measure TMA load latency vs pipeline lead in the best config, from in-kernel clock64 stamps (dbg=True).

Per SM, per global k index it (WG0's view; producer and consumer share the SM clock):
  t_issue : producer issued stage it's A/B TMA        t_need : WG0 starts waiting on full[it]
  t_ready : WG0's full wait returns (bytes + 128 producer arrivals landed)
  lead    = t_need - t_issue   how early the load was issued relative to when it's needed
  stall   = t_ready - t_need   consumer wait (≈ 0 when the data was already there)
  latency = t_ready - t_issue  issue -> landed; only an exact load latency when the consumer actually stalled
Run: python probe_latency.py   (modal: modal run modal_app.py::script --path kernels/hopper_fp8_gemm/probe_latency.py)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import make_inputs  # noqa: E402
from gemm_ws_persistent import fp8_gemm_persistent  # noqa: E402


def pct(x, qs=(0.1, 0.5, 0.9, 0.99)):
    x = x.double()
    return "  ".join(f"p{int(q * 100)} {torch.quantile(x, q).item():6.0f}" for q in qs)


def probe(a, b, sa, sb, stages):
    kw = dict(stages=stages, swizzle=8, chunk=4, mode="none", scale_tma=False, pf=4, tma_epi=True)
    ref = fp8_gemm_persistent(a, b, sa, sb, **kw)
    for _ in range(3):  # warm; keep the last call's stamps
        C, D = fp8_gemm_persistent(a, b, sa, sb, dbg=True, **kw)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(10):
        fp8_gemm_persistent(a, b, sa, sb, **kw)
    en.record()
    en.synchronize()
    print(f"\n=== S={stages}: {st.elapsed_time(en) / 10 * 1e3:.0f} us/call (no dbg), dbg output == non-dbg: {torch.equal(C, ref)}")

    n_k = 64
    ti, tn_, tr = D[..., 0], D[..., 1], D[..., 2]
    # T.empty is uninitialised: rows past this SM's last tile hold garbage -> keep only ordered, plausible stamps
    valid = (ti > 0) & (tn_ >= ti - 10**6) & (tr >= tn_) & (tr - ti < 10**6)
    k = torch.arange(D.shape[1], device=D.device) % n_k
    steady = valid & (k >= 4)[None, :]
    lead, stall, lat = (tn_ - ti), (tr - tn_), (tr - ti)
    both = valid[:, 1:] & valid[:, :-1] & (k[1:] != 0)[None, :]
    print("k-step period          :", pct((tr[:, 1:] - tr[:, :-1])[both]))
    print("lead   (need - issue)  :", pct(lead[steady]))
    print("stall  (ready - need)  :", pct(stall[steady]))
    stl = steady & (stall > 64)
    print(f"stalled steps          : {100 * stl.sum().item() / steady.sum().item():.1f}%")
    print("LATENCY (ready - issue), stalled steps:", pct(lat[stl]))
    first = steady & (torch.arange(D.shape[1], device=D.device) < n_k)[None, :]
    print("  first tile only (colder L2)         :", pct(lat[first & (stall > 64)]))
    return D


def main():
    a, b, sa, sb = make_inputs(0, "scale_stress")
    os.makedirs("/tmp/out", exist_ok=True)
    for S in (4, 5):
        D = probe(a, b, sa, sb, S)
        torch.save(D.cpu(), f"/tmp/out/probe_latency_S{S}.pt")


if __name__ == "__main__":
    main()
