"""Shared timing utilities: cold-cache, CUDA-graph timing."""

import torch


def time_graph(fn, arg_sets, calls_per_set: int = 1, reps: int = 30, warmup: int = 5):
    """Capture one CUDA graph that calls fn(*args) for every args in arg_sets (rotating
    through them so operands are cold in L2), replay it, and return per-call µs
    (median, min) over `reps` replays. Launch overhead is excluded by the graph."""
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warm-up outside the graph (JIT, cuBLAS handles, allocator)
        for a in arg_sets[:2]:
            fn(*a)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls_per_set):
            for a in arg_sets:
                fn(*a)
    n_calls = calls_per_set * len(arg_sets)

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        times.append(st.elapsed_time(en) * 1e3 / n_calls)
    times.sort()
    return times[len(times) // 2], times[0]
