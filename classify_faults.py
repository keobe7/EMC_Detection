import sys
import numpy as np
import pandas as pd

CFG = dict(
    ref_window=(100, 600),   # healthy frames
    skip_start=40,           # startup errors
    trim_end=20,             # video-end errors
    spike_scale=1.5,         # threshold = 1.5 * p99.5 of reference roi_diff
    spike_mad=8,             # ... or median + 8*MAD, whichever is larger
    cluster_gap=12,          # distance between spikes to become one edge
    min_edge_peak=2.5,       # without 3 spikes, it's just a blip
    blackout_min_gap=200,    # minimal interval between the 2 edges of a blackout
    flicker_min_edges=4,
    flicker_max_med_gap=150, # median gap between consecutive edges
    pre_win=(40, 5),         # frames before an edge for d_lit / d_spread
    post_win=(30, 70),       # frames after an edge (skips AE settling)
)

def load(path):
    return pd.read_csv(path).replace([np.inf, -np.inf], np.nan)

def spike_threshold(rd, cfg):
    a, b = cfg["ref_window"]
    ref = rd[a:b]
    med = np.nanmedian(ref)
    mad = np.nanmedian(np.abs(ref - med)) * 1.4826
    return max(np.nanpercentile(ref, 99.5) * cfg["spike_scale"], med + cfg["spike_mad"] * mad)

def cluster_spikes(rd, thr, cfg):
    n = len(rd)
    idx = np.flatnonzero(rd > thr)
    idx = idx[(idx >= cfg["skip_start"]) & (idx < n - cfg["trim_end"])]
    clusters = []
    for f in idx:
        if clusters and f - clusters[-1]["end"] <= cfg["cluster_gap"]:
            c = clusters[-1]
            c["end"], c["n"], c["peak"] = f, c["n"] + 1, max(c["peak"], rd[f])
        else:
            clusters.append(dict(start=f, end=f, n=1, peak=rd[f]))
    edges = [c for c in clusters if c["peak"] >= cfg["min_edge_peak"] * thr]
    blips = [c for c in clusters if c["peak"] < cfg["min_edge_peak"] * thr]
    return edges, blips

def window_median(x, lo, hi):
    lo, hi = max(lo, 0), min(hi, len(x))
    return np.nanmedian(x[lo:hi]) if hi > lo else np.nan

def add_context(edges, lit, spread, cfg):
    for c in edges:
        s, e = c["start"], c["end"]
        c["d_lit"] = window_median(lit, e + cfg["post_win"][0], e + cfg["post_win"][1]) - \
                     window_median(lit, s - cfg["pre_win"][0], s - cfg["pre_win"][1])
        c["d_spread"] = window_median(spread, e + cfg["post_win"][0], e + cfg["post_win"][1]) - \
                        window_median(spread, s - cfg["pre_win"][0], s - cfg["pre_win"][1])

def verdict_from_edges(edges, blips, cfg):
    if not edges:
        return "NO ANOMALY DETECTED", "no edges above the noise floor"
    starts = np.array([c["start"] for c in edges])
    gaps = np.diff(starts)
    if len(edges) == 1:
        return "SINGLE TRANSITION", "one unpaired edge (level shift or open-ended change), manual"
    if len(edges) == 2 and gaps[0] >= cfg["blackout_min_gap"]:
        interior = sum(1 for b in blips if edges[0]["end"] < b["start"] < edges[1]["start"])
        return "BLACKOUT", (f"2 edges {gaps[0]} frames apart, quiet between "
                            f"({interior} blip(s) in between)")
    if len(edges) >= cfg["flicker_min_edges"] and np.median(gaps) <= cfg["flicker_max_med_gap"]:
        return "FLICKER", (f"{len(edges)} edges, median gap {np.median(gaps):.0f} frames, "
                           f"gaps {sorted(set(int(g) for g in gaps))[:1]}..{int(gaps.max())}")
    return "UNCERTAIN", f"{len(edges)} edges with gaps {gaps.tolist()}; fits neither pattern"

def analyze(path, cfg=CFG):
    d = load(path)
    rd = d["baseline_ff"].to_numpy(float)
    thr = spike_threshold(rd, cfg)
    edges, blips = cluster_spikes(rd, thr, cfg)
    add_context(edges, d["roi_lit"].to_numpy(float), d["spread"].to_numpy(float), cfg)
    v, why = verdict_from_edges(edges, blips, cfg)
    return thr, edges, blips, v, why

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <data.csv> [more.csv ...]")
        sys.exit(1)
    pd.set_option("display.width", 160)
    for path in sys.argv[1:]:
        thr, edges, blips, v, why = analyze(path)
        print(f"\n=== {path}")
        print(f"roi_diff spike threshold: {thr:.2f}   edges: {len(edges)}   blips ignored: {len(blips)}")
        if edges:
            t = pd.DataFrame(edges)[["start", "end", "n", "peak", "d_lit", "d_spread"]]
            t.columns = ["start", "end", "spikes", "peak", "d_lit", "d_spread"]
            print(t.round(3).to_string(index=False))
        print(f"VERDICT: {v}  --  {why}")

if __name__ == "__main__":
    main()
