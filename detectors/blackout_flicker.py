import numpy as np
from .base import Detection


CFG = dict(
    ref_window=(100, 600),
    skip_start=40,
    trim_end=20,

    spike_scale=1.5,
    spike_mad=8,

    cluster_gap=12,
    min_edge_spikes=3,

    blackout_min_gap=200,

    flicker_min_edges=4,
    flicker_max_med_gap=150,
)


def _spike_threshold(rd, cfg):
    a, b = cfg["ref_window"]
    ref = rd[a:b]

    med = np.nanmedian(ref)
    mad = np.nanmedian(np.abs(ref - med)) * 1.4826

    return max(
        np.nanpercentile(ref, 99.5) * cfg["spike_scale"],
        med + cfg["spike_mad"] * mad,
    )


def _cluster_spikes(rd, threshold, cfg):
    n = len(rd)

    idx = np.flatnonzero(rd > threshold)
    idx = idx[
        (idx >= cfg["skip_start"]) &
        (idx < n - cfg["trim_end"])
    ]

    clusters = []

    for f in idx:
        if clusters and f - clusters[-1]["end"] <= cfg["cluster_gap"]:
            c = clusters[-1]
            c["end"] = f
            c["n"] += 1
            c["peak"] = max(c["peak"], rd[f])
        else:
            clusters.append({
                "start": f,
                "end": f,
                "n": 1,
                "peak": rd[f],
            })

    edges = [
        c for c in clusters
        if c["n"] >= cfg["min_edge_spikes"]
    ]

    blips = [
        c for c in clusters
        if c["n"] < cfg["min_edge_spikes"]
    ]

    return edges, blips


def _window_median(x, lo, hi):
    lo = max(lo, 0)
    hi = min(hi, len(x))

    if hi <= lo:
        return np.nan

    return np.nanmedian(x[lo:hi])


def _add_context(edges, spread, cfg):
    for c in edges:
        s = c["start"]
        e = c["end"]

        before = _window_median(
            spread,
            s - 40,
            s - 5,
        )

        after = _window_median(
            spread,
            e + 30,
            e + 70,
        )

        c["d_spread"] = after - before


def detect_blackout_flicker(feat, cfg=CFG):
    """
    Detect blackout/flicker from frame-difference edges.

    Returns:
        (blackouts, flickers)
    """

    roi_diff = feat["roi_diff"]

    # row_diff is used to derive the local background-subtraction
    # context signal.
    row_diff = feat["row_diff"]

    baseline = np.percentile(row_diff, 20, axis=1)
    spread = np.percentile(row_diff, 90, axis=1) - baseline

    threshold = _spike_threshold(roi_diff, cfg)

    edges, blips = _cluster_spikes(
        roi_diff,
        threshold,
        cfg,
    )

    _add_context(edges, spread, cfg)

    blackouts = []
    flickers = []

    if not edges:
        return blackouts, flickers

    starts = np.array([e["start"] for e in edges])
    gaps = np.diff(starts)

    # ---------------------------------------------------------
    # BLACKOUT
    # ---------------------------------------------------------

    if len(edges) == 2:
        gap = gaps[0]

        if gap >= cfg["blackout_min_gap"]:
            first = edges[0]["d_spread"]
            second = edges[1]["d_spread"]

            # Blackout should produce opposite edge directions.
            if first * second < 0:
                blackouts.append(
                    Detection(
                        "blackout",
                        edges[0]["start"],
                        edges[1]["end"],
                    )
                )

    # ---------------------------------------------------------
    # FLICKER
    # ---------------------------------------------------------

    if (
        len(edges) >= cfg["flicker_min_edges"]
        and np.median(gaps) <= cfg["flicker_max_med_gap"]
    ):
        flickers.append(
            Detection(
                "flicker",
                edges[0]["start"],
                edges[-1]["end"],
            )
        )

    return blackouts, flickers
