import numpy as np
from .base import Detection, _runs, _merge


def detect_geometric_warp(feat, baseline_eps=1.0, spread_eps=1.0, gap=30, min_run=100):
    rd = feat["row_diff"]
    baseline = np.percentile(rd, 20, axis=1)
    spread = np.percentile(rd, 90, axis=1) - baseline
    elevated = (baseline > baseline_eps) & (spread > spread_eps)
    merged = _merge(_runs(elevated, 1), gap)
    return [Detection("geometric_warp", s, e) for s, e in merged if e - s + 1 >= min_run]
