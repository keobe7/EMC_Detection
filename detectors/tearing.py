import numpy as np
from .base import Detection, _runs


def detect_tearing(feat, touch_eps=1.5, min_frac=0.015, max_frac=0.6, baseline_eps=1.0, min_run=20):
    rd = feat["row_diff"]
    touched = rd > touch_eps
    frac = touched.mean(axis=1)
    baseline = np.percentile(rd, 20, axis=1)
    mask = (frac > min_frac) & (frac < max_frac) & (baseline < baseline_eps)
    return [Detection("tearing", s, e) for s, e in _runs(mask, min_run)]
