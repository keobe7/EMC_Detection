import numpy as np
from .base import Detection, _runs, _running_background

def detect_blackout(feat, alpha=0.02, level_eps=0.05, min_run=3,
                     pretrend_window=20, pretrend_max=0.01):
    luma = feat["luma"]
    bg = _running_background(luma, alpha)
    ratio = luma / bg
    dim = ratio < 1.0 - level_eps
    out = []
    for s, e in _runs(dim, min_run):
        pre_start = max(0, s - pretrend_window)
        if pre_start < s:
            pre_bg = bg[pre_start:s]
            pre_drift = (pre_bg[-1] - pre_bg[0]) / max(abs(pre_bg[0]), 1e-6)
            if pre_drift < -pretrend_max:
                continue  # background was already falling — likely an oscillation trough
        out.append(Detection("blackout", s, e))
    return out
