import numpy as np
from .base import Detection, _runs, _merge, _running_background


def detect_flicker(feat, alpha=0.02, level_eps=0.02, gap=45, min_run=60, max_jitter=0.2):
    luma = feat["luma"]
    bg = _running_background(luma, alpha)
    ratio = luma / bg
    dev = np.abs(ratio - 1.0) > level_eps
    out = []
    for s, e in _merge(_runs(dev, 1), gap):
        if e - s + 1 < min_run:
            continue
        seg = ratio[s:e + 1]
        two_sided = (seg > 1 + level_eps).any() and (seg < 1 - level_eps).any()
        lo, hi = np.percentile(seg, [5, 95])
        jitter = np.median(np.abs(np.diff(seg))) / max(hi - lo, 1e-9)
        if two_sided and jitter <= max_jitter:
            out.append(Detection("flicker", s, e))
    return out
