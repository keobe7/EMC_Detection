from .base import Detection, _runs


def detect_freeze(feat, diff_eps=0.3, lit_min=0.01, min_run=8):
    static = (feat["roi_diff"] < diff_eps) & (feat["roi_lit"] > lit_min)
    return [Detection("freeze", s, e) for s, e in _runs(static, min_run)]
