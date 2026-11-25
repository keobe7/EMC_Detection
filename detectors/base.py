from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Detection:
    fault_type: str
    start_frame: int
    end_frame: int  # inclusive, same convention as manifest.csv

REF_SCALE = 0.1  # downsample factor for row-profile features

def _running_background(values, alpha=0.02):
    bg = np.empty_like(values, dtype=np.float64)
    bg[0] = values[0]
    for i in range(1, len(values)):
        bg[i] = alpha * values[i - 1] + (1 - alpha) * bg[i - 1]
    return bg

def _runs(mask, min_len):
    runs, start = [], None
    for k, v in enumerate(mask):
        if v and start is None:
            start = k
        elif not v and start is not None:
            if k - start >= min_len:
                runs.append((start, k - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask) - 1))
    return runs

def _merge(runs, gap):
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged

def load_reference(path, roi, scale=REF_SCALE, frame_range=None):
    cap = cv2.VideoCapture(path)
    x0, y0, x1, y1 = [int(v * scale) for v in roi]

    if frame_range is None:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"couldn't read reference frame from {path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        start, end = frame_range
        acc, n = None, 0
        brightness = []
        for i in range(end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if i < start:
                continue
            brightness.append(frame.mean())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
            acc = small if acc is None else acc + small
            n += 1
        cap.release()
        if n == 0:
            raise RuntimeError(f"no frames read from {path} in range {frame_range}")
        brightness = np.array(brightness)
        if brightness.std() / max(brightness.mean(), 1e-6) > 0.15:
            raise RuntimeError(
                f"reference range {frame_range} in {path} isn't uniformly bright "
                f"(brightness varied {brightness.min():.1f}-{brightness.max():.1f}) "
                f"— may include a fault frame or a cut; check the range"
            )
        small = (acc / n).astype(np.uint8)

    small[y0:y1, x0:x1] = 0
    return small

def extract_features(path, roi, reference=None, lit_level=128, scale=REF_SCALE):
    x0, y0, x1, y1 = roi
    sx0, sy0, sx1, sy1 = [int(v * scale) for v in roi]
    cap = cv2.VideoCapture(path)
    diffs, lit, luma, row_diffs, row_diffs_ff = [], [], [], [], []   # CHANGED
    prev = None
    prev_small = None                                                # NEW
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray_full[y0:y1, x0:x1]
        lit.append(float((crop > lit_level).mean()))
        diffs.append(np.inf if prev is None else float(cv2.absdiff(crop, prev).mean()))
        luma.append(float(frame.mean()))

        small = cv2.resize(gray_full, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)             # CHANGED (moved out)
        small[sy0:sy1, sx0:sx1] = 0                                  # same timestamp mask
        if reference is not None:
            row_diffs.append(cv2.absdiff(small, reference).mean(axis=1))
        if prev_small is None:                                       # NEW
            row_diffs_ff.append(np.zeros(small.shape[0]))
        else:
            row_diffs_ff.append(cv2.absdiff(small, prev_small).mean(axis=1))
        prev_small = small                                           # NEW (resize makes a fresh array, no copy needed)

        prev = crop
    cap.release()
    out = {"roi_diff": np.array(diffs), "roi_lit": np.array(lit),
           "luma": np.array(luma), "row_diff_ff": np.array(row_diffs_ff)}  # CHANGED
    if reference is not None:
        out["row_diff"] = np.array(row_diffs)
    return out
