import argparse
import numpy as np
import pandas as pd

# Variables
WARMUP      = 20     # Startup errors
SPREAD_THR  = 1.0    # spread_ff above this = "frame is changing row-wise"
                     # (clean/blackout-hold floor is ~0.15, p99 ~0.8-1.4)
GAP         = 6      # bridge quiet gaps up to this many frames inside a run to sustain
                     # (tear/warp gaps are <=2; flicker bursts are split by ~40 frame gaps)
MIN_RUN     = 250    # flicker bursts are ~100 frames, tear/warp ~590
WARP_RATIO  = 0.06   # median(frac_ff)/median(spread_ff): tearing 0.026-0.048, warp 0.077-0.141
                     # (single thresholds on frac_ff or spread_ff fail across severities)
                     # NOTE: mild files overlap (tearing_mild 0.051, warp_mild 0.049-0.078 depending
                     # on which frames are in the run), so this is now only the fallback.
# The spread_ff threshold also adapts to the camera: "healthy" spread_ff is ~0.12-0.16 in every
# recording so far, and tearing seen through a different set-up raised it only to ~0.45 (4x), well
# under the old absolute 1.0. So the run threshold is SPREAD_REL x the healthy median, kept between
# SPREAD_MIN and SPREAD_THR.
SPREAD_REL  = 2.5
SPREAD_MIN  = 0.25
EXCL_PAD    = 5      # extra frames around excluded (blackout/flicker) edges

# roi_diff test for geometric warp (warp makes the ROI itself change a lot frame to frame)
#   warp_mild  : roi_diff ~51-57 vs healthy ~7.4  (about 7x)
#   tearing_mild: roi_diff ~9.7  vs healthy ~8.3  (about 1.2x)
WARP_ROI    = 2.0    # run counts as warp if median roi_diff > WARP_ROI * healthy roi_diff
ROI_REF     = (100, 600)  # frames used to measure healthy roi_diff (same window as classify_faults)
ROI_SMOOTH  = 5      # rolling-median window applied to roi_diff before thresholding
ROI_GAP     = 10     # bridge gaps up to this many frames in the roi_diff mask


def find_runs(mask, gap):
    runs, start, last = [], None, None
    for i, v in enumerate(mask):
        if not v:
            continue
        if start is None:
            start = last = i
        elif i - last - 1 > gap:
            runs.append((start, last))
            start = i
        last = i
    if start is not None:
        runs.append((start, last))
    return runs

def parse_exclude(s):
    out = []
    for part in filter(None, (s or "").split(",")):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out

def healthy_roi_diff(roi):
    a, b = ROI_REF
    ref = roi[a:b]
    ref = ref[~np.isnan(ref)]
    if len(ref) < 100:                       # short video: fall back to whole-video median
        ref = roi[~np.isnan(roi)]
    return float(np.median(ref)) if len(ref) else np.nan

def classify(df, exclude=(), spread_thr=SPREAD_THR, gap=GAP, min_run=MIN_RUN,
             warp_ratio=WARP_RATIO, warp_roi=WARP_ROI):
    spread_raw = df["spread_ff"].to_numpy(dtype=float)
    spread = spread_raw.copy()
    frac = df["frac_ff"].to_numpy(dtype=float)
    roi = (df["roi_diff"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
           if "roi_diff" in df.columns else None)

    spread[:WARMUP] = 0
    for a, b in exclude:                      # belt-and-suspenders overlap guard
        spread[max(0, a - EXCL_PAD): b + EXCL_PAD + 1] = 0

    found = []

    a_, b_ = ROI_REF
    ref_sp = spread_raw[a_:b_]
    ref_sp = ref_sp[~np.isnan(ref_sp)]
    if len(ref_sp) >= 100 and spread_thr == SPREAD_THR:
        spread_thr = float(min(SPREAD_THR, max(SPREAD_MIN, SPREAD_REL * np.median(ref_sp))))

    # 1) warp from roi_diff: a long run where the ROI changes far more than in healthy frames.
    warp_spans = []
    healthy = np.nan
    if roi is not None:
        healthy = healthy_roi_diff(roi)
        if healthy and not np.isnan(healthy):
            sm = pd.Series(roi).rolling(ROI_SMOOTH, center=True, min_periods=1).median().to_numpy()
            mask = sm > warp_roi * healthy
            mask[:WARMUP] = False
            for a, b in exclude:
                mask[max(0, a - EXCL_PAD): b + EXCL_PAD + 1] = False
            for s, e in find_runs(mask, ROI_GAP):
                n = e - s + 1
                if n < min_run:
                    continue
                f_med = float(np.median(frac[s:e + 1]))
                sp_med = float(np.median(spread_raw[s:e + 1]))
                r_med = float(np.nanmedian(roi[s:e + 1]))
                found.append(dict(kind="geometric_warp", start=s, end=e, length=n,
                                  spread_ff=round(sp_med, 2), frac_ff=round(f_med, 3),
                                  ratio=round(f_med / sp_med if sp_med > 0 else 0.0, 3),
                                  roi_x=round(r_med / healthy, 1)))
                warp_spans.append((s, e))

    # 2) tearing / warp from spread_ff runs (skipping spans already reported as warp)
    for s, e in find_runs(spread > spread_thr, gap):
        n = e - s + 1
        if n < min_run:
            continue
        if any(s <= we and e >= ws for ws, we in warp_spans):
            continue
        f_med = float(np.median(frac[s:e + 1]))
        sp_med = float(np.median(spread[s:e + 1]))
        ratio = f_med / sp_med if sp_med > 0 else 0.0
        roi_x = (float(np.nanmedian(roi[s:e + 1])) / healthy
                 if roi is not None and healthy and not np.isnan(healthy) else np.nan)
        # roi_diff is the reliable separator (warp 5-8x healthy, tearing ~1.1-1.2x). The frac/spread ratio
        # changes with the camera set-up, so it is only the fallback when roi_diff is not available.
        is_warp = (roi_x > warp_roi) if not np.isnan(roi_x) else ratio > warp_ratio
        found.append(dict(kind="geometric_warp" if is_warp else "tearing", start=s, end=e, length=n,
                          spread_ff=round(sp_med, 2), frac_ff=round(f_med, 3),
                          ratio=round(ratio, 3),
                          roi_x=None if np.isnan(roi_x) else round(roi_x, 1)))
    found.sort(key=lambda r: r["start"])
    return found

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--exclude", default="",
                    help="frame ranges already claimed by blackout/flicker, e.g. 646-656,1245-1259")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    res = classify(df, parse_exclude(args.exclude))
    print(f"=== {args.csv}")
    if not res:
        print("VERDICT: no sustained tearing/warp")
    for r in res:
        print(f"{r['kind']:15s} {r['start']:5d}-{r['end']:<5d} len={r['length']:4d} "
              f"median spread_ff={r['spread_ff']:<5} median frac_ff={r['frac_ff']} "
              f"ratio={r['ratio']} roi_diff x{r['roi_x']}")
