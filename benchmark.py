#!/usr/bin/env python3
"""
benchmark.py - measure how the detector itself performs (CPU, memory, speed, latency).

Give it a folder (or files) of recordings and/or *_data.csv files and run it ON THE MACHINE YOU WANT
TO REPORT (e.g. the Jetson).  Nothing is invented: every number comes from a measurement made here,
and hardware.json records which machine it was.

    python benchmark.py recordings/ --roi 390,151,576,180 --out bench_out
    python benchmark.py recordings/ --ts-box 40,20,200,50 --out bench_out     # locate the screen itself
    python benchmark.py data_csvs/ --out bench_out                             # CSVs only (no video stages)

Stages
  1 extraction   video -> per-frame features (decode time and feature time per frame, CPU use, peak memory)
  2 analysis     catalog + general detector + EMC grade, whole recording at once (time, memory)
  3 replay       the live monitor fed frame by frame: cost of every periodic re-analysis, detection latency
  4 scaling      analysis cost against recording length (the real CSV tiled to longer lengths; synthetic)
Optional
  --truth truth.csv   columns file,type,start_s,end_s  -> detected / missed / false alarms per recording

Outputs (in --out): hardware.json, extraction.csv, frame_times.csv, analysis.csv, replay.csv,
latency.csv, scaling.csv, truth_check.csv (if --truth), summary.md
Options: --repeats N (default 3; repeated runs give the +/- 95% confidence interval), --fps 30,
--no-video (skip stages that need video), --max-frames N (cut long recordings).
"""
import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:                                   # pip install psutil
    sys.exit("benchmark.py needs psutil:  pip install psutil")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


# ------------------------------------------------------------------ helpers
def ci95(vals):
    """mean and 95% half-width (Student t) of a list of repeat measurements."""
    v = [float(x) for x in vals]
    if len(v) < 2:
        return (v[0] if v else float("nan")), float("nan")
    t = T95.get(len(v) - 1, 1.96)
    return statistics.mean(v), t * statistics.stdev(v) / len(v) ** 0.5


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


class Sampler:
    """Samples this process's resident memory and CPU every `dt` seconds while a block runs."""

    def __init__(self, dt=0.1):
        self.p, self.dt = psutil.Process(), dt
        self.rss, self.cpu = [], []

    def __enter__(self):
        self.p.cpu_percent(None)
        self.t0, self.c0 = time.perf_counter(), sum(self.p.cpu_times()[:2])
        self._stop = threading.Event()
        self.rss0 = self.p.memory_info().rss
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def _run(self):
        while not self._stop.wait(self.dt):
            self.rss.append(self.p.memory_info().rss)
            self.cpu.append(self.p.cpu_percent(None))      # % of ONE core, can exceed 100

    def __exit__(self, *a):
        self._stop.set()
        self._th.join()
        self.wall = time.perf_counter() - self.t0
        self.cpu_s = sum(self.p.cpu_times()[:2]) - self.c0
        self.rss.append(self.p.memory_info().rss)
        self.peak_mb = max(self.rss) / 2**20
        self.start_mb = self.rss0 / 2**20
        self.mean_cpu_pct = self.cpu_s / self.wall * 100 if self.wall else float("nan")


def hardware():
    info = dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                cpu_count_logical=psutil.cpu_count(True), cpu_count_physical=psutil.cpu_count(False),
                ram_gb=round(psutil.virtual_memory().total / 2**30, 2), numpy=np.__version__, pandas=pd.__version__)
    try:
        import cv2
        info["opencv"] = cv2.__version__
        info["opencv_threads"] = cv2.getNumThreads()
    except Exception:
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "hardware", "model\t")):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    for f in ("/etc/nv_tegra_release", "/proc/device-tree/model"):
        try:
            info["jetson_" + os.path.basename(f)] = open(f, "rb").read().decode(errors="ignore").strip("\x00\n ")
        except OSError:
            pass
    try:
        import subprocess
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout.strip()
        if out:
            info["nvpmodel"] = out.replace("\n", " | ")
    except Exception:
        pass
    return info


def find_inputs(paths):
    vids, csvs = [], []
    for p in map(Path, paths):
        items = sorted(p.rglob("*")) if p.is_dir() else [p]
        for it in items:
            if it.suffix.lower() in VIDEO_EXT:
                vids.append(it)
            elif it.name.endswith("_data.csv") or (it.suffix == ".csv" and it.name != "truth.csv"
                                                   and "roi_diff" in open(it).readline()):
                csvs.append(it)
    return vids, csvs


def get_roi(video, args):
    import stream_features as sf
    if args.roi:
        return tuple(int(v) for v in args.roi.split(","))
    import cv2
    if args.ts_box:
        x0, y0, x1, y1 = (int(v) for v in args.ts_box.split(","))
        box = ((x0, y0), (x1, y1))
    else:
        from video_generator_timestamp import TS_BOX as box
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    picks = []
    for k in np.linspace(0, min(n, 300) - 1, 40).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, fr = cap.read()
        if ok:
            picks.append(fr)
    cap.release()
    return sf.locate_roi_from_frames(picks, box, src_size=(w, h))


# ------------------------------------------------------------------ stage 1: extraction
def extract(video, roi, max_frames=None):
    """One pass over a video. Returns (rows, decode_ms[], feature_ms[], sampler)."""
    import cv2
    import stream_features as sf
    fs = sf.FeatureStream(roi)
    cap = cv2.VideoCapture(str(video))
    dec, feat, meta, back = [], [], [], {}
    with Sampler() as s:
        while max_frames is None or len(dec) < max_frames:
            t0 = time.perf_counter()
            ok, fr = cap.read()
            t1 = time.perf_counter()
            if not ok:
                break
            r = fs.update(fr)
            cols_row = r["row_diff"]
            if r["backfill"]:
                back.update(r["backfill"])
            c = sf.csv_columns(r["roi_diff"], r["roi_lit"], cols_row, r["row_diff_ff"], r["index"])
            t2 = time.perf_counter()
            dec.append((t1 - t0) * 1e3)
            feat.append((t2 - t1) * 1e3)
            meta.append((r, c))
        back.update(fs.finish())
    cap.release()
    rows = []
    for k, (r, c) in enumerate(meta):
        if r["row_diff"] is None and k in back:
            b = float(np.percentile(back[k], 20))
            c[3], c[4] = b, float(np.percentile(back[k], 90)) - b
        rows.append(c)
    return rows, np.array(dec), np.array(feat), s, cap


def to_df(rows):
    import stream_features as sf
    return pd.DataFrame(rows, columns=sf.CSV_HEADER)


# ------------------------------------------------------------------ stage 2: analysis
def analyse(df, fps, name=""):
    import catalog
    import emc_verdict
    import general_detector
    t = {}
    with Sampler(0.02) as s:
        a = time.perf_counter()
        cat = catalog.build_catalog(df, fps, name)
        b = time.perf_counter()
        g = general_detector.detect(df, tuple(cat["reference_window"]), cat["events"], fps)
        c = time.perf_counter()
        res = emc_verdict.grade(cat, df=df)
        d = time.perf_counter()
    t.update(catalog_ms=(b - a) * 1e3, general_ms=(c - b) * 1e3, grade_ms=(d - c) * 1e3,
             total_ms=(d - a) * 1e3, grade_incl_general_ms=(d - c) * 1e3)
    return cat, g, res, t, s


# ------------------------------------------------------------------ stage 3: replay
def replay(df, fps):
    """Feed the rows to the live monitor one by one, timing each push (the periodic re-analysis dominates)."""
    from live_monitor import OnlineMonitor
    mon = OnlineMonitor(fps=fps)
    cols = ["roi_diff", "roi_lit", "baseline", "spread", "baseline_ff", "spread_ff", "frac_ff"]
    arr = df[cols].to_numpy(float)
    ms, trig, msgs = [], [], []
    for i, row in enumerate(arr):
        t0 = time.perf_counter()
        m = mon.push(*row)
        dt = (time.perf_counter() - t0) * 1e3
        ms.append(dt)
        trig.append(mon.n >= mon.min_frames and mon.n % mon.interval == 0)
        msgs += [(i, x) for x in m]
    t0 = time.perf_counter()
    msgs += [(len(arr), x) for x in mon.finish()]
    final_ms = (time.perf_counter() - t0) * 1e3
    lat = []
    for i, x in msgs:
        if x["kind"] == "NEW":
            e = x["event"]
            lat.append(dict(label=x["event"]["label"], start_frame=e["start"], seen_frame=x["frame"],
                            latency_s=(x["frame"] - e["start"]) / fps, found_in_finish=e["found_in_finish"]))
    return np.array(ms), np.array(trig), final_ms, lat, mon


# ------------------------------------------------------------------ stage 4: scaling
def tiled(df, n):
    reps = int(np.ceil(n / len(df)))
    big = pd.concat([df] * reps, ignore_index=True).iloc[:n].copy()
    big["frame"] = np.arange(len(big))
    return big


def scaling(df, fps, lengths, repeats):
    import catalog
    import general_detector
    out = []
    for n in lengths:
        big = tiled(df, n)
        for r in range(repeats):
            with Sampler(0.02) as s:
                t0 = time.perf_counter()
                cat = catalog.build_catalog(big, fps, "scaling")
                t1 = time.perf_counter()
                general_detector.detect(big, tuple(cat["reference_window"]), cat["events"], fps)
                t2 = time.perf_counter()
            out.append(dict(frames=n, minutes=n / fps / 60, rep=r, catalog_ms=(t1 - t0) * 1e3,
                            general_ms=(t2 - t1) * 1e3, total_ms=(t2 - t0) * 1e3, peak_mb=s.peak_mb,
                            budget_ms=10 / fps * 1e3))
    return out


# ------------------------------------------------------------------ truth check (optional)
def truth_check(cats, truth_path, fps, tol_s=2.0):
    truth = pd.read_csv(truth_path)
    rows = []
    for name, cat in cats.items():
        T = truth[truth["file"].apply(lambda f: Path(f).name.replace("_data.csv", "") in name or name in str(f))]
        ev = [e for e in cat["events"] if not e["minor"]]
        used = set()
        for _, t in T.iterrows():
            hit = next((i for i, e in enumerate(ev) if i not in used
                        and not (e["end_s"] < t.start_s - tol_s or e["start_s"] > t.end_s + tol_s)), None)
            if hit is not None:
                used.add(hit)
            rows.append(dict(file=name, truth_type=t["type"], start_s=t.start_s, end_s=t.end_s,
                             detected=hit is not None,
                             detected_type=ev[hit]["type"] if hit is not None else "",
                             type_correct=hit is not None and ev[hit]["type"].lower() == str(t["type"]).lower()))
        for i, e in enumerate(ev):
            if i not in used:
                rows.append(dict(file=name, truth_type="", start_s=e["start_s"], end_s=e["end_s"],
                                 detected=False, detected_type=e["type"], type_correct=False, false_alarm=True))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="videos, *_data.csv files, or folders of them")
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--roi", help="x0,y0,x1,y1 timestamp ROI in camera pixels (skips screen location)")
    ap.add_argument("--ts-box", help="x0,y0,x1,y1 timestamp box on the source video (screen location)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--truth")
    ap.add_argument("--scaling-lengths", default="3000,6000,12000,24000,48000")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids, csvs = find_inputs(a.paths)
    if a.no_video:
        vids = []
    if not vids and not csvs:
        sys.exit("no videos or *_data.csv files found")
    hw = hardware()
    hw["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "hardware.json").write_text(json.dumps(hw, indent=2))
    print(f"machine: {hw.get('cpu_model', hw['machine'])}, {hw['cpu_count_logical']} cores, {hw['ram_gb']} GB RAM")
    print(f"{len(vids)} video(s), {len(csvs)} csv file(s)\n")

    ext_rows, frame_rows, tables = [], [], {}
    # ---- stage 1
    for v in vids:
        try:
            roi = get_roi(v, a)
        except Exception as exc:
            print(f"[skip] {v.name}: cannot locate the screen ({exc}); pass --roi")
            continue
        runs = []
        for r in range(a.repeats):
            rows, dec, feat, s, _ = extract(v, roi, a.max_frames)
            n = len(rows)
            runs.append(dict(frames=n, wall_s=s.wall, cpu_s=s.cpu_s, peak_mb=s.peak_mb,
                             decode_ms=dec.mean(), feature_ms=feat.mean(), feature_p95=pct(feat, 95),
                             feature_p99=pct(feat, 99), feature_max=feat.max(),
                             fps=n / s.wall, cpu_pct=s.mean_cpu_pct, start_mb=s.start_mb))
            if r == 0:
                tables[v.name] = to_df(rows)
                tables[v.name].to_csv(out / (v.stem + "_data.csv"), index=False)
                frame_rows += [dict(video=v.name, frame=i, decode_ms=d, feature_ms=f)
                               for i, (d, f) in enumerate(zip(dec, feat))]
            print(f"  extract {v.name} run {r + 1}/{a.repeats}: {runs[-1]['fps']:.1f} fps, "
                  f"feature {feat.mean():.2f} ms/frame, cpu {s.mean_cpu_pct:.0f}%, peak {s.peak_mb:.0f} MB")
        import cv2
        _c = cv2.VideoCapture(str(v))
        size = f"{int(_c.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(_c.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        _c.release()
        row = dict(video=v.name, size=size, frames=runs[0]["frames"], duration_s=runs[0]["frames"] / a.fps, roi=str(roi))
        for k in ("wall_s", "cpu_s", "peak_mb", "decode_ms", "feature_ms", "feature_p95", "feature_p99",
                  "feature_max", "fps", "cpu_pct"):
            m, h = ci95([r[k] for r in runs])
            row[k], row[k + "_ci95"] = m, h
        row["real_time_factor"] = row["fps"] / a.fps          # >1 = faster than real time
        row["repeats"] = a.repeats
        ext_rows.append(row)
    for c in csvs:
        tables.setdefault(c.name, pd.read_csv(c))
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(out / "extraction.csv", index=False)
        pd.DataFrame(frame_rows).to_csv(out / "frame_times.csv", index=False)

    # ---- stages 2 and 3
    an_rows, rep_rows, lat_rows, cats = [], [], [], {}
    for name, df in tables.items():
        df = df.replace([np.inf], np.nan) if False else df
        runs = []
        for r in range(a.repeats):
            cat, g, res, t, s = analyse(df, a.fps, name)
            runs.append(dict(**t, peak_mb=s.peak_mb))
        cats[name] = cat
        row = dict(file=name, frames=len(df), duration_s=len(df) / a.fps, grade=res["grade"],
                   events="; ".join(res["events"]), general_episodes=len(g))
        for k in ("catalog_ms", "general_ms", "total_ms", "peak_mb"):
            m, h = ci95([x[k] for x in runs])
            row[k], row[k + "_ci95"] = m, h
        row["analysis_x_realtime"] = row["duration_s"] * 1e3 / row["total_ms"]
        an_rows.append(row)
        ms, trig, final_ms, lat, mon = replay(df, a.fps)
        tr = ms[trig]
        rep_rows.append(dict(file=name, frames=len(df), push_mean_ms=ms[~trig].mean() if (~trig).any() else float("nan"),
                             analysis_calls=int(trig.sum()), analysis_mean_ms=tr.mean() if len(tr) else float("nan"),
                             analysis_p95_ms=pct(tr, 95), analysis_max_ms=tr.max() if len(tr) else float("nan"),
                             last_analysis_ms=tr[-1] if len(tr) else float("nan"),
                             budget_ms=mon.interval / a.fps * 1e3, final_pass_ms=final_ms,
                             calls_over_budget=int((tr > mon.interval / a.fps * 1e3).sum())))
        for L in lat:
            lat_rows.append(dict(file=name, **L))
        print(f"  analysis {name}: {row['total_ms']:.0f} ms for {row['duration_s']:.0f} s of video, grade {res['grade']}")

    pd.DataFrame(an_rows).to_csv(out / "analysis.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(out / "replay.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(out / "latency.csv", index=False)

    # ---- stage 4
    longest = max(tables.values(), key=len)
    sc = scaling(longest, a.fps, [int(x) for x in a.scaling_lengths.split(",")], max(2, a.repeats))
    pd.DataFrame(sc).to_csv(out / "scaling.csv", index=False)

    if a.truth:
        truth_check(cats, a.truth, a.fps).to_csv(out / "truth_check.csv", index=False)

    write_summary(out, hw, a)
    print(f"\nDone. Results in {out}/  (send the whole folder back; summary.md is the quick read)")


def write_summary(out, hw, a):
    L = ["# Benchmark summary", "", "## Machine", ""]
    L += [f"- {k}: {v}" for k, v in hw.items()]
    if (out / "extraction.csv").exists():
        e = pd.read_csv(out / "extraction.csv")
        L += ["", "## Feature extraction (video -> features)", "",
              "| video | frames | fps | real-time factor | feature ms/frame (95% CI) | p95 ms | p99 ms | max ms | CPU % of one core | peak MB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in e.iterrows():
            L.append(f"| {r.video} | {r.frames:.0f} | {r.fps:.1f} | {r.real_time_factor:.2f} | "
                     f"{r.feature_ms:.2f} +/- {r.feature_ms_ci95:.2f} | {r.feature_p95:.2f} | {r.feature_p99:.2f} | "
                     f"{r.feature_max:.1f} | {r.cpu_pct:.0f} | {r.peak_mb:.0f} |")
    an = pd.read_csv(out / "analysis.csv")
    L += ["", "## Whole-recording analysis (catalog + general detector + grade)", "",
          "| file | frames | total ms (95% CI) | x real time | peak MB | grade |", "|---|---|---|---|---|---|"]
    for _, r in an.iterrows():
        L.append(f"| {r.file} | {r.frames:.0f} | {r.total_ms:.0f} +/- {r.total_ms_ci95:.0f} | "
                 f"{r.analysis_x_realtime:.0f} | {r.peak_mb:.0f} | {r.grade} |")
    rp = pd.read_csv(out / "replay.csv")
    L += ["", "## Live monitor replay (re-analysis every 10 frames)", "",
          "| file | frames | calls | mean ms | p95 ms | max ms | last ms | budget ms | over budget |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in rp.iterrows():
        L.append(f"| {r.file} | {r.frames:.0f} | {r.analysis_calls:.0f} | {r.analysis_mean_ms:.1f} | {r.analysis_p95_ms:.1f} | "
                 f"{r.analysis_max_ms:.1f} | {r.last_analysis_ms:.1f} | {r.budget_ms:.0f} | {r.calls_over_budget:.0f} |")
    lt = pd.read_csv(out / "latency.csv")
    if len(lt):
        L += ["", "## Detection latency (stream time between fault start and first report)", "",
              "| type | n | median s | min s | max s |", "|---|---|---|---|---|"]
        for k, g in lt.groupby("label"):
            L.append(f"| {k} | {len(g)} | {g.latency_s.median():.2f} | {g.latency_s.min():.2f} | {g.latency_s.max():.2f} |")
    sc = pd.read_csv(out / "scaling.csv")
    L += ["", "## Analysis cost against recording length (tiled real data - synthetic length)", "",
          "| frames | minutes | analysis ms (mean) | peak MB | budget ms |", "|---|---|---|---|---|"]
    for n, g in sc.groupby("frames"):
        L.append(f"| {n} | {g.minutes.iloc[0]:.1f} | {g.total_ms.mean():.0f} | {g.peak_mb.mean():.0f} | {g.budget_ms.iloc[0]:.0f} |")
    L += ["", f"Repeats per measurement: {a.repeats}. +/- is the 95% confidence interval of the mean across repeats."]
    (out / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
