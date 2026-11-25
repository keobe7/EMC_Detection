#!/usr/bin/env python3
"""live_benchmark.py - one command: run live.py, measure its CPU and memory while it runs, then
benchmark the recording it made and compare the two.

    python live_benchmark.py --source csi --roi 390,151,576,180 --duration 120
    python live_benchmark.py --source 0 --roi 390,151,576,180 --duration 60 --out bench_live
    python live_benchmark.py --source test.mp4 --roi 20,20,276,63 --duration 20      # a file works too

Steps
  1. starts  live.py --no-preview --record <out>/live_rec.mp4  (your --source/--roi/--width/... passed on)
  2. samples the live.py process (and its children) every 0.5 s: CPU % of one core, memory (RSS);
     on a Jetson also samples `tegrastats` (RAM, CPU load, GPU load) when it is installed
  3. after --duration seconds sends Ctrl+C so live.py prints its EMC grade and saves
  4. counts the recorded frames against wall time x fps -> frames dropped (the real "keeping up" test)
  5. runs benchmark.py on the recording (same ROI) and puts live vs offline side by side

Writes <out>/live_summary.md and <out>/live_samples.csv; benchmark files go to <out>/offline/.
Needs psutil (pip install psutil). Keep this file next to live.py and benchmark.py.
"""
import argparse
import csv
import os
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import psutil

HERE = Path(__file__).resolve().parent


def tegra_reader(stop, store):
    """Parse tegrastats lines (Jetson only): RAM used MB and mean CPU load %."""
    exe = shutil.which("tegrastats")
    if not exe:
        return
    try:
        p = subprocess.Popen([exe, "--interval", "1000"], stdout=subprocess.PIPE, text=True)
    except Exception:
        return
    while not stop.is_set():
        line = p.stdout.readline()
        if not line:
            break
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        c = re.search(r"CPU \[([^\]]*)\]", line)
        g = re.search(r"GR3D_FREQ (\d+)%", line)
        row = {}
        if m:
            row["ram_used_mb"] = int(m.group(1))
        if c:
            loads = [int(x.split("%")[0]) for x in c.group(1).split(",") if "%" in x]
            if loads:
                row["cpu_mean_pct"] = sum(loads) / len(loads)
        if g:
            row["gpu_pct"] = int(g.group(1))
        if row:
            store.append(row)
    p.terminate()


def proc_tree(p):
    try:
        return [p] + p.children(recursive=True)
    except psutil.Error:
        return [p]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--roi", required=True, help="x0,y0,x1,y1 in camera pixels (needed so no key press is wanted)")
    ap.add_argument("--warmup", type=float, default=10.0, help="camera warm-up seconds thrown away before recording (default 10)")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of DETECTION (counted from the line \"started\", after the warm-up) before it stops")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--out", default="bench_live")
    ap.add_argument("--repeats", type=int, default=3, help="repeats for the offline benchmark")
    ap.add_argument("--no-record", action="store_true",
                    help="do not save the frames (writing the mp4 costs CPU time on the Jetson): measures live speed "
                         "without the recording overhead; no offline benchmark then")
    ap.add_argument("--skip-offline", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="show the live.py window (needs a display); press SPACE/ENTER in it to start. Adds drawing cost, so use it for set-up, not for the numbers you report")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = out / "live_rec.mp4"
    cmd = [sys.executable, "-u", str(HERE / "live.py"), "--source", a.source, "--roi", a.roi,
           "--fps", str(a.fps), "--out", str(out / "live_events.csv")]
    if not a.preview:
        cmd.insert(cmd.index("--roi"), "--no-preview")
    if not a.no_record:
        cmd += ["--record", str(rec)]
    if a.warmup > 0:
        cmd += ["--warmup", str(a.warmup)]
    if a.width:
        cmd += ["--width", str(a.width)]
    if a.height:
        cmd += ["--height", str(a.height)]
    print("running:", " ".join(cmd))

    t0 = time.time()
    log = open(out / "live_stdout.txt", "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    started_at = []

    def tee():                      # show live.py's output here AND keep it in live_stdout.txt
        for line in proc.stdout:
            log.write(line); log.flush()
            print(line, end="", flush=True)
            if line.startswith("started, ROI"):
                started_at.append(time.time())
                print("\n>>> DETECTING NOW - start playing the test video from its beginning <<<\n", flush=True)
    threading.Thread(target=tee, daemon=True).start()
    ps = psutil.Process(proc.pid)
    samples, tegra, stop = [], [], threading.Event()
    threading.Thread(target=tegra_reader, args=(stop, tegra), daemon=True).start()
    ncpu = psutil.cpu_count() or 1
    for p in proc_tree(ps):
        p.cpu_percent(None)
    stopped = False
    while proc.poll() is None:
        time.sleep(0.5)
        el = time.time() - t0
        cpu = rss = 0.0
        for p in proc_tree(ps):
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        samples.append((round(el, 1), round(cpu, 1), round(rss / 2**20, 1)))
        run_el = (time.time() - started_at[0]) if started_at else 0.0     # --duration counts from "started"
        if run_el >= a.duration and not stopped:
            print(f"{a.duration:g} s of detection reached - sending Ctrl+C to live.py")
            proc.send_signal(signal.SIGINT)
            stopped = True
        if el >= a.duration + a.warmup + 60:
            proc.kill()
            break
    stop.set()
    time.sleep(0.5)
    wall = time.time() - t0
    log.close()
    # live.py saves the per-frame features in the folder it was started from; copy that file next to the results
    try:
        m = re.search(r"saved the session's features to (\S+_data\.csv)", (out / "live_stdout.txt").read_text(errors="ignore"))
        if m and Path(m.group(1)).exists():
            shutil.copy(m.group(1), out / "features.csv")
            print("copied the features file to", out / "features.csv")
    except Exception as ex:
        print("could not copy the features file:", ex)
    with open(out / "live_samples.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", "cpu_pct_of_one_core", "rss_mb"]); w.writerows(samples)

    # frames kept vs expected
    frames = 0
    if rec.exists():
        cap = cv2.VideoCapture(str(rec)); frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    cpu_s = [s[1] for s in samples[2:]] or [0]
    rss_s = [s[2] for s in samples] or [0]
    lines = ["# Live benchmark", "", f"- command: `{' '.join(cmd)}`", f"- wall time: {wall:.1f} s",
             f"- machine: {psutil.cpu_count(logical=True)} logical cores"]
    lines += ["", "## While live.py ran (no preview)", "",
              f"- CPU of one core: mean {st.mean(cpu_s):.0f}%, max {max(cpu_s):.0f}%  "
              f"(= {st.mean(cpu_s) / ncpu:.0f}% of the whole machine)",
              f"- memory (RSS): mean {st.mean(rss_s):.0f} MB, peak {max(rss_s):.0f} MB"]
    if tegra:
        def col(k):
            v = [r[k] for r in tegra if k in r]
            return v
        for k, label in (("cpu_mean_pct", "tegrastats CPU load (all cores)"), ("ram_used_mb", "tegrastats RAM used (MB, whole system)"),
                         ("gpu_pct", "tegrastats GPU load")):
            v = col(k)
            if v:
                lines.append(f"- {label}: mean {st.mean(v):.0f}, max {max(v):.0f}")
    else:
        lines.append("- tegrastats not found (not a Jetson, or not installed): system-wide Jetson load not recorded")
    lines += ["", "## Did it keep up?", ""]
    pr = [l for l in open(out / "live_stdout.txt").read().splitlines() if l.startswith("processed ")]
    if pr:
        lines.append("- " + pr[-1])
    if frames:
        active = max(wall - a.warmup - 3.0, 1.0)
        lines.append(f"- frames recorded: {frames} in about {active:.0f} s of processing "
                     f"-> {frames / active:.1f} fps (camera asked for {a.fps:g} fps)")
        lines.append("- if this is clearly below the camera fps, the program is not keeping up (frames are dropped); "
                     "the start-up time is approximate, so compare over a long run")
    else:
        lines.append("- no recording was written - see live_stdout.txt")
    if not a.skip_offline and not a.no_record and rec.exists():
        print("benchmarking the recording ...")
        off = out / "offline"
        subprocess.run([sys.executable, str(HERE / "benchmark.py"), str(rec), "--roi", a.roi,
                        "--repeats", str(a.repeats), "--out", str(off)], check=False)
        lines += ["", "## Offline benchmark of the same recording", "", f"see `{off}/summary.md` (extraction ms per frame, CPU, memory)"]
        try:
            import pandas as pd
            e = pd.read_csv(off / "extraction.csv").iloc[0]
            lines.append(f"- offline: {e.feature_ms:.2f} ms/frame features, {e.cpu_pct:.0f}% of one core while flat out, "
                         f"peak {e.peak_mb:.0f} MB")
        except Exception:
            pass
    lines += ["", "## Last lines of live.py output", "", "```", *open(out / "live_stdout.txt").read().strip().splitlines()[-15:], "```"]
    (out / "live_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:22]))
    print("\nwrote", out / "live_summary.md")


if __name__ == "__main__":
    main()
