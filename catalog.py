"""catalog.py - turn one *_data.csv into a catalog of separate fault events.

    python catalog.py recording.mp4               # runs collect_data first, then catalogs the result
    python catalog.py recording_data.csv
    python catalog.py folder/ --json catalog.json --csv catalog.csv     # videos and/or *_data.csv

What it adds on top of main.py / the individual detectors:
  1. EPISODE SPLITTING for blackout/flicker. classify_faults gives ONE verdict per file, so a
     blackout and a flicker in the same video were merged. Here the edges it finds are parsed
     one by one: a run of closely spaced edges is a FLICKER episode, an isolated edge plus the
     next edge a long gap later is a BLACKOUT, and anything left over is SINGLE TRANSITION /
     UNCERTAIN.
  2. REFERENCE CHECK. The detectors measure "normal" on frames 100-599. If that window is not
     quiet (a fault starts early), an earlier/shorter quiet window is used instead, and a
     warning is recorded.
  3. A CATALOG: id, type, start/end frame and time, detector, evidence, flags - plus the stretches
     where nothing was detected. Stray edges inside another event are kept as "notes", not events.

Assumes faults happen one after another, not at the same time.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import classify_faults
import classify_sustained
import detect_freeze
import detect_restart

ALL_DETECTORS = ("faults", "sustained", "freeze", "restart")


class _Skip(Exception):
    pass


REF_DEFAULT = (100, 600)
REF_LENGTHS = (500, 400, 300, 250, 200, 150)   # tried in this order for each start frame
MIN_REF = 150
END_TAIL = 120     # frames: a lone edge this close to the end of the data is only a note


# --------------------------------------------------------------------------- reference window
def _arrays(df):
    """The four columns _quiet reads, as numpy arrays (built once per recording; the pandas version
    rebuilt Series for every candidate window, which made noisy recordings 100x slower)."""
    return {c: df[c].to_numpy(float) for c in ("spread_ff", "baseline_ff", "roi_diff", "roi_lit")}


def _quiet(df, a, b, max_bad=0.15, arr=None):
    # max_bad: a short harmless bump (flicker_severe has ~9% in frames 161-216) must not disqualify
    # the window; a real fault inside it affects far more (tearing/warp: 60%+, freeze/blackout fail
    # the roi_diff / roi_lit checks below).
    arr = arr or _arrays(df)
    b = min(b, len(arr["roi_diff"]))
    if b - a < MIN_REF:
        return False
    sp, bf = arr["spread_ff"][a:b], arr["baseline_ff"][a:b]
    if ((sp > 1.0) | (bf > 2.0)).mean() > max_bad:
        return False
    if (bf > 2.0).sum() >= 2:       # a blackout/flicker edge (baseline_ff 5-16 for a few frames)
        return False
    h = (b - a) // 2
    rd, rl = arr["roi_diff"][a:b], arr["roi_lit"][a:b]
    r1, r2 = np.nanmedian(rd[:h]), np.nanmedian(rd[h:])
    if not (0.7 <= r2 / max(r1, 1e-9) <= 1.4):          # roi_diff level changes inside the window
        return False
    if abs(np.nanmedian(rl[:h]) - np.nanmedian(rl[h:])) > 0.1:
        return False
    return True


def choose_reference_window(df):
    """(window, warning). Uses frames 100-599 when they are quiet, otherwise the first quiet
    stretch of at least 150 frames."""
    n = len(df)
    arr = _arrays(df)
    a, b = REF_DEFAULT[0], min(REF_DEFAULT[1], n)
    if _quiet(df, a, b, arr=arr):
        return (a, b), None
    for s in range(100, max(n - MIN_REF, 100), 10):
        for L in REF_LENGTHS:
            if s + L <= n and _quiet(df, s, s + L, arr=arr):
                return (s, s + L), (f"frames {a}-{b} are not quiet (a fault starts early?); "
                                    f"reference taken from frames {s}-{s + L}")
    return (a, b), "no quiet reference window found; results may be unreliable"


# --------------------------------------------------------------------------- edge parsing
def parse_edges(edges, cfg=None):
    """Split classify_faults edges into episodes. Returns a list of (kind, [edges])."""
    cfg = cfg or classify_faults.CFG
    link = cfg["blackout_min_gap"]            # edges closer than this may belong to one flicker
    n = len(edges)
    out, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and edges[j + 1]["start"] - edges[j]["start"] < link:
            j += 1
        chain = edges[i:j + 1]
        gaps = np.diff([e["start"] for e in chain]) if len(chain) > 1 else np.array([])
        if len(chain) >= cfg["flicker_min_edges"] and np.median(gaps) <= cfg["flicker_max_med_gap"]:
            out.append(("FLICKER", chain))
            i = j + 1
        elif len(chain) == 1 and i + 1 < n:    # lone edge, long gap after it -> blackout start
            out.append(("BLACKOUT", [edges[i], edges[i + 1]]))
            i += 2
        else:
            out.append(("SINGLE TRANSITION" if len(chain) == 1 else "UNCERTAIN", chain))
            i = j + 1
    # A flicker chain directly followed by a lone edge far away: the chain's last edge was really
    # the START of a blackout that ends at the lone edge (flicker -> blackout with little quiet time).
    for k in range(len(out) - 1):
        kind, chain = out[k]
        nxt_kind, nxt = out[k + 1]
        if kind == "FLICKER" and nxt_kind == "SINGLE TRANSITION" and len(chain) - 1 >= cfg["flicker_min_edges"]:
            rest = chain[:-1]
            if (nxt[0]["start"] - chain[-1]["start"] >= link
                    and np.median(np.diff([e["start"] for e in rest])) <= cfg["flicker_max_med_gap"]):
                out[k] = ("FLICKER", rest)
                out[k + 1] = ("BLACKOUT", [chain[-1], nxt[0]])
    return out


# --------------------------------------------------------------------------- catalog
def _event(typ, start, end, detector, evidence, **flags):
    return dict(type=typ, start=int(start), end=int(end), detector=detector, evidence=evidence, **flags)


def build_catalog(df, fps=30.0, name="", detectors=ALL_DETECTORS):
    """detectors: which of "faults" (blackout/flicker), "sustained" (tearing/warp), "freeze",
    "restart" to run - the live monitor runs only "restart" while it is still warming up."""
    df = df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    n = len(df)
    warnings, events = [], []
    if {"faults", "sustained", "freeze"} & set(detectors):
        window, warn = choose_reference_window(df)
        if warn:
            warnings.append(warn)
    else:
        window = REF_DEFAULT

    # ---- blackout / flicker via edge episodes
    exclude = []
    try:
        if "faults" not in detectors:
            raise _Skip
        cfg = dict(classify_faults.CFG, ref_window=window)
        rd = df["baseline_ff"].to_numpy(float)
        thr = classify_faults.spike_threshold(rd, cfg)
        edges, blips = classify_faults.cluster_spikes(rd, thr, cfg)
        classify_faults.add_context(edges, df["roi_lit"].to_numpy(float), df["spread"].to_numpy(float), cfg)
        for kind, chain in parse_edges(edges, cfg):
            s, e = chain[0]["start"], chain[-1]["end"]
            if kind == "BLACKOUT":
                gap = chain[1]["start"] - chain[0]["start"]
                events.append(_event("BLACKOUT", s, e, "classify_faults",
                                     f"2 edges {gap} frames apart"))
                exclude += [(c["start"], c["end"]) for c in chain]
            elif kind == "FLICKER":
                gaps = np.diff([c["start"] for c in chain])
                events.append(_event("FLICKER", s, e, "classify_faults",
                                     f"{len(chain)} edges, median gap {np.median(gaps):.0f} frames"))
                exclude += [(c["start"], c["end"]) for c in chain]
            elif kind == "SINGLE TRANSITION" and s >= len(df) - END_TAIL:
                # a lone edge in the last few seconds can't be confirmed (a real blackout would need a
                # second edge later) and is usually the recording ending: keep it as a note, not a fault
                ev = _event(kind, s, e, "classify_faults",
                            "1 edge near the very end of the recording - probably the video ending")
                ev["minor"] = True
                events.append(ev)
            else:
                events.append(_event(kind, s, e, "classify_faults",
                                     f"{len(chain)} edge(s) that fit neither pattern", minor_candidate=True))
    except _Skip:
        pass
    except Exception as exc:
        warnings.append(f"classify_faults failed: {exc!r}")

    # ---- tearing / warp (uses the reference window chosen above)
    old_ref = classify_sustained.ROI_REF
    classify_sustained.ROI_REF = window
    try:
        for r in (classify_sustained.classify(df, exclude) if "sustained" in detectors else []):
            ev = f"median spread_ff {r['spread_ff']}, frac_ff {r['frac_ff']}, roi_diff x{r.get('roi_x')}"
            events.append(_event(r["kind"].upper(), r["start"], r["end"], "classify_sustained", ev))
    except Exception as exc:
        warnings.append(f"classify_sustained failed: {exc!r}")
    finally:
        classify_sustained.ROI_REF = old_ref

    # ---- freeze
    try:
        cfg = dict(detect_freeze.CFG, fps=fps, ref_window=window)
        for r in (detect_freeze.find_freezes(df, cfg) if "freeze" in detectors else []):
            events.append(_event("FREEZE", r["start"], r["end"], "detect_freeze",
                                 f"roi_diff {r['level']:.2f} vs normal {r['normal']:.2f}"))
    except Exception as exc:
        warnings.append(f"detect_freeze failed: {exc!r}")

    # ---- restart
    try:
        cfg = dict(detect_restart.CFG, fps=fps)
        for r in (detect_restart.find_restarts(df, cfg) if "restart" in detectors else []):
            events.append(_event("RESTART", r["start"], r["end"], "detect_restart",
                                 f"roi_diff {r['peak_roi_diff']:.2f}, spread_ff {r['peak_spread_ff']:.2f}"))
    except Exception as exc:
        warnings.append(f"detect_restart failed: {exc!r}")

    # ---- order, flag stray edges inside another event as notes, assign ids
    events.sort(key=lambda e: (e["start"], e["end"]))
    majors = [e for e in events if not e.get("minor_candidate")]
    for e in events:
        if e.pop("minor_candidate", False):
            host = next((m for m in majors if m["start"] <= e["start"] and e["end"] <= m["end"]), None)
            e["minor"] = host is not None
            if host is not None:
                e["within"] = id(host)
    k = m = 0
    for e in events:
        if e.get("minor"):
            m += 1
            e["id"] = f"n{m}"
        else:
            k += 1
            e["id"] = f"E{k}"
    ids = {id(e): e["id"] for e in events}
    for e in events:
        if "within" in e:
            e["within"] = ids[e["within"]]
        e.setdefault("minor", False)
        e["start_s"], e["end_s"] = round(e["start"] / fps, 2), round((e["end"] + 1) / fps, 2)
        e["dur_s"] = round(e["end_s"] - e["start_s"], 2)

    # ---- stretches with nothing detected
    quiet_runs, cur = [], 0
    for e in sorted((x for x in events if not x["minor"] and x["type"] != "RESTART"), key=lambda x: x["start"]):
        if e["start"] > cur:
            quiet_runs.append((cur, e["start"] - 1))
        cur = max(cur, e["end"] + 1)
    if cur < n:
        quiet_runs.append((cur, n - 1))

    return dict(file=name, frames=n, fps=fps, duration_s=round(n / fps, 2),
                reference_window=list(window), warnings=warnings, events=events,
                nothing_detected=[dict(start=a, end=b, start_s=round(a / fps, 2),
                                       end_s=round((b + 1) / fps, 2)) for a, b in quiet_runs if b - a >= 30])


# --------------------------------------------------------------------------- output
def format_catalog(cat):
    out = [f"=== {cat['file']}   {cat['frames']} frames ({cat['duration_s']} s)   "
           f"reference frames {cat['reference_window'][0]}-{cat['reference_window'][1]}"]
    for w in cat["warnings"]:
        out.append(f"  ! {w}")
    major = [e for e in cat["events"] if not e["minor"]]
    if not major:
        out.append("  no faults detected")
    for e in major:
        out.append(f"  {e['id']:<4} {e['type']:<16} {e['start_s']:>6.1f}s - {e['end_s']:>6.1f}s "
                   f"({e['dur_s']:>5.1f}s)  f{e['start']}-{e['end']}   {e['evidence']}")
    for e in cat["events"]:
        if e["minor"]:
            out.append(f"  {e['id']:<4} note: {e['type'].lower()} at f{e['start']}-{e['end']}"
                       + (f" inside {e['within']}" if e.get("within") else "") + f" ({e['evidence']})")
    tl = sorted([(e["start"], e["end"], e["type"]) for e in major if e["type"] != "RESTART"] +
                [(q["start"], q["end"], "ok") for q in cat["nothing_detected"]])
    if tl:
        out.append("  timeline: " + " | ".join(
            f"{t if t != 'ok' else 'nothing detected'} {a / cat['fps']:.0f}-{(b + 1) / cat['fps']:.0f}s"
            for a, b, t in tl))
    return "\n".join(out)


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def video_to_csv(video, extractor, recollect):
    """Run collect_data(video) (your feature extraction) and return the *_data.csv it writes.
    An existing CSV next to the video is reused unless --recollect is given."""
    out = video.with_name(video.stem + "_data.csv")
    if out.exists() and not recollect:
        print(f"[reuse] {out.name} already exists (use --recollect to rebuild it)")
        return out
    import importlib
    module = importlib.import_module(extractor)       # your collect_data.py
    module.collect_data(video)                         # it expects a pathlib.Path
    if not out.exists():
        raise FileNotFoundError(f"{extractor}.collect_data ran but {out.name} was not written")
    return out


def resolve_inputs(paths, extractor, recollect):
    """Videos (mp4/avi/mov/mkv/webm) and *_data.csv files, or folders of them -> list of CSV paths."""
    csvs, failed, seen = [], [], set()
    for p in map(Path, paths):
        in_dir = p.is_dir()
        for it in (sorted(p.iterdir()) if in_dir else [p]):
            try:
                if it.suffix.lower() in VIDEO_EXT:
                    c = video_to_csv(it, extractor, recollect)
                elif it.suffix.lower() == ".csv" and (not in_dir or it.name.endswith("_data.csv")):
                    c = it
                else:
                    continue
            except Exception as exc:                    # one bad video must not stop the rest
                print(f"[error] {it.name}: {exc!r}", file=sys.stderr)
                failed.append(it.name)
                continue
            if c not in seen:
                seen.add(c)
                csvs.append(c)
    return csvs, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="video file(s), *_data.csv file(s), or folder(s) of either")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--extractor", default="collect_data",
                    help="module whose collect_data(video) writes <video>_data.csv (default: collect_data)")
    ap.add_argument("--recollect", action="store_true", help="rebuild CSVs even if they already exist")
    ap.add_argument("--json", help="write the full catalogs to this JSON file")
    ap.add_argument("--csv", help="write one row per event to this CSV file")
    a = ap.parse_args()
    files, failed = resolve_inputs(a.paths, a.extractor, a.recollect)
    cats = []
    for f in files:
        cat = build_catalog(pd.read_csv(f), a.fps, f.name)
        cats.append(cat)
        print(format_catalog(cat) + "\n")
    if failed:
        print("COULD NOT PROCESS:", ", ".join(failed))
    if a.json:
        Path(a.json).write_text(json.dumps(cats, indent=2))
    if a.csv:
        rows = [dict(file=c["file"], **{k: v for k, v in e.items()}) for c in cats for e in c["events"]]
        pd.DataFrame(rows).to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
