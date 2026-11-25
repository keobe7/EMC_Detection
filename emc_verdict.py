"""emc_verdict.py - turn the fault catalog into an EMC performance grade: A, B, or C/D (manual check).

Basis: CISPR 35 clause 9 (EN 55035). Paraphrased:
    A  equipment keeps working as intended, no degradation below the specified performance level
    B  degradation allowed DURING the disturbance; after it the equipment must work as intended again,
       with no operator action and no lasting change of state
    C  loss of function allowed if it is self-recoverable or restorable by the user (a reset or restart
       is permitted)
CISPR 35 defines only A, B and C - there is no criterion D. Here anything that is not clearly A or B
goes to one bucket, "C/D - MANUAL CHECK", for the operator.

How the catalog is mapped (my reading of the standard, not something the standard spells out for video -
Annex D, which covers displays, was not available to me; check the rules in RULES below against it):
    nothing detected (classifiers AND the general detector) -> A
    only degradation faults (flicker, tearing,
      warp, noise band), all ended, and the picture
      was healthy again for `recovery_s` seconds  -> B
    anything else: loss of function (blackout, restart, freeze), a fault still going at the end of the
      recording, a fault that starts after the test window, a lone unexplained edge, or no usable
      healthy reference, or anything outside healthy that matches no known fault (general_detector.py)
                                                  -> C/D, operator checks by hand

Limits you should state when you use it:
  * Events that end inside the first 100 frames (camera start-up) are not graded; a real fault there is missed.
  * "A" means "the detectors saw nothing". The detectors miss the lowest severities, so A is weaker than
    "no degradation at all".
  * Without --test-window the whole recording counts as "during the test" and only the recovery check
    at the end is applied. Give the window if you know when the disturbance was applied.
  * The thresholds (recovery_s, settle_s) are defaults I chose; set them from your test plan.

    python emc_verdict.py recording_data.csv
    python emc_verdict.py folder/ --test-window 20,40 --recovery 10
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import catalog
import general_detector

GRADE_A = "A"
GRADE_B = "B"
GRADE_MANUAL = "C/D - MANUAL CHECK"

# which catalog event types count as what. Anything not listed is treated as loss of function (safe side).
DEGRADATION = {"FLICKER", "TEARING", "GEOMETRIC_WARP", "WARP", "NOISE"}      # picture degraded, function still there  -> B
LOSS_OF_FUNCTION = {"BLACKOUT", "RESTART", "FREEZE"}       # function lost                            -> C/D
RULES = dict(DEGRADATION=sorted(DEGRADATION), LOSS_OF_FUNCTION=sorted(LOSS_OF_FUNCTION))


def grade(cat, test_window=None, recovery_s=10.0, settle_s=2.0, df=None):
    """cat: the dict from catalog.build_catalog. test_window: (start_s, end_s) of the disturbance or None.
    df: the feature table the catalog was built from; when given, the general detector also looks for
    anything outside healthy that matches no known fault (and any such thing means the operator checks).
    Returns dict(grade, reasons, events, notes)."""
    dur = cat["duration_s"]
    events = [e for e in cat["events"] if not e["minor"]]
    notes = [e for e in cat["events"] if e["minor"]]
    # camera start-up: anything that ends before the reference window starts (frames 0-99) is the camera
    # settling (exposure, white balance), not a fault. It is listed as a note, not graded.
    startup = [e for e in events if e["end"] < catalog.REF_DEFAULT[0]]
    events = [e for e in events if e not in startup]
    notes = notes + startup
    reasons, manual = [], []

    # no usable reference -> nothing can be judged
    if cat["warnings"]:
        for w in cat["warnings"]:
            if "no quiet reference" in w.lower() or "failed" in w.lower():
                manual.append(f"cannot judge reliably: {w}")
            elif "not quiet" in w.lower():
                manual.append(f"the first 20 s were not healthy, so the reference had to be moved and may already "
                              f"contain the fault ({w}) - record at least 20 s of healthy picture before the fault")
    if dur < (catalog.REF_DEFAULT[1] + 1) / cat["fps"]:
        manual.append(f"recording is only {dur:.1f}s, shorter than the reference window - no healthy baseline")

    anomalies = []
    if df is not None:
        try:
            anomalies = general_detector.detect(df, tuple(cat["reference_window"]), cat["events"], cat["fps"])
        except Exception as exc:                          # never let the extra check hide a result
            manual.append(f"general detector failed ({exc!r}) - check by hand")
    for g in general_detector.unclassified(anomalies):
        manual.append(f"unclassified anomaly {g['start_s']:.1f}-{g['end_s']:.1f}s: outside the healthy range "
                      f"({', '.join(g['columns'])}; score {g['peak']:g}) but matches no known fault")

    if not events and not manual:
        reasons.append("no fault detected over the whole recording" + (" (known faults and general check)" if df is not None else ""))
        return _result(GRADE_A, reasons, events, notes)

    for e in events:
        tag = f"{e['id']} {e['type']} {e['start_s']:.1f}-{e['end_s']:.1f}s"
        if e["type"] in LOSS_OF_FUNCTION or e["type"] not in DEGRADATION:
            manual.append(f"{tag}: loss of function or unrecognised fault - criterion C territory, operator decides")
        if e["end_s"] >= dur - 1.0 / cat["fps"] * 2:
            manual.append(f"{tag}: still going when the recording ends - not shown to recover")

    if events:
        last_end = max([e["end_s"] for e in events] + [g["end_s"] for g in anomalies if g["explained"]])
        if last_end < dur and dur - last_end < recovery_s and not any("still going" in m for m in manual):
            manual.append(f"only {dur - last_end:.1f}s of healthy picture after the last fault "
                          f"(need {recovery_s:g}s to call it recovered)")

    if test_window:
        a, b = test_window
        for e in events:
            if e["start_s"] > b + settle_s:
                manual.append(f"{e['id']} {e['type']} starts at {e['start_s']:.1f}s, after the test window "
                              f"ended ({b:g}s) - degradation after the disturbance is not allowed by B")
            elif e["start_s"] < a - settle_s:
                manual.append(f"{e['id']} {e['type']} starts at {e['start_s']:.1f}s, before the test window "
                              f"({a:g}s) - not caused by the disturbance? operator should check")

    if manual:
        return _result(GRADE_MANUAL, manual, events, notes)

    reasons.append(f"{len(events)} degradation event(s), all ended, then {dur - last_end:.1f}s "
                   f"of healthy picture")
    if test_window:
        reasons.append(f"all events inside the test window {test_window[0]:g}-{test_window[1]:g}s")
    else:
        reasons.append("no test window given: whole recording treated as 'during the test'")
    return _result(GRADE_B, reasons, events, notes)


def _result(g, reasons, events, notes):
    return dict(grade=g, reasons=reasons,
                events=[f"{e['id']} {e['type']} {e['start_s']:.1f}-{e['end_s']:.1f}s" for e in events],
                notes=[f"{n['id']} {n['type'].lower()} at {n['start_s']:.1f}s" for n in notes])


def format_grade(name, r):
    out = [f"{name}: {r['grade']}"]
    if r["events"]:
        out.append("  events: " + "; ".join(r["events"]))
    for x in r["reasons"]:
        out.append(f"  - {x}")
    if r["notes"]:
        out.append("  minor notes (not graded): " + "; ".join(r["notes"]))
    return "\n".join(out)


def parse_window(s):
    try:
        a, b = (float(x) for x in s.split(","))
        assert b > a >= 0
        return a, b
    except Exception:
        raise argparse.ArgumentTypeError("use --test-window START,END in seconds, e.g. 20,40")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="*_data.csv files, videos, or folders (same as main.py)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--test-window", type=parse_window, help="START,END seconds when the disturbance was applied")
    ap.add_argument("--recovery", type=float, default=10.0, help="seconds of healthy picture needed after the last fault")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds of slack around the test window")
    ap.add_argument("--extractor", default="collect_data")
    ap.add_argument("--recollect", action="store_true")
    ap.add_argument("--out", help="write one row per file (grade and reasons) to this CSV")
    a = ap.parse_args()

    files, failed = catalog.resolve_inputs(a.paths, a.extractor, a.recollect)
    if not files and not failed:
        sys.exit("nothing to process")
    rows = []
    for f in files:
        df_ = pd.read_csv(f)
        cat = catalog.build_catalog(df_, a.fps, f.name)
        r = grade(cat, a.test_window, a.recovery, a.settle, df=df_)
        print(format_grade(f.name, r) + "\n")
        rows.append(dict(file=f.name, grade=r["grade"], events="; ".join(r["events"]), reasons=" | ".join(r["reasons"])))
    for f in failed:
        print(f"{f}: {GRADE_MANUAL}\n  - could not be processed\n")
        rows.append(dict(file=str(f), grade=GRADE_MANUAL, events="", reasons="could not be processed"))
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
