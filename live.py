"""live.py - run the fault detectors on a camera, a stream or a video file, as it plays.

    python live.py                          # camera 0 (usually the built-in one)
    python live.py --source 1               # next camera (usually the external USB one)
    python live.py --source csi             # Jetson CSI camera (csi:1 = sensor 1; --width/--height, default 1280x720)
    python live.py --source "nvarguscamerasrc ! ... ! appsink"   # any GStreamer pipeline
    python live.py --source rtsp://...      # network stream
    python live.py --source recording.mp4   # a recording: paused at frame 0 until you press a key,
                                            # then played in real time (--speed 4 = faster)
    python live.py --roi 390,151,576,180    # skip automatic screen location
    python live.py --no-preview             # no window: starts as soon as the ROI is known

How a session goes
  1. SET-UP: a window shows what the camera sees. The program looks for the colorbar screen
     and draws the region of interest (ROI) as a box (green = found, red = not found yet).
     Point/adjust the camera until the box sits on the timestamp area.
  2. Press SPACE or ENTER to start detecting. Frames before that are thrown away; the frame you
     start on is frame 0. Keep the screen healthy for the next ~20 s: that is how the reference
     picture is built (frames 100-599). Restarts are already watched during that time.
  3. Faults are printed as they are found (and listed in the window). Keys while running:
         r  start over (new reference, same ROI)        q / ESC  stop and print the catalog
  Keys during set-up:  SPACE/ENTER start   l  locate the screen again   q / ESC quit

Options: --test-window START,END (seconds the disturbance was on; sharpens the EMC grade), --recovery SECONDS
(healthy time needed after the last fault for grade B, default 10), --fps (frame rate used for seconds, default 30), --width/--height (ask the camera for a
resolution), --log FILE (every message is appended there), --out FILE (final events as CSV),
--ts-box x0,y0,x1,y1 (timestamp box; default TS_BOX from video_generator_timestamp.py).

At the end it prints the EMC grade (A, B or C/D manual check; see emc_verdict.py).

Keep this file next to emc_verdict.py, stream_features.py, live_monitor.py, main.py, catalog.py, the detector
files, screen_locator.py and video_generator_timestamp.py.
"""
import argparse
import os
import signal
import sys
import time
from collections import deque

# Jetson: the OpenCV/GStreamer build needs this set before cv2 is imported (same as the recording script)
_NV_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(_NV_EGL):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NV_EGL)

import cv2
import numpy as np
import pandas as pd

import catalog
import emc_verdict
import stream_features as sf
from live_monitor import OnlineMonitor

SPACE, ENTER, ESC = 32, 13, 27
LOCATE_EVERY = 30      # frames between automatic ROI attempts during set-up
LOCATE_FRAMES = 40     # frames used for each attempt
KEEP = 5               # how many recent messages the window lists


# ---------------------------------------------------------------------------- display
class Window:
    """Thin wrapper so the loop does not care whether a window exists."""

    def __init__(self, enabled, title="live fault monitor", max_width=960):
        self.title, self.max_width = title, max_width
        self.enabled = enabled
        if enabled:
            try:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            except cv2.error:
                print("(no window available - opencv-python-headless? running without preview)")
                self.enabled = False

    def show(self, frame, lines, roi=None, roi_ok=True, wait=1):
        """Draw `lines` and the ROI on a copy of the frame; return the pressed key (or -1)."""
        if not self.enabled:
            return -1
        h, w = frame.shape[:2]
        k = min(1.0, self.max_width / w)
        img = cv2.resize(frame, None, fx=k, fy=k) if k < 1 else frame.copy()
        if roi is not None:
            x0, y0, x1, y1 = [int(v * k) for v in roi]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0) if roi_ok else (0, 0, 255), 2)
        y = 22
        for text, color in lines:
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            y += 24
        cv2.imshow(self.title, img)
        return cv2.waitKey(wait) & 0xFF

    def select(self, frame, prompt="drag a box over the TIMESTAMP area, then press ENTER (c = cancel)"):
        """Let the person draw the ROI by hand. Returns (x0, y0, x1, y1) in frame pixels or None."""
        if not self.enabled:
            return None
        h, w = frame.shape[:2]
        k = min(1.0, self.max_width / w)
        img = cv2.resize(frame, None, fx=k, fy=k) if k < 1 else frame.copy()
        cv2.putText(img, prompt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        x, y, bw, bh = cv2.selectROI(self.title, img, showCrosshair=False)
        if bw < 4 or bh < 4:
            return None
        return (int(x / k), int(y / k), int((x + bw) / k), int((y + bh) / k))

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()


WHITE, GREEN, RED, YELLOW = (255, 255, 255), (0, 220, 0), (60, 60, 255), (0, 220, 255)


# ---------------------------------------------------------------------------- source
def csi_pipeline(sensor_id=0, width=1280, height=720, fps=30):
    """The Jetson CSI camera pipeline (same as the recording script; BGR frames out)."""
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},framerate={int(fps)}/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false")


def open_source(arg, width=None, height=None, fps=30.0):
    text = str(arg)
    low = text.lower()
    if low == "csi" or low.startswith("csi:"):                   # Jetson CSI camera: csi, csi:1 (sensor id)
        sid = int(low.split(":")[1]) if ":" in low else 0
        cap = cv2.VideoCapture(csi_pipeline(sid, width or 1280, height or 720, fps), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            sys.exit("could not open the CSI camera (is another program using it? try: sudo systemctl restart nvargus-daemon)")
        return cap, True
    if "!" in text:                                              # a full GStreamer pipeline string
        cap = cv2.VideoCapture(text, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            sys.exit("could not open the GStreamer pipeline")
        return cap, True
    src = int(arg) if text.isdigit() else arg
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"could not open source {arg!r} (for a camera try --source 0, 1, 2 ...)")
    if isinstance(src, int):
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, isinstance(src, int)


def read(cap, is_camera, retries=30):
    """Next frame, or None at the end of a file / when a camera stops delivering."""
    for _ in range(retries if is_camera else 1):
        ok, frame = cap.read()
        if ok:
            return frame
        time.sleep(0.05)
    return None


def get_ts_box(text):
    if text:
        x0, y0, x1, y1 = [int(v) for v in text.split(",")]
        return ((x0, y0), (x1, y1))
    from video_generator_timestamp import TS_BOX
    return TS_BOX


# ---------------------------------------------------------------------------- phases
def setup_file(cap, win, ts_box, fixed_roi, preview):
    """Video file: do not play during set-up. Sample the first frames to find the ROI, rewind, show
    frame 0 paused and wait for the start key. Detection then begins at frame 0 of the file."""
    def sample():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        picks = []
        for i in range(LOCATE_FRAMES * 8):
            ok, f = cap.read()
            if not ok:
                break
            if i % 8 == 0:
                picks.append(f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return picks

    def first():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, f = cap.read()                     # leaves the file positioned on frame 1
        return f if ok else None

    roi, status = fixed_roi, "ROI fixed by --roi"
    if fixed_roi is None:
        picks = sample()
        h, w = picks[0].shape[:2] if picks else (0, 0)
        try:
            roi = sf.locate_roi_from_frames(picks, ts_box, src_size=(w, h))
            status = f"screen found, ROI = {roi}"
        except RuntimeError as e:
            status = f"{e} - use --roi x0,y0,x1,y1"
    frame0 = first()
    if frame0 is None:
        print("could not read the first frame")
        return None, None
    print(status)
    if not preview:
        return (roi, frame0) if roi else (None, None)
    lines = [(status, GREEN if roi else RED),
             ("PAUSED at frame 0.  SPACE/ENTER = start    m = draw ROI by hand    q = quit", WHITE)]
    while True:
        key = win.show(frame0, lines, roi, roi_ok=roi is not None, wait=30)
        if key in (ord("q"), ESC):
            return None, None
        if key == ord("m"):
            hand = win.select(frame0)
            if hand:
                roi, status = hand, f"ROI drawn by hand = {hand}"
                lines[0] = (status, GREEN)
        if key in (SPACE, ENTER) and roi is not None:
            return roi, frame0


def setup_phase(cap, is_camera, win, ts_box, fixed_roi, preview, is_file=False):
    """Show the feed, find the ROI, wait for the start key. Returns (roi, first_frame_or_None)
    or (None, None) when the user quits / the source ends."""
    if is_file:
        return setup_file(cap, win, ts_box, fixed_roi, preview)
    recent = deque(maxlen=LOCATE_FRAMES * 3)
    roi, since, status = fixed_roi, 0, ("ROI fixed by --roi" if fixed_roi else "looking for the screen ...")
    frame, manual, last_cand = None, None, None
    while True:
        frame = read(cap, is_camera)
        if frame is None:
            print("source ended during set-up")
            return None, None
        recent.append(frame)
        since += 1
        if fixed_roi is None and since >= LOCATE_EVERY:
            since = 0
            if len(recent) >= LOCATE_FRAMES:
                pick = list(recent)[::max(1, len(recent) // LOCATE_FRAMES)][-LOCATE_FRAMES:]
                h, w = frame.shape[:2]
                try:
                    cand = sf.locate_roi_from_frames(pick, ts_box, src_size=(w, h))
                    if manual is None and last_cand and max(abs(a_ - b_) for a_, b_ in zip(cand, last_cand)) <= 8:
                        roi, status = cand, f"screen found, ROI = {cand}"
                    elif manual is None:
                        roi, status = None, "screen candidate - checking it stays put ..."
                    last_cand = cand
                except RuntimeError as e:
                    if manual is None:
                        roi, status = None, f"{e} - adjust the camera, or press m to draw the ROI"
                    last_cand = None
        lines = [(status, GREEN if roi else RED),
                 ("SPACE/ENTER = start    l = locate again    m = draw ROI by hand    q = quit", WHITE)]
        key = win.show(frame, lines, roi, roi_ok=roi is not None)
        if key == ord("l") and fixed_roi is None:
            manual, last_cand, since = None, None, LOCATE_EVERY
        if key == ord("m"):
            hand = win.select(frame)
            if hand:
                manual = roi = hand
                status = f"ROI drawn by hand = {hand}"
                since = 0
        if key in (ord("q"), ESC):
            return None, None
        if key in (SPACE, ENTER) and roi is not None:
            return roi, frame
        if not preview and roi is not None:
            print(f"ROI {roi} - starting (no preview, so no key press)")
            return roi, frame
        if not preview and since == 0 and roi is None:
            print(status)


STOP = False          # set by Ctrl+C: finish the frame, then stop cleanly (so the grade is still printed and saved)


def _on_sigint(signum, frm):
    global STOP
    if STOP:                       # second Ctrl+C: really quit
        raise KeyboardInterrupt
    STOP = True
    print("\nstopping ... (Ctrl+C again to quit at once)", flush=True)


def run_session(cap, is_camera, win, roi, first_frame, fps, speed=0.0, record=None):
    """Detect until the user quits / the source ends. Returns (monitor, restart_requested)."""
    fs = sf.FeatureStream(roi)
    mon = OnlineMonitor(fps=fps)
    log, shown = deque(maxlen=KEEP), 0
    t0, frame, restart = time.time(), first_frame, None
    r_count = [0]
    writer = None

    def finish_writer():
        if writer is not None:
            writer.release()

    def feed(index, row_diff):
        b = float(np.percentile(row_diff, 20))
        return b, float(np.percentile(row_diff, 90)) - b

    def handle(msgs):
        for m in msgs:
            print(m["text"], flush=True)
            if LOG_FILE:
                LOG_FILE.write(m["text"] + "\n")
                LOG_FILE.flush()
            log.append(m["text"])

    while frame is not None and not STOP:
        if record and writer is None:
            hh, ww = frame.shape[:2]
            writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
        if writer is not None:
            writer.write(frame)
        try:
            r = fs.update(frame)
        except RuntimeError as e:      # reference window not healthy (fault, camera moved, cut)
            msg = f"REFERENCE FAILED: {e}. Starting over - keep the screen healthy."
            handle([dict(text=msg)])
            finish_writer()
            return mon, "auto"
        if r["backfill"]:
            for k, rd in r["backfill"].items():
                mon.backfill(k, *feed(k, rd))
        cols = sf.csv_columns(r["roi_diff"], r["roi_lit"], r["row_diff"], r["row_diff_ff"], r["index"])
        handle(mon.push(*cols[1:]))
        n = r["index"] + 1
        r_count[0] = n
        if n == 600:       # real frame rate check once the warm-up is done
            real = n / max(time.time() - t0, 1e-6)
            if is_camera and abs(real - fps) / fps > 0.10:
                note = (f"NOTE: the camera delivered {real:.1f} fps but --fps is {fps:g}; "
                        f"times shown are off by {fps / real:.2f}x. Re-run with --fps {real:.0f}.")
                print(note)
                log.append(note)

        if n < 600:
            phase = (f"BUILDING REFERENCE  {n}/600 - keep the screen healthy", YELLOW)
        else:
            phase = ("DETECTING", GREEN)
        ongoing = [e for e in mon.current_events() if e["ongoing"]]
        lines = [(f"{phase[0]}   t={n / fps:5.1f}s   events: {len(mon.current_events())}"
                  + (f"   NOW: {', '.join(e['label'] for e in ongoing)}" if ongoing else ""),
                  RED if ongoing else phase[1]),
                 ("r = start over    q = stop", WHITE)]
        lines += [(t, WHITE) for t in log]
        key = win.show(frame, lines, roi)
        if key in (ord("q"), ESC):
            break
        if key == ord("r"):
            restart = "manual"
            break
        if speed > 0:                  # play a file at (roughly) its real speed
            ahead = t0 + n / (fps * speed) - time.time()
            if ahead > 0:
                time.sleep(ahead)
        frame = read(cap, is_camera)

    finish_writer()
    el = time.time() - t0
    if is_camera and r_count[0] > 60:
        print(f"processed {r_count[0]} frames in {el:.1f} s = {r_count[0] / el:.1f} fps "
              f"(camera asked for {fps:g} fps; clearly lower means frames were dropped)", flush=True)
    for k, rd in fs.finish().items():
        mon.backfill(k, *feed(k, rd))
    if not restart:
        handle(mon.finish())
    return mon, restart


def save_csv(mon, path):
    """The features of this session, same columns as collect_data's *_data.csv."""
    if mon.n:
        mon._table().to_csv(path, index=False)
        print(f"saved the session's features to {path}  (replay it with: python replay.py {path})")


def report(mon, fps, out, test_window=None, recovery_s=10.0, settle_s=2.0):
    if mon.n < 21:
        print("not enough frames for a report")
        print("\nEMC GRADE: " + emc_verdict.GRADE_MANUAL + "  (session too short to judge)")
        return
    if mon.n < 600:
        print(f"\nstopped after {mon.n} frames ({mon.n / fps:.0f} s) - before the full 600-frame reference, "
              "so restarts are reliable but the other faults may be missed")
    cat = catalog.build_catalog(mon._table(), fps, "live session")
    print("\n" + catalog.format_catalog(cat))
    res = emc_verdict.grade(cat, test_window, recovery_s, settle_s, df=mon._table())
    print("\n" + "=" * 60)
    print("EMC GRADE (CISPR 35): " + res["grade"])
    for line in emc_verdict.format_grade("", res).splitlines()[1:]:
        print(line)
    print("=" * 60)
    if out:
        rows = [dict(detector=e["detector"], label=e["type"], start=e["start"], end=e["end"],
                     start_s=e["start_s"], dur_s=e["dur_s"], note=e["evidence"]) for e in cat["events"]]
        pd.DataFrame(rows or [dict(label="CLEAN")]).assign(emc_grade=res["grade"]).to_csv(out, index=False)
        print("wrote", out)


LOG_FILE = None


def main():
    global LOG_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index (0,1,2..), video file or stream URL")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, help="ask the camera for this width")
    ap.add_argument("--height", type=int, help="ask the camera for this height")
    ap.add_argument("--roi", help="x0,y0,x1,y1 in camera pixels (skips automatic location)")
    ap.add_argument("--ts-box", help="timestamp box x0,y0,x1,y1 (default: TS_BOX from video_generator_timestamp)")
    ap.add_argument("--speed", type=float,
                    help="playback speed for a video file: 1 = real time, 4 = four times faster, 0 = as fast as "
                         "possible (default: 1 with a preview window, 0 without)")
    ap.add_argument("--no-preview", action="store_true", help="no window; start as soon as the ROI is known")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="camera only: read and throw away this many seconds first, so auto-exposure and white balance "
                         "settle before anything is recorded or analysed (try 10 on the Jetson CSI camera)")
    ap.add_argument("--no-save", action="store_true",
                    help="do not save the session's features as live_<time>_data.csv")
    ap.add_argument("--record", help="also save the frames used (from the start key on) to this .mp4")
    ap.add_argument("--log", help="append every live message to this file")
    ap.add_argument("--out", help="write the final events (and the EMC grade) to this CSV")
    ap.add_argument("--test-window", type=emc_verdict.parse_window,
                    help="START,END seconds (from the start key) when the disturbance was applied; used for the EMC grade")
    ap.add_argument("--recovery", type=float, default=10.0,
                    help="seconds of healthy picture needed after the last fault for grade B (default 10)")
    a = ap.parse_args()
    signal.signal(signal.SIGINT, _on_sigint)

    fixed = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
    ts_box = None if fixed else get_ts_box(a.ts_box)
    if a.log:
        LOG_FILE = open(a.log, "a")
    cap, is_camera = open_source(a.source, a.width, a.height, a.fps)
    if is_camera and a.warmup > 0:
        print(f"warming up the camera for {a.warmup:g} s ...")
        t_end = time.time() + a.warmup
        while time.time() < t_end and not STOP:
            if read(cap, is_camera) is None:
                break
    win = Window(not a.no_preview)
    preview = win.enabled
    is_file = (not is_camera) and os.path.isfile(a.source)
    speed = a.speed if a.speed is not None else (1.0 if preview else 0.0)
    if not is_file:
        speed = 0.0                    # cameras and streams set their own pace

    try:
        while True:
            roi, first = setup_phase(cap, is_camera, win, ts_box, fixed, preview, is_file)
            if roi is None:
                break
            print(f"\nstarted, ROI {roi}, fps {a.fps:g}\n")
            mon, again = run_session(cap, is_camera, win, roi, first, a.fps, speed, a.record)
            while again == "auto":      # failed reference: start again at once, same ROI
                nxt = read(cap, is_camera)
                if nxt is None:
                    print("source ended")
                    return
                print("\n--- starting over ---\n")
                mon, again = run_session(cap, is_camera, win, roi, nxt, a.fps, speed, a.record)
            if again:
                print("\n--- starting over ---\n")
                fixed = roi            # keep the same ROI, go back to set-up (press SPACE to start)
                continue
            if not a.no_save:
                save_csv(mon, time.strftime("live_%Y%m%d_%H%M%S_data.csv"))
            report(mon, a.fps, a.out, a.test_window, a.recovery)
            break
    finally:
        cap.release()
        win.close()
        if LOG_FILE:
            LOG_FILE.close()


if __name__ == "__main__":
    main()
