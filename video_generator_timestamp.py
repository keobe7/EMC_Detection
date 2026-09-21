#!/home/keobe/Programs/myvenv/bin/python
# change shebang line as needed

import os
import csv
import numpy as np
import cv2

#video properties
WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION_SEC = 60
N_FRAMES = FPS * DURATION_SEC
OUT_DIR = "Videos"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

#timestamp properties
FONT = cv2.FONT_HERSHEY_SIMPLEX
TS_ORG = (40, 110)
TS_SCALE = 2.2
TS_THICK = 4
TS_PAD =15

#static box to remove box changing sizes inducing noise
TS_TEMPLATE = "00000 | 000.000s"          
_DIGIT_W = cv2.getTextSize("0", FONT, TS_SCALE, TS_THICK)[0][0]
(_, _TH), _BASE = cv2.getTextSize(TS_TEMPLATE, FONT, TS_SCALE, TS_THICK)
_TW = _DIGIT_W * len(TS_TEMPLATE)

TS_BOX = ((TS_ORG[0] - TS_PAD, TS_ORG[1] - _TH - TS_PAD),
          (TS_ORG[0] + _TW + TS_PAD, TS_ORG[1] + _BASE + TS_PAD))

#different severity levels
SEVERITIES = {
    "negligible": 0.05,
    "minimum": 0.1,
    "mild": 0.3,
    "moderate": 0.6,
    "severe": 1.0,
}

#output directory, will make if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

#function to draw the timestamp
def draw_timestamp(frame, i, fps=FPS):
    out = frame.copy()
    label = f"{i:05d} | {i / fps:07.3f}s"
    x, y = TS_ORG
    for ch in label:
        cv2.putText(out, ch, (x, y), FONT, TS_SCALE, (255, 255, 255),
                    TS_THICK, cv2.LINE_8)
        x += _DIGIT_W
    return out

#function to bake it into the video
def bake_timestamp_box(frame):
    out = frame.copy()
    cv2.rectangle(out, TS_BOX[0], TS_BOX[1], (0, 0, 0), -1)
    return out

#colorbars are standard for error testing in monitors
def make_colorbars(width, height):
    colors = [
        (192, 192, 192),  # white/gray
        (0, 192, 192),    # yellow (BGR)
        (192, 192, 0),    # cyan
        (0, 192, 0),      # green
        (192, 0, 192),    # magenta
        (0, 0, 192),      # red
        (192, 0, 0),      # blue
    ]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    bar_w = width // len(colors)
    for i, color in enumerate(colors):
        x0 = i * bar_w
        x1 = width if i == len(colors) - 1 else x0 + bar_w
        frame[:, x0:x1] = color
    return frame

#grid based pattern video, currently unused
def make_grid(width, height, cell=40):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                frame[y:y + cell, x:x + cell] = (255, 255, 255)
    return frame

#another video pattern, currently unused
def make_ramp(width, height):
    ramp_row = np.linspace(0, 255, width, dtype=np.uint8)
    frame = np.tile(ramp_row, (height, 1))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame

#array for video patterns later generated, uncomment to enable the specific pattern
PATTERNS = {
    "colorbars": make_colorbars,
    # "grid": make_grid,
    # "ramp": make_ramp,
}

#defines each type of fault and how they will affect the video
def fault_flicker(frame, severity, rng, t=0):
    gain = 1.0 + severity * 0.8 * np.sin(2 * np.pi * t / 5.0)
    gain += rng.uniform(-0.1, 0.1) * severity
    out = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out

def fault_color_shift(frame, severity, rng):
    out = frame.astype(np.float32)
    offsets = rng.uniform(-1, 1, size=3) * 60 * severity
    gains = 1.0 + rng.uniform(-1, 1, size=3) * 0.3 * severity
    for c in range(3):
        out[:, :, c] = out[:, :, c] * gains[c] + offsets[c]
    return np.clip(out, 0, 255).astype(np.uint8)

def fault_noise(frame, severity, rng):
    out = frame.astype(np.float32)
    sigma = 60 * severity
    out += rng.normal(0, sigma, frame.shape)
    out = np.clip(out, 0, 255)
    sp_ratio = 0.02 * severity
    mask = rng.random(frame.shape[:2])
    out[mask < sp_ratio / 2] = 0
    out[mask > 1 - sp_ratio / 2] = 255
    return out.astype(np.uint8)

def fault_tearing(frame, severity, rng):
    out = frame.copy()
    h, w = frame.shape[:2]
    band_h = int(h * (0.1 + 0.3 * severity))
    y0 = rng.integers(0, max(1, h - band_h))
    shift = int(w * 0.15 * severity) * rng.choice([-1, 1])
    band = out[y0:y0 + band_h]
    out[y0:y0 + band_h] = np.roll(band, shift, axis=1)
    return out

def fault_geometric_warp(frame, severity, rng):
    h, w = frame.shape[:2]
    max_shift = 0.08 * severity * w
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = src + rng.uniform(-max_shift, max_shift, size=src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def fault_blackout(frame, severity, rng):
    out = frame.astype(np.float32) * (1 - severity)
    return np.clip(out, 0, 255).astype(np.uint8)

#faults based on spacial properties
SPATIAL_FAULTS = {
    #"color_shift": fault_color_shift,
    #"noise": fault_noise,
    "tearing": fault_tearing,
    "geometric_warp": fault_geometric_warp,
    "blackout": fault_blackout,
}

#stream video
def stream_video(path, size, n_frames, frame_fn, fps=FPS):
    w, h = size
    writer = cv2.VideoWriter(path, FOURCC, fps, (w, h))
    for i in range(n_frames):
        writer.write(frame_fn(i))
    writer.release()

#drawing the timestamp on the video itself
def clean_fn(base):
    return lambda i: draw_timestamp(base, i)

def spatial_fn(base, fault_fn, severity, rng, start, end):
    def fn(i):
        frame = draw_timestamp(base, i)
        return fault_fn(frame, severity, rng) if start <= i < end else frame
    return fn

def flicker_fn(base, severity, rng, start, end, fps=FPS):
    def fn(i):
        frame = draw_timestamp(base, i)
        if start <= i < end:
            return fault_flicker(frame, severity, rng, t=(i - start) / fps)
        return frame
    return fn

def freeze_fn(base, start, end):
    def fn(i):
        return draw_timestamp(base, start - 1 if start <= i < end else i)
    return fn

def main():
    rng = np.random.default_rng(seed=42)
    manifest_rows = []
    size = (WIDTH, HEIGHT)

    for pattern_name, gen_fn in PATTERNS.items():
        base_frame = bake_timestamp_box(gen_fn(WIDTH, HEIGHT))

	#create clean reference for external referencing if needed
        ref_path = os.path.join(OUT_DIR, f"reference_{pattern_name}.mp4")
        stream_video(ref_path, size, N_FRAMES, clean_fn(base_frame))
        print(f"Wrote {ref_path}")

        #blackout, geometric_warp, tearing
        for fault_name, fault_fn in SPATIAL_FAULTS.items():
            for sev_name, sev_val in SEVERITIES.items():
                start = N_FRAMES // 3
                end = start + N_FRAMES // 3
                out_path = os.path.join(
                    OUT_DIR, f"faulty_{pattern_name}_{fault_name}_{sev_name}.mp4")
                stream_video(out_path, size, N_FRAMES,
                             spatial_fn(base_frame, fault_fn, sev_val, rng, start, end))
                manifest_rows.append([out_path, pattern_name, fault_name, sev_name,
                                      start, end - 1])
                print(f"Wrote {out_path}")

        #flicker
        for sev_name, sev_val in SEVERITIES.items():
            start = N_FRAMES // 3
            end = start + N_FRAMES // 3
            out_path = os.path.join(
                OUT_DIR, f"faulty_{pattern_name}_flicker_{sev_name}.mp4")
            stream_video(out_path, size, N_FRAMES,
                         flicker_fn(base_frame, sev_val, rng, start, end))
            manifest_rows.append([out_path, pattern_name, "flicker", sev_name,
                                  start, end - 1])
            print(f"Wrote {out_path}")

        #freeze, severity will affect duration of error
        for sev_name, sev_val in SEVERITIES.items():
            start = N_FRAMES // 3
            end = start + int((N_FRAMES // 4) * (0.5 + sev_val))
            out_path = os.path.join(
                OUT_DIR, f"faulty_{pattern_name}_freeze_{sev_name}.mp4")
            stream_video(out_path, size, N_FRAMES, freeze_fn(base_frame, start, end))
            manifest_rows.append([out_path, pattern_name, "freeze", sev_name,
                                  start, end - 1])
            print(f"Wrote {out_path}")

    #write manifest file for documenting each video generated and it's properties
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "pattern", "fault_type", "severity",
                         "fault_start_frame", "fault_end_frame"])
        writer.writerows(manifest_rows)
    print(f"Wrote {manifest_path} ({len(manifest_rows)} fault clips)")

if __name__ == "__main__":
    main()
