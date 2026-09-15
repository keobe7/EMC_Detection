#!/home/keobe/Programs/myvenv/bin/python
# change shebang line as needed

import os
import csv
import numpy as np
import cv2

WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION_SEC = 60
N_FRAMES = FPS * DURATION_SEC
OUT_DIR = "Videos"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

SEVERITIES = {
    "negligible": 0.05,
    "minimum": 0.1,
    "mild": 0.3,
    "moderate": 0.6,
    "severe": 1.0,
}

os.makedirs(OUT_DIR, exist_ok=True)


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


def make_grid(width, height, cell=40):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                frame[y:y + cell, x:x + cell] = (255, 255, 255)
    return frame


def make_ramp(width, height):
    ramp_row = np.linspace(0, 255, width, dtype=np.uint8)
    frame = np.tile(ramp_row, (height, 1))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


PATTERNS = {
    "colorbars": make_colorbars,
    # "grid": make_grid,
    # "ramp": make_ramp,
}


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


SPATIAL_FAULTS = {
    #"color_shift": fault_color_shift,
    #"noise": fault_noise,
    "tearing": fault_tearing,
    "geometric_warp": fault_geometric_warp,
    "blackout": fault_blackout,
}


def stream_clean_video(path, base_frame, n_frames, fps=FPS):
    h, w = base_frame.shape[:2]
    writer = cv2.VideoWriter(path, FOURCC, fps, (w, h))
    for _ in range(n_frames):
        writer.write(base_frame)
    writer.release()


def stream_spatial_fault_video(path, base_frame, n_frames, fault_fn, severity, rng,
                                start, duration, fps=FPS):
    h, w = base_frame.shape[:2]
    writer = cv2.VideoWriter(path, FOURCC, fps, (w, h))
    end = start + duration
    for i in range(n_frames):
        if start <= i < end:
            frame = fault_fn(base_frame.copy(), severity, rng)
        else:
            frame = base_frame
        writer.write(frame)
    writer.release()


def stream_flicker_video(path, base_frame, n_frames, severity, rng, start, duration, fps=FPS):
    h, w = base_frame.shape[:2]
    writer = cv2.VideoWriter(path, FOURCC, fps, (w, h))
    end = start + duration
    for i in range(n_frames):
        if start <= i < end:
            t = (i - start) / fps
            frame = fault_flicker(base_frame.copy(), severity, rng, t=t)
        else:
            frame = base_frame
        writer.write(frame)
    writer.release()


def stream_freeze_video(path, base_frame, n_frames, start, duration, fps=FPS):
    h, w = base_frame.shape[:2]
    writer = cv2.VideoWriter(path, FOURCC, fps, (w, h))
    for _ in range(n_frames):
        writer.write(base_frame)
    writer.release()


def stream_dropped_frames_video(path, base_frame, n_frames, severity, rng, start, duration, fps=FPS):
    h, w = base_frame.shape[:2]
    writer = cv2.VideoWriter(path, FOURCC, fps, (w, h))
    end = start + duration
    duty = 0.3 + 0.5 * severity
    for i in range(n_frames):
        if start <= i < end and rng.random() < duty:
            if rng.random() < 0.5:
                frame = (rng.random((h, w, 3)) * 255).astype(np.uint8)
            else:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            frame = base_frame
        writer.write(frame)
    writer.release()


def main():
    rng = np.random.default_rng(seed=42)
    manifest_rows = []

    for pattern_name, gen_fn in PATTERNS.items():
        base_frame = gen_fn(WIDTH, HEIGHT)  # ONE frame in memory, reused everywhere

        ref_path = os.path.join(OUT_DIR, f"reference_{pattern_name}.mp4")
        stream_clean_video(ref_path, base_frame, N_FRAMES)
        print(f"Wrote {ref_path}")

        for fault_name, fault_fn in SPATIAL_FAULTS.items():
            for sev_name, sev_val in SEVERITIES.items():
                start = N_FRAMES // 3
                duration = N_FRAMES // 6
                out_path = os.path.join(OUT_DIR, f"faulty_{pattern_name}_{fault_name}_{sev_name}.mp4")
                stream_spatial_fault_video(out_path, base_frame, N_FRAMES, fault_fn, sev_val, rng,
                                            start, duration)
                manifest_rows.append([out_path, pattern_name, fault_name, sev_name,
                                       start, start + duration - 1])
                print(f"Wrote {out_path}")

        for sev_name, sev_val in SEVERITIES.items():
            start = N_FRAMES // 3
            duration = N_FRAMES // 4
            out_path = os.path.join(OUT_DIR, f"faulty_{pattern_name}_flicker_{sev_name}.mp4")
            stream_flicker_video(out_path, base_frame, N_FRAMES, sev_val, rng, start, duration)
            manifest_rows.append([out_path, pattern_name, "flicker", sev_name,
                                   start, start + duration - 1])
            print(f"Wrote {out_path}")

        for sev_name, sev_val in SEVERITIES.items():
            start = N_FRAMES // 3
            duration = int((N_FRAMES // 6) * (0.5 + sev_val))
            out_path = os.path.join(OUT_DIR, f"faulty_{pattern_name}_freeze_{sev_name}.mp4")
            stream_freeze_video(out_path, base_frame, N_FRAMES, start, duration)
            manifest_rows.append([out_path, pattern_name, "freeze", sev_name,
                                   start, start + duration - 1])
            print(f"Wrote {out_path}")

        #for sev_name, sev_val in SEVERITIES.items():
           # start = N_FRAMES // 3
           # duration = N_FRAMES // 5
           # out_path = os.path.join(OUT_DIR, f"faulty_{pattern_name}_dropped_{sev_name}.mp4")
           # stream_dropped_frames_video(out_path, base_frame, N_FRAMES, sev_val, rng, start, duration)
           # manifest_rows.append([out_path, pattern_name, "dropped_frames", sev_name,
           #                        start, start + duration - 1])
           # print(f"Wrote {out_path}")

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "pattern", "fault_type", "severity",
                          "fault_start_frame", "fault_end_frame"])
        writer.writerows(manifest_rows)
    print(f"Wrote {manifest_path} ({len(manifest_rows)} fault clips)")


if __name__ == "__main__":
    main()
