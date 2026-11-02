import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Noise:
    fps: float = 30.0
    band_top: float = 0.20        # where the top of the band is, fractionally compared against video height
    band_height: float = 0.17     # height of the band, fraction as well
    edge: float = 6.0             # soft-edge width in pixels
    strength: float = 0.9         # 0 to 1, how strong the noise is
    desat: float = 0.85           # 0 to 1, how much colour is removed inside the band
    level: float = 110.0          # mean grey level of the noise floor (0 to 255) mixed in
    amp: float = 40.0             # std of the noise in grey levels
    streak: int = 14              # horizontal correlation length of the noise, pixels (at 960 columns)
    row_amp: float = 0.45         # extra per-row brightness offset, fraction of amp (the "scan line" look)
    severity: float = 0.5         # 0 to 1, sets how fast the band scrolls vertically
    speed_lo: float = 0.04        # band speed at severity 0, in picture heights per second
    speed_hi: float = 0.60        # band speed at severity 1, in picture heights per second
    speed_override: float = -1.0  # >= 0 forces a speed (heights per second) regardless of severity
    direction: int = 1            # +1 scrolls down, -1 scrolls up; the band wraps round the picture
    start: int = 600
    end: int = 1200
    ramp: int = 10
    seed: int = 1

    def describe(self, H):
        return (f"band {self.band_height * H:.0f} of {H} rows ({self.band_height:.0%}), top {self.band_top:.0%}, "
                f"strength {self.strength}, desat {self.desat}, noise {self.level:.0f}+-{self.amp:.0f}, "
                f"streak {self.streak}px, severity {self.severity}, speed {self.speed():.2f} heights/s "
                f"({self.speed() * H / self.fps:.1f} rows/frame)")

    def speed(self):
        if self.speed_override >= 0:
            return self.speed_override
        sv = min(max(self.severity, 0.0), 1.0)
        return self.speed_lo + sv * (self.speed_hi - self.speed_lo)


class NoiseSim:
    def __init__(self, params: Noise, height: int, width: int):
        self.p, self.H, self.W = params, height, width
        self.rng = np.random.default_rng(params.seed)
        self.y = np.arange(height, dtype=np.float32)[:, None]

    def envelope(self, k):
        p = self.p
        if k < p.start or k > p.end:
            return 0.0
        return float(min(1.0, (k - p.start + 1) / max(p.ramp, 1), (p.end - k + 1) / max(p.ramp, 1)))

    def top(self, k):
        p = self.p
        return (p.band_top + p.direction * p.speed() * max(k - p.start, 0) / p.fps) % 1.0

    def weight(self, k):
        p, H = self.p, self.H
        e = self.envelope(k)
        if e == 0:
            return None
        y0 = self.top(k) * H
        soft = max(p.edge * H / 540.0, 0.5)
        sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        w = 0.0
        for shift in (-H, 0.0, H):
            a = y0 + shift
            w = w + sig((self.y - a) / soft) - sig((self.y - (a + p.band_height * H)) / soft)
        return (np.clip(w, 0, 1) * e * p.strength).astype(np.float32)

    def noise_field(self, rows):
        p, W = self.p, self.W
        n = self.rng.normal(0, 1, (rows, W)).astype(np.float32)
        L = max(int(round(p.streak * W / 960.0)), 1)
        if L > 1:
            n = cv2.blur(n, (L, 1))
            n *= np.sqrt(L)                                   
        n += self.rng.normal(0, 1, (rows, 1)).astype(np.float32) * p.row_amp
        return n

    def apply(self, frame, k):
        w = self.weight(k)
        if w is None:
            return frame
        p = self.p
        f = frame.astype(np.float32)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[:, :, None]
        mixed = p.desat * grey + (1 - p.desat) * f                     
        noise = (p.level + p.amp * self.noise_field(self.H))[:, :, None]
        target = 0.5 * mixed + 0.5 * noise                             
        out = (1 - w[:, :, None]) * f + w[:, :, None] * target
        return np.clip(out, 0, 255).astype(np.uint8)

# Colorbar Video for generation
BARS = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255), (255, 0, 0)]


def colorbar_frame(k, fps, H=540, W=960):
    img = np.zeros((H, W, 3), np.uint8)
    bw = W // 8                                          
    for i, c in enumerate(BARS):
        img[:, i * bw:(i + 1) * bw] = c
    img = (img * 0.78).astype(np.uint8)
    cv2.rectangle(img, (14, 12), (14 + 170, 12 + 44), (30, 30, 30), -1)
    cv2.putText(img, f"{k / fps:07.3f}s", (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    return img


def contact_sheet(frames, labels, path):
    tiles = []
    for f, lab in zip(frames, labels):
        t = cv2.resize(f, (480, 270))
        cv2.putText(t, lab, (8, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, lab, (8, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def write_video(path, frames_iter, fps, size):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in frames_iter:
        vw.write(f)
    vw.release()


def add_params(ap):
    d = Noise()
    ap.add_argument("--fps", type=float, default=d.fps)
    ap.add_argument("--band-top", type=float, default=d.band_top)
    ap.add_argument("--band-height", type=float, default=d.band_height)
    ap.add_argument("--edge", type=float, default=d.edge)
    ap.add_argument("--strength", type=float, default=d.strength)
    ap.add_argument("--desat", type=float, default=d.desat)
    ap.add_argument("--level", type=float, default=d.level)
    ap.add_argument("--amp", type=float, default=d.amp)
    ap.add_argument("--streak", type=int, default=d.streak)
    ap.add_argument("--row-amp", type=float, default=d.row_amp)
    ap.add_argument("--severity", type=float, default=d.severity, help="0..1, faster vertical scroll when higher")
    ap.add_argument("--speed-lo", type=float, default=d.speed_lo)
    ap.add_argument("--speed-hi", type=float, default=d.speed_hi)
    ap.add_argument("--speed", dest="speed_override", type=float, default=d.speed_override,
                    help="force a speed in picture heights per second (ignores severity)")
    ap.add_argument("--direction", type=int, choices=[-1, 1], default=d.direction, help="1 = down, -1 = up")
    ap.add_argument("--start", type=int, default=d.start)
    ap.add_argument("--end", type=int, default=d.end)
    ap.add_argument("--ramp", type=int, default=d.ramp)
    ap.add_argument("--seed", type=int, default=d.seed)


def params_from(a):
    return Noise(fps=a.fps, band_top=a.band_top, band_height=a.band_height, edge=a.edge, strength=a.strength,
                 desat=a.desat, level=a.level, amp=a.amp, streak=a.streak, row_amp=a.row_amp,
                 severity=a.severity, speed_lo=a.speed_lo, speed_hi=a.speed_hi, speed_override=a.speed_override,
                 direction=a.direction, start=a.start, end=a.end, ramp=a.ramp, seed=a.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="make a colour-bar demo video with the band, plus pictures of it")
    d.add_argument("out_dir")
    d.add_argument("--frames", type=int, default=1500)
    add_params(d)
    p = sub.add_parser("apply", help="add the band to an existing video")
    p.add_argument("video"); p.add_argument("out")
    add_params(p)
    a = ap.parse_args()
    prm = params_from(a)

    if a.cmd == "demo":
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        H, W, n = 540, 960, a.frames
        sim = NoiseSim(prm, H, W)
        print("model:", sim.p.describe(H))
        write_video(out / "demo_noise_band.mp4", (sim.apply(colorbar_frame(k, prm.fps, H, W), k) for k in range(n)),
                    prm.fps, (W, H))
        sim2 = NoiseSim(prm, H, W)
        picks = [min(max(k, 0), n - 1) for k in (prm.start - 30, prm.start + 2, prm.start + 40, prm.start + 41,
                                                  prm.start + 300, prm.end + 40)]
        fr = [sim2.apply(colorbar_frame(k, prm.fps, H, W), k) for k in picks]
        contact_sheet(fr, [f"frame {k}" + (" (healthy)" if sim2.envelope(k) == 0 else "") for k in picks],
                      out / "demo_contact_sheet.png")
        sim3 = NoiseSim(prm, H, W)
        cv2.imwrite(str(out / "demo_full_frame.png"), sim3.apply(colorbar_frame(picks[2], prm.fps, H, W), picks[2]))
        print("wrote", out / "demo_noise_band.mp4", out / "demo_contact_sheet.png", out / "demo_full_frame.png")
    else:
        cap = cv2.VideoCapture(a.video)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        prm.fps = (cap.get(cv2.CAP_PROP_FPS) or prm.fps) if a.fps == 30.0 else a.fps
        sim = NoiseSim(prm, H, W)
        print("model:", sim.p.describe(H))

        def gen():
            k = 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield sim.apply(f, k)
                k += 1
        write_video(a.out, gen(), prm.fps, (W, H))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
