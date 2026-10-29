from __future__ import annotations
import os
import sys
import traceback

# Windowed exe has no console: make stdout/stderr safe for libraries that write to them.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


def report(text):
    """Show the error in a message box and save it to crash_log.txt next to the exe."""
    for folder in (os.path.dirname(os.path.abspath(sys.argv[0])), os.environ.get("TEMP", ".")):
        try:
            with open(os.path.join(folder, "crash_log.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            break
        except Exception:
            pass
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text[-1800:], "Star Trail Studio - error", 0x10)
    except Exception:
        print(text)


# ===== image IO =====
"""Universal image loading -> RGB uint16 arrays, and uncompressed saving."""
import os
import numpy as np
import cv2

try:
    import rawpy
except Exception:  # pragma: no cover
    rawpy = None

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pillow_heif = None

from PIL import Image, ImageOps

RAW_EXT = {".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2",
           ".orf", ".rw2", ".raf", ".pef", ".srw", ".3fr", ".iiq", ".erf", ".kdc",
           ".mrw", ".x3f", ".rwl"}
HEIF_EXT = {".heic", ".heif", ".hif", ".avif"}
STD_EXT = {".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".bmp", ".webp", ".jp2"}
ALL_EXT = RAW_EXT | HEIF_EXT | STD_EXT


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ALL_EXT


def natural_key(path: str):
    """Natural order: burst 2 < burst 10 < burst 100 (numbers compared as numbers)."""
    import re
    name = os.path.basename(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def sort_paths(paths, mode: str = "name") -> list[str]:
    if mode == "date":
        return sorted(paths, key=lambda p: (os.path.getmtime(p), natural_key(p)))
    return sorted(paths, key=lambda p: (os.path.dirname(p).lower(), natural_key(p)))


def list_images(folder: str) -> list[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if is_supported(f)]
    return sort_paths(files)


def _to_u16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return (arr.astype(np.uint16) << 8) | arr.astype(np.uint16)  # exact 0..65535 scaling
    if arr.dtype in (np.float32, np.float64):
        return np.clip(arr * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    return arr.astype(np.uint16)


def load_image(path: str, half_size: bool = False) -> np.ndarray:
    """Return HxWx3 RGB uint16. half_size speeds RAW decoding for previews."""
    ext = os.path.splitext(path)[1].lower()

    if ext in RAW_EXT:
        if rawpy is None:
            raise RuntimeError("rawpy is not installed (needed for RAW/DNG files)")
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                                  output_bps=16, half_size=half_size,
                                  gamma=(2.222, 4.5))
        return rgb

    if ext in HEIF_EXT:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        # Try for >8-bit data first
        try:
            hf = pillow_heif.open_heif(path, convert_hdr_to_8bit=False)
            if hf.mode.startswith("RGB") and "16" in hf.mode:
                arr = np.asarray(hf)
                arr = arr[..., :3]
                # pillow-heif returns 10/12-bit values left-aligned in 16 bits
                return _to_u16(arr)
        except Exception:
            pass
        im = im.convert("RGB")
        return _to_u16(np.asarray(im))

    # Standard formats via OpenCV (keeps 16-bit PNG/TIFF), Pillow as fallback
    data = np.fromfile(path, dtype=np.uint8)  # works with unicode Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return _to_u16(np.asarray(im))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if ext in (".jpg", ".jpeg", ".jpe"):
        try:  # honour EXIF orientation for JPEG
            o = Image.open(path).getexif().get(274, 1)
            if o != 1:
                img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        except Exception:
            pass
    return _to_u16(img)


def load_preview(path: str, max_side: int = 1280) -> np.ndarray:
    img = load_image(path, half_size=True)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def save_uncompressed(path: str, rgb16: np.ndarray, bits: int = 16) -> str:
    """Save as PNG (16/8-bit, stored without compression), uncompressed TIFF, or BMP.
    Works with non-ASCII Windows paths."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = rgb16 if bits == 16 else (rgb16 >> 8).astype(np.uint8)
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    elif ext == ".png":
        arr = rgb16 if bits == 16 else (rgb16 >> 8).astype(np.uint8)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    elif ext == ".bmp":
        arr = (rgb16 >> 8).astype(np.uint8)
        params = []
    else:
        raise ValueError("Output must be .png, .tif/.tiff or .bmp")
    ok, buf = cv2.imencode(ext, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise RuntimeError("Could not encode image")
    buf.tofile(path)
    return path
# ===== engine =====
import subprocess
import tempfile
from typing import Callable, Sequence
import imageio_ffmpeg

Progress = Callable[[int, int, str], None]
Cancel = Callable[[], bool]


def _noop(*a): pass
def _never(): return False


def _fit_size(img, size):
    if size is None:
        return img
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ===== hot-pixel and noise cleanup =====
import warnings


class HotPixelFixer:
    """Repairs known hot pixels by replacing them with the median of their clean neighbours."""

    def __init__(self, mask):
        self.shape = mask.shape
        self.ys, self.xs = np.nonzero(mask)
        r = 3
        offs = np.array([(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if (dy, dx) != (0, 0)])
        h, w = mask.shape
        self.iy = np.clip(self.ys[:, None] + offs[None, :, 0], 0, h - 1)
        self.ix = np.clip(self.xs[:, None] + offs[None, :, 1], 0, w - 1)
        self.valid = ~mask[self.iy, self.ix]

    @property
    def count(self):
        return int(len(self.ys))

    def apply(self, img):
        if img.shape[:2] != self.shape or self.count == 0:
            return img
        out = img.copy()
        step = 50000
        for a in range(0, self.count, step):
            sl = slice(a, a + step)
            vals = img[self.iy[sl], self.ix[sl]].astype(np.float32)      # (N, 48, 3)
            vals[~self.valid[sl]] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                repl = np.nanmedian(vals, axis=1)
            ok = ~np.isnan(repl).any(axis=1)
            out[self.ys[sl][ok], self.xs[sl][ok]] = np.clip(repl[ok], 0, 65535).astype(np.uint16)
        return out


def build_hot_pixel_fixer(files, sensitivity=5, samples=12, progress=_noop, cancel=_never):
    """
    Hot pixels stay on the same sensor pixel while stars drift, so a spot that is bright
    compared with its surroundings in most sampled frames (and is tiny) is a hot pixel.
    Returns a HotPixelFixer, or None if there are too few photos to tell.
    """
    if len(files) < 3:
        return None
    idx = np.unique(np.linspace(0, len(files) - 1, min(samples, len(files))).astype(int))
    thr = max(6, 22 - 2 * sensitivity) * 257
    counts = None
    for j, i in enumerate(idx):
        if cancel():
            raise InterruptedError("Cancelled")
        progress(j, len(idx), f"Looking for hot pixels {j+1}/{len(idx)}")
        ch = load_image(files[i]).max(axis=2)
        hot = (ch.astype(np.int32) - cv2.medianBlur(ch, 5).astype(np.int32)) > thr
        if counts is None:
            counts = hot.astype(np.uint16)
        elif hot.shape == counts.shape:
            counts += hot
    need = max(2, int(np.ceil(0.6 * len(idx))))
    mask = (counts >= need).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.zeros(num, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= 12          # tiny spots only (not lamps)
    mask = cv2.dilate(small[labels].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    fixer = HotPixelFixer(mask)
    return fixer if fixer.count else None


def denoise_frame(img16, level):
    """level 0 off | 1 light (colour noise) | 2 medium | 3 strong (adds edge-preserving brightness denoise).
    Colour noise is smoothed at reduced resolution (fast); brightness detail is kept sharp."""
    if level <= 0:
        return img16
    y, cr, cb = cv2.split(cv2.cvtColor(img16, cv2.COLOR_RGB2YCrCb))
    h, w = y.shape
    k = 4 if max(h, w) >= 1600 else 2
    sig = {1: 1.5, 2: 2.5, 3: 3.5}[level] / k

    def smooth(c):
        small = cv2.resize(c, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sig)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    cr, cb = smooth(cr), smooth(cb)
    if level >= 2:
        strength = {2: 2.0, 3: 5.0}[level] * 257
        search = {2: 11, 3: 15}[level]
        y = cv2.fastNlMeansDenoising(y, [strength], None, 5, search, cv2.NORM_L1)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)


class TrailAccumulator:
    """Running star-trail stack. mode: 'lighten' (trails stay) or 'comet' (trails fade)."""

    def __init__(self, mode="lighten", comet_length=0.5, gap_fill=False):
        self.comet = (mode == "comet")
        self.decay = 1.0 - (1.0 - comet_length) * 0.2   # per-frame fade; length 1 = no fade
        self.gap_fill = gap_fill
        self.acc = None
        self.prev = None

    def add(self, img):
        if self.acc is None:
            self.acc = img.copy()
        else:
            if img.shape != self.acc.shape:
                img = cv2.resize(img, (self.acc.shape[1], self.acc.shape[0]), interpolation=cv2.INTER_AREA)
            base = self.acc
            if self.comet:
                base = (base.astype(np.float32) * self.decay).astype(np.uint16)
            self.acc = np.maximum(base, img)
            if self.gap_fill and self.prev is not None:
                mid = ((self.prev.astype(np.uint32) + img.astype(np.uint32)) >> 1).astype(np.uint16)
                self.acc = np.maximum(self.acc, mid)
        self.prev = img
        return self.acc


# ---------------------------------------------------------------- star trails
def stack_star_trails(files, mode="lighten", comet_length=0.5, gap_fill=False,
                      hot_pixels=False, hot_sensitivity=5, denoise=0,
                      progress=_noop, cancel=_never):
    """Stack all frames into one image. Streams one frame at a time (low memory)."""
    if not files:
        raise ValueError("No images selected")
    fixer = build_hot_pixel_fixer(files, hot_sensitivity, progress=progress, cancel=cancel) if hot_pixels else None
    acc = TrailAccumulator(mode, comet_length, gap_fill)
    n = len(files)
    for i, f in enumerate(files):
        if cancel():
            raise InterruptedError("Cancelled")
        progress(i, n, f"Stacking {os.path.basename(f)}")
        img = load_image(f)
        if fixer is not None:
            img = fixer.apply(img)
        acc.add(denoise_frame(img, denoise))
    progress(n, n, "Done")
    return acc.acc


# ------------------------------------------------------------------ timelapse
def _luma(img16):
    return float(img16[::8, ::8].mean())


def compute_deflicker_gains(files, window=7, progress=_noop, cancel=_never):
    lum = []
    for i, f in enumerate(files):
        if cancel():
            raise InterruptedError("Cancelled")
        progress(i, len(files), f"Measuring brightness {i+1}/{len(files)}")
        lum.append(max(_luma(load_image(f, half_size=True)), 1.0))
    lum = np.array(lum)
    k = max(1, window | 1)
    pad = k // 2
    padded = np.pad(lum, pad, mode="edge")
    smooth = np.convolve(padded, np.ones(k) / k, mode="valid")
    return list(np.clip(smooth / lum, 0.5, 2.0))


_ENC_CACHE = {}


def _has_encoder(ffmpeg, name):
    if name not in _ENC_CACHE:
        try:
            flags = 0x08000000 if os.name == "nt" else 0
            out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                                 text=True, creationflags=flags).stdout
            _ENC_CACHE[name] = name in out
        except Exception:
            _ENC_CACHE[name] = False
    return _ENC_CACHE[name]


def render_timelapse(files, out_path, fps=24.0, size=None, deflicker=False, deflicker_window=7,
                     fmt="mp4", trail_mode=None, comet_length=0.6, gap_fill=False,
                     hold_seconds=0.0, meteors=None, meteor_persist=2.0, meteor_mark=False,
                     hot_pixels=False, hot_sensitivity=5, denoise=0,
                     progress=_noop, cancel=_never):
    """
    fmt: "mp4" (H.264, high quality) | "avi_raw" (uncompressed 8-bit AVI) | "tiff_seq" (16-bit TIFF folder)
    trail_mode: None = normal timelapse
                "lighten" = star trails grow frame by frame
                "comet"   = moving comet trails (fading tails); comet_length 0..1
    hold_seconds: for video, keep the final frame on screen this long.
    """
    if not files:
        raise ValueError("No images selected")
    gains = None
    if deflicker and trail_mode is None:
        gains = compute_deflicker_gains(files, deflicker_window, progress, cancel)

    fixer = build_hot_pixel_fixer(files, hot_sensitivity, progress=progress, cancel=cancel) if hot_pixels else None
    first = load_image(files[0])
    if size is None:
        size = (first.shape[1], first.shape[0])
    size = (size[0] // 2 * 2, size[1] // 2 * 2)

    proc = None
    errlog = None
    if fmt in ("mp4", "avi_raw"):
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an"]
        if fmt == "mp4":
            if _has_encoder(ff, "libx264"):
                cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "14",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            else:
                cmd += ["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "rawvideo", "-pix_fmt", "bgr24"]
        cmd.append(out_path)
        errlog = tempfile.TemporaryFile()
        flags = 0x08000000 if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog, creationflags=flags)
    elif fmt == "tiff_seq":
        os.makedirs(out_path, exist_ok=True)
    else:
        raise ValueError(f"Unknown format {fmt}")

    def ffmpeg_error():
        try:
            errlog.seek(0)
            return errlog.read().decode(errors="replace")[-600:].strip()
        except Exception:
            return ""

    def write(img16, idx):
        if proc is not None:
            bgr8 = cv2.cvtColor((img16 >> 8).astype(np.uint8), cv2.COLOR_RGB2BGR)
            try:
                proc.stdin.write(np.ascontiguousarray(bgr8).tobytes())
            except (BrokenPipeError, OSError):
                proc.wait()
                raise RuntimeError("Video encoder stopped: " + ffmpeg_error())
        else:
            save_uncompressed(os.path.join(out_path, f"frame_{idx:06d}.tif"), img16, 16)

    acc = TrailAccumulator(trail_mode, comet_length, gap_fill) if trail_mode else None
    n = len(files)
    last = None
    by_path = meteors["by_path"] if meteors else {}
    det_size = meteors["det_size"] if meteors else None
    glow = None
    marks = []                                    # (last_frame_index, center, radius)
    decay = 0.03 ** (1.0 / max(1.0, meteor_persist * fps)) if meteor_persist > 0 else 0.0
    try:
        for i, f in enumerate(files):
            if cancel():
                raise InterruptedError("Cancelled")
            progress(i, n, f"Rendering frame {i+1}/{n}")
            img = first if i == 0 else load_image(f)
            if fixer is not None:
                img = fixer.apply(img)              # repair hot pixels at full resolution first
            img = _fit_size(img, size)
            img = denoise_frame(img, denoise)
            if gains is not None:
                img = np.clip(img.astype(np.float32) * gains[i], 0, 65535).astype(np.uint16)
            raw = img
            if acc is not None:
                img = acc.add(img)
            if meteors:
                if glow is not None:
                    glow = glow * decay
                evs = by_path.get(f)
                if evs:
                    m = _event_mask(evs, det_size, size)
                    contrib = raw.astype(np.float32) * m
                    glow = contrib if glow is None else np.maximum(glow, contrib)
                    for e in evs:
                        sx = size[0] / det_size[0]
                        marks.append((i + max(1, int(fps)), (int(e["center"][0] * sx), int(e["center"][1] * sx)),
                                      int(e["length"] * sx * 0.7) + 25))
                if glow is not None:
                    img = np.maximum(img, np.clip(glow, 0, 65535).astype(np.uint16))
                if meteor_mark:
                    marks = [mk for mk in marks if mk[0] > i]
                    if marks:
                        img = img.copy()
                        for _, ctr, rad in marks:
                            cv2.circle(img, ctr, rad, (65535, 54000, 15000), max(2, size[0] // 600), cv2.LINE_AA)
            write(img, i + 1)
            last = img
        if proc is not None and hold_seconds > 0 and last is not None:
            for _ in range(int(round(hold_seconds * fps))):
                write(last, n)
        if proc is not None:
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("Video encoder failed: " + ffmpeg_error())
    except BaseException:
        if proc is not None:
            try: proc.stdin.close()
            except Exception: pass
            try: proc.kill()
            except Exception: pass
        raise
    finally:
        if errlog is not None:
            errlog.close()
    progress(n, n, "Done")
    return out_path


# ===== meteor detection =====
import shutil


def _meteor_gray(path, max_side=1600):
    ext = os.path.splitext(path)[1].lower()
    img = load_image(path, half_size=(ext in RAW_EXT))
    g = cv2.cvtColor((img >> 8).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    sc = max_side / max(h, w)
    if sc < 1:
        g = cv2.resize(g, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (0, 0), 1.0)


def _angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def detect_meteors(files, sensitivity=5, min_length=30, ignore_satellites=True,
                   progress=_noop, cancel=_never):
    """
    Finds short bright streaks that appear in one or two frames.
    sensitivity 1..10 (higher finds fainter meteors, but more false hits)
    min_length: shortest streak, in pixels at 1600 px image width
    ignore_satellites: drop streaks that continue over 3+ consecutive frames (planes, satellites)
    Returns {"events": [...], "rejected": int, "det_size": (w, h), "n": int}
    """
    n = len(files)
    if n < 2:
        raise ValueError("Need at least 2 photos to look for meteors")
    thr = max(8, 46 - 4 * sensitivity)
    k5 = np.ones((5, 5), np.uint8)
    cands = []
    prev_g, cur_g, next_g = None, _meteor_gray(files[0]), _meteor_gray(files[1])
    det_size = (cur_g.shape[1], cur_g.shape[0])
    diag = (det_size[0] ** 2 + det_size[1] ** 2) ** 0.5
    for i in range(n):
        if cancel():
            raise InterruptedError("Cancelled")
        progress(i, n, f"Looking for meteors {i+1}/{n}")
        refs = [g for g in (prev_g, next_g) if g is not None]
        ref = cv2.dilate(refs[0] if len(refs) == 1 else np.maximum(refs[0], refs[1]), k5)
        mask = (cv2.subtract(cur_g, ref) > thr).astype(np.uint8)
        if 0 < mask.sum() and mask.mean() < 0.04:       # skip global flashes (clouds, headlights)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for k in range(1, num):
                if stats[k, cv2.CC_STAT_AREA] < 6:
                    continue
                x, y, w, h = [int(v) for v in stats[k, :4]]
                comp = labels[y:y + h, x:x + w] == k
                ys, xs = np.nonzero(comp)
                pts = np.column_stack([xs + x, ys + y]).astype(np.float32)
                (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
                long_, short_ = max(rw, rh), max(min(rw, rh), 1.0)
                if long_ < min_length or long_ / short_ < 3.5:
                    continue
                ang = (ang if rw >= rh else ang + 90) % 180
                cands.append(dict(frame=i, path=files[i], length=float(long_), angle=float(ang),
                                  center=(float(cx), float(cy)), bbox=(x, y, w, h),
                                  patch=comp.astype(np.uint8)))
        prev_g, cur_g = cur_g, next_g
        next_g = _meteor_gray(files[i + 2]) if i + 2 < n else None

    rejected = 0
    if ignore_satellites and cands:
        parent = list(range(len(cands)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        by_frame = {}
        for idx, c in enumerate(cands):
            by_frame.setdefault(c["frame"], []).append(idx)
        for idx, c in enumerate(cands):
            for pj in by_frame.get(c["frame"] - 1, []):
                p = cands[pj]
                d = ((c["center"][0] - p["center"][0]) ** 2 + (c["center"][1] - p["center"][1]) ** 2) ** 0.5
                if _angle_diff(c["angle"], p["angle"]) < 20 and d < 0.35 * diag:
                    parent[find(idx)] = find(pj)
        sizes = {}
        for idx in range(len(cands)):
            sizes[find(idx)] = sizes.get(find(idx), 0) + 1
        kept = [c for idx, c in enumerate(cands) if sizes[find(idx)] < 3]
        rejected = len(cands) - len(kept)
        cands = kept
    progress(n, n, "Done")
    return {"events": cands, "rejected": rejected, "det_size": det_size, "n": n}


def _event_mask(evs, det_size, out_size):
    """Soft 0..1 mask (H, W, 1) of the meteor streaks of one frame at output size."""
    dw, dh = det_size
    ow, oh = out_size
    m = np.zeros((dh, dw), np.uint8)
    for e in evs:
        x, y, w, h = e["bbox"]
        m[y:y + h, x:x + w] |= e["patch"]
    m = cv2.resize(m * 255, (ow, oh), interpolation=cv2.INTER_LINEAR)
    k = max(3, int(round(2.5 * ow / dw)) * 2 + 1)
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, k / 3))
    return (m.astype(np.float32) / 255.0)[..., None]


def _group_by_path(events):
    by = {}
    for e in events:
        by.setdefault(e["path"], []).append(e)
    return by


def make_meteor_composite(files, events, det_size, progress=_noop, cancel=_never):
    """One sky photo (middle frame) with every detected meteor added on top (full resolution)."""
    if not events:
        raise ValueError("No meteors to combine - run 'Find meteors' first")
    out = load_image(files[len(files) // 2])
    h, w = out.shape[:2]
    by = _group_by_path(events)
    for j, (p, evs) in enumerate(by.items()):
        if cancel():
            raise InterruptedError("Cancelled")
        progress(j, len(by), f"Adding meteor {j+1}/{len(by)}")
        fr = load_image(p)
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        m = _event_mask(evs, det_size, (w, h))
        out = np.maximum(out, np.clip(fr.astype(np.float32) * m, 0, 65535).astype(np.uint16))
    progress(len(by), len(by), "Done")
    return out


def export_meteor_frames(files, events, folder, progress=_noop, cancel=_never):
    """Copy the original photos that contain meteors into a folder and write meteors.csv."""
    os.makedirs(folder, exist_ok=True)
    by = _group_by_path(events)
    rows = ["file,streak_length_px,angle_deg"]
    for j, (p, evs) in enumerate(sorted(by.items(), key=lambda kv: natural_key(kv[0]))):
        if cancel():
            raise InterruptedError("Cancelled")
        progress(j, len(by), f"Copying {os.path.basename(p)}")
        shutil.copy2(p, os.path.join(folder, os.path.basename(p)))
        for e in evs:
            rows.append(f'"{os.path.basename(p)}",{e["length"]:.0f},{e["angle"]:.0f}')
    with open(os.path.join(folder, "meteors.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    progress(len(by), len(by), "Done")
    return folder


# ===== starry sky art (generated in code, no image files needed) =====
def make_starry_sky(w, h, seed=7):
    """Night sky with a milky-way glow, star-trail arcs and stars. Returns RGB uint8."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny, nx = yy / h, xx / w
    top = np.array([3, 5, 18], np.float32)
    bot = np.array([46, 38, 96], np.float32)
    img = top + (bot - top) * (ny ** 1.6)[..., None]
    u = nx * 0.55 + ny * 0.85 - 0.62
    band = np.exp(-(u / 0.13) ** 2)
    tex = cv2.resize(rng.random((max(h // 10, 4), max(w // 10, 4))).astype(np.float32),
                     (w, h), interpolation=cv2.INTER_CUBIC)
    tex = cv2.GaussianBlur(tex, (0, 0), max(h, w) / 90)
    tex = (tex - tex.min()) / max(float(tex.max() - tex.min()), 1e-6)
    img += (band * (0.45 + 1.1 * tex) * 0.38)[..., None] * np.array([120, 135, 200], np.float32)
    arcs = np.zeros((h, w), np.uint8)
    cx, cy = int(w * 0.68), int(h * 1.12)
    for _ in range(max(10, w // 40)):
        r = int(rng.uniform(0.2, 1.25) * h)
        a0 = rng.uniform(190, 330)
        cv2.ellipse(arcs, (cx, cy), (r, r), 0, a0, a0 + rng.uniform(18, 55),
                    int(rng.uniform(70, 230)), 1, cv2.LINE_AA)
    arcs = cv2.GaussianBlur(arcs, (0, 0), 0.7).astype(np.float32) / 255.0
    img += arcs[..., None] * np.array([210, 225, 255], np.float32) * 0.8
    n = max(60, w * h // 1800)
    layer = np.zeros((h, w), np.float32)
    layer[rng.integers(0, h, n), rng.integers(0, w, n)] = rng.random(n) ** 3 * 255
    img += cv2.GaussianBlur(layer, (0, 0), 0.8)[..., None] * 4.0
    nb = max(5, n // 40)
    bright = np.zeros((h, w), np.float32)
    bright[rng.integers(0, h, nb), rng.integers(0, w, nb)] = 255
    img += cv2.GaussianBlur(bright, (0, 0), 2.2)[..., None] * np.array([1.0, 0.95, 0.85], np.float32) * 16
    return np.clip(img, 0, 255).astype(np.uint8)


def make_icon_array(size=256):
    """Rounded starry-sky app icon (RGBA uint8) with a bright guiding star."""
    img = make_starry_sky(size, size, seed=11).astype(np.float32)
    star = np.zeros((size, size), np.float32)
    c = (int(size * 0.5), int(size * 0.45))
    L = int(size * 0.34)
    cv2.line(star, (c[0] - L, c[1]), (c[0] + L, c[1]), 255, 2, cv2.LINE_AA)
    cv2.line(star, (c[0], c[1] - L), (c[0], c[1] + L), 255, 2, cv2.LINE_AA)
    d = int(L * 0.45)
    cv2.line(star, (c[0] - d, c[1] - d), (c[0] + d, c[1] + d), 160, 1, cv2.LINE_AA)
    cv2.line(star, (c[0] - d, c[1] + d), (c[0] + d, c[1] - d), 160, 1, cv2.LINE_AA)
    cv2.circle(star, c, max(3, size // 36), 255, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(star, (0, 0), size / 40) * 1.6 + cv2.GaussianBlur(star, (0, 0), size / 8) * 2.5
    img += (np.clip(star * 0.9 + glow, 0, 255))[..., None] * np.array([1.0, 0.93, 0.78], np.float32)
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    r = int(size * 0.22)
    m = np.zeros((size, size), np.uint8)
    cv2.rectangle(m, (r, 0), (size - r, size), 255, -1)
    cv2.rectangle(m, (0, r), (size, size - r), 255, -1)
    for p in [(r, r), (size - r, r), (r, size - r), (size - r, size - r)]:
        cv2.circle(m, p, r, 255, -1, cv2.LINE_AA)
    return np.dstack([rgb, m])


def save_icon(path="icon.ico"):
    """Write the starry-sky .ico used for the exe file icon (called by the build)."""
    Image.fromarray(make_icon_array(256), "RGBA").save(
        path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


# ===== GUI =====
from PySide6 import QtCore, QtGui, QtWidgets

APP_NAME = "Astro Star Stack Studio"
CREDIT = "Created by Vaishnav KV"
TAGLINE = "Star trails & timelapses from your night-sky photos"

STYLE = """
QWidget { background:#0a0f24; color:#dfe6ff; font-size:13px; }
QToolTip { background:#111a40; color:#ffffff; border:1px solid #f4b942; padding:4px; }
QGroupBox { border:1px solid #243060; border-radius:10px; margin-top:16px; padding:14px 10px 10px 10px;
            font-weight:bold; color:#ffd88a; }
QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 8px; }
QPushButton { background:#1a2552; border:1px solid #2f3f85; border-radius:8px; padding:7px 12px; }
QPushButton:hover { background:#24357a; }
QPushButton:disabled { color:#6a7399; background:#121a3a; border-color:#1d2750; }
QPushButton#primary { background:#f4b942; color:#1a1300; font-weight:bold; font-size:14px;
                      padding:11px; border:none; }
QPushButton#primary:hover { background:#ffd36b; }
QPushButton#primary:disabled { background:#4a4228; color:#8a8160; }
QPushButton#danger { background:#3a1d2e; border-color:#7a2f55; }
QListWidget { background:#0d1431; border:1px solid #243060; border-radius:8px; outline:0; }
QListWidget::item { padding:3px; }
QListWidget::item:selected { background:#24357a; color:#ffffff; }
QListWidget::indicator, QCheckBox::indicator { width:16px; height:16px; border:1px solid #5a6bb5;
                                               border-radius:4px; background:#0d1431; }
QListWidget::indicator:checked, QCheckBox::indicator:checked { background:#f4b942; border-color:#f4b942; }
QComboBox, QDoubleSpinBox, QSpinBox { background:#111a40; border:1px solid #2f3f85; border-radius:6px;
                                      padding:5px 8px; min-height:18px; }
QComboBox QAbstractItemView { background:#111a40; selection-background-color:#24357a; border:1px solid #2f3f85; }
QTabWidget::pane { border:1px solid #243060; border-radius:10px; top:-1px; }
QTabBar::tab { background:#111a40; padding:9px 12px; border-top-left-radius:8px; border-top-right-radius:8px;
               margin-right:3px; color:#9aa6d6; }
QTabBar::tab:selected { background:#1f2d6b; color:#ffd88a; font-weight:bold; }
QProgressBar { border:1px solid #2f3f85; border-radius:8px; text-align:center; background:#0d1431;
               min-height:18px; color:#ffffff; }
QProgressBar::chunk { background:#f4b942; border-radius:7px; }
QSlider::groove:horizontal { height:6px; background:#243060; border-radius:3px; }
QSlider::sub-page:horizontal { background:#f4b942; border-radius:3px; }
QSlider::handle:horizontal { background:#ffd36b; width:16px; margin:-6px 0; border-radius:8px; }
QScrollBar:vertical { background:#0a0f24; width:12px; }
QScrollBar::handle:vertical { background:#2f3f85; border-radius:5px; min-height:24px; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
QSplitter::handle { background:#0a0f24; }
QLabel#muted { color:#8f9bc8; }
QLabel#header { background:#0b1020; }
"""


def rgb8_to_qimage(arr):
    h, w, _ = arr.shape
    arr = np.ascontiguousarray(arr)
    return QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


def to_qimage(rgb16):
    return rgb8_to_qimage((rgb16 >> 8).astype(np.uint8))


def app_icon_pixmap(size=256):
    a = np.ascontiguousarray(make_icon_array(size))
    img = QtGui.QImage(a.data, size, size, 4 * size, QtGui.QImage.Format_RGBA8888).copy()
    return QtGui.QPixmap.fromImage(img)


class PreviewLabel(QtWidgets.QLabel):
    resized = QtCore.Signal()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.resized.emit()


class Worker(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda i, n, m: self.progress.emit(i, n, m),
                          cancel=lambda: self._cancel, **self.kwargs)
            self.done.emit(res)
        except InterruptedError:
            self.failed.emit("Cancelled")
        except Exception as e:  # noqa
            self.failed.emit(str(e))


TRAIL_OUTPUTS = [
    ("PNG 16-bit  (best quality)", ".png", 16, "PNG (*.png)"),
    ("PNG 8-bit", ".png", 8, "PNG (*.png)"),
    ("TIFF 16-bit (uncompressed)", ".tif", 16, "TIFF (*.tif)"),
    ("TIFF 8-bit (uncompressed)", ".tif", 8, "TIFF (*.tif)"),
    ("BMP 8-bit (uncompressed)", ".bmp", 8, "BMP (*.bmp)"),
]
TL_FORMATS = ["MP4  (H.264, high quality)", "Uncompressed AVI (8-bit)", "TIFF sequence (16-bit, uncompressed)"]
TL_FMT_KEYS = ["mp4", "avi_raw", "tiff_seq"]
TL_EFFECTS = ["Normal timelapse", "Growing star trails", "Comet star trails"]
TL_EFFECT_KEYS = [None, "lighten", "comet"]
DEFAULT_HINT = "Drop your night-sky photos here"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QtGui.QIcon(app_icon_pixmap(256)))
        self.resize(1320, 840)
        self.worker = None
        self.last_output = None
        self._sky = None
        self._preview_pix = None
        self._hint = DEFAULT_HINT

        # ================= left: photos =================
        photos = QtWidgets.QGroupBox("1   Add your photos")
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setToolTip("Tick the frames to use. Untick bad ones (planes, clouds, headlights).")
        self.list.currentItemChanged.connect(self.show_preview_of_current)
        b_folder = QtWidgets.QPushButton("📁  Add folder"); b_folder.clicked.connect(self.add_folder)
        b_files = QtWidgets.QPushButton("🖼  Add files"); b_files.clicked.connect(self.add_files)
        b_clear = QtWidgets.QPushButton("Clear"); b_clear.setObjectName("danger")
        b_clear.clicked.connect(lambda: (self.list.clear(), self.update_count()))
        b_all = QtWidgets.QPushButton("Tick all"); b_all.clicked.connect(lambda: self.set_checks(True))
        b_none = QtWidgets.QPushButton("Untick all"); b_none.clicked.connect(lambda: self.set_checks(False))
        b_off = QtWidgets.QPushButton("Untick selected"); b_off.clicked.connect(self.exclude_selected)
        self.sort_combo = QtWidgets.QComboBox()
        self.sort_combo.addItems(["Name (1, 2, 10, 100…)", "Date modified"])
        self.sort_combo.setToolTip("Order in which frames are stacked / rendered")
        self.sort_combo.currentIndexChanged.connect(self.sort_list)
        self.count_lbl = QtWidgets.QLabel("0 frames"); self.count_lbl.setObjectName("muted")
        hint = QtWidgets.QLabel("Tip: you can also drag & drop photos or a folder into this window.")
        hint.setObjectName("muted"); hint.setWordWrap(True)
        lv = QtWidgets.QVBoxLayout(photos)
        r1 = QtWidgets.QHBoxLayout(); r1.addWidget(b_folder); r1.addWidget(b_files); r1.addWidget(b_clear)
        r2 = QtWidgets.QHBoxLayout(); r2.addWidget(b_all); r2.addWidget(b_none); r2.addWidget(b_off)
        r3 = QtWidgets.QHBoxLayout(); r3.addWidget(QtWidgets.QLabel("Order:")); r3.addWidget(self.sort_combo, 1)
        lv.addLayout(r1); lv.addLayout(r2); lv.addLayout(r3); lv.addWidget(self.list, 1)
        lv.addWidget(self.count_lbl); lv.addWidget(hint)

        # ================= centre: preview =================
        self.preview = PreviewLabel()
        self.preview.resized.connect(self.refresh_preview)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(420, 320)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.preview.setStyleSheet("background:#050816;border:1px solid #243060;border-radius:10px;")

        # ================= right: create =================
        tabs = QtWidgets.QTabWidget()
        tabs.setUsesScrollButtons(False)

        def note(text):
            l = QtWidgets.QLabel(text); l.setObjectName("muted"); l.setWordWrap(True); return l

        def slider_row(sl, fmt="{}%"):
            lbl = QtWidgets.QLabel(fmt.format(sl.value())); lbl.setFixedWidth(42)
            sl.valueChanged.connect(lambda v: lbl.setText(fmt.format(v)))
            row = QtWidgets.QHBoxLayout(); row.addWidget(sl, 1); row.addWidget(lbl)
            return row

        def cleanup_widgets():
            hot = QtWidgets.QCheckBox("Remove hot pixels")
            hot.setToolTip("Finds pixels that stay bright in the same spot in your photos and repairs them.\n"
                           "Stops dotted lines in star trails and fixed bright dots in videos.")
            nr = QtWidgets.QComboBox(); nr.addItems(["Off", "Light (fast)", "Medium", "Strong (slower)"])
            nr.setToolTip("Light: removes colour speckle (fast).\nMedium/Strong: also smooth brightness noise - slower.\n"
                          "In timelapse it runs after resizing, so choosing a smaller video size is much faster.")
            return hot, nr
        self.hot_cb, self.nr_cb = cleanup_widgets()
        self.hot_cb2, self.nr_cb2 = cleanup_widgets()
        self.hot_cb.toggled.connect(self.hot_cb2.setChecked); self.hot_cb2.toggled.connect(self.hot_cb.setChecked)
        self.nr_cb.currentIndexChanged.connect(self.nr_cb2.setCurrentIndex)
        self.nr_cb2.currentIndexChanged.connect(self.nr_cb.setCurrentIndex)

        st = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(st)
        f.setVerticalSpacing(10)
        f.addRow(note("Combines every ticked photo into ONE star-trail picture."))
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["Full trails (lighten)", "Comet (fading trails)"])
        self.mode.setToolTip("Full trails keep every streak. Comet makes older parts of each trail fade out.")
        self.comet = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.comet.setRange(0, 100); self.comet.setValue(60)
        self.comet.setToolTip("Higher = longer comet tail (only used in Comet mode)")
        self.gap = QtWidgets.QCheckBox("Fill gaps between frames")
        self.gap.setToolTip("Smooths the tiny breaks in trails caused by shutter gaps")
        self.st_fmt = QtWidgets.QComboBox(); self.st_fmt.addItems([o[0] for o in TRAIL_OUTPUTS])
        self.b_trail = QtWidgets.QPushButton("✨  Create star trail image"); self.b_trail.setObjectName("primary")
        self.b_trail.clicked.connect(self.run_trails)
        f.addRow("Style", self.mode); f.addRow("Comet length", slider_row(self.comet)); f.addRow("", self.gap)
        f.addRow("", self.hot_cb); f.addRow("Noise reduction", self.nr_cb)
        f.addRow("Save as", self.st_fmt); f.addRow(self.b_trail)
        tabs.addTab(st, "⭐ Trails")

        tl = QtWidgets.QWidget(); g = QtWidgets.QFormLayout(tl)
        g.setVerticalSpacing(10)
        g.addRow(note("Turns your photos into a video. Choose a star-trail effect for growing or comet trails."))
        self.effect = QtWidgets.QComboBox(); self.effect.addItems(TL_EFFECTS)
        self.effect.currentIndexChanged.connect(self.effect_changed)
        self.tl_comet = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.tl_comet.setRange(0, 100); self.tl_comet.setValue(70)
        self.tl_comet.setToolTip("Higher = longer comet tail")
        self.tl_gap = QtWidgets.QCheckBox("Fill gaps between frames")
        self.hold = QtWidgets.QDoubleSpinBox(); self.hold.setRange(0, 30); self.hold.setValue(2); self.hold.setSuffix(" s")
        self.hold.setToolTip("Keep the finished trail on screen at the end of the video")
        self.fps = QtWidgets.QDoubleSpinBox(); self.fps.setRange(1, 120); self.fps.setValue(24); self.fps.setSuffix(" fps")
        self.res = QtWidgets.QComboBox(); self.res.addItems(["Original size", "3840 x 2160 (4K)", "1920 x 1080 (Full HD)", "1280 x 720 (HD)"])
        self.defl = QtWidgets.QCheckBox("Deflicker")
        self.defl_w = QtWidgets.QSpinBox(); self.defl_w.setRange(3, 51); self.defl_w.setValue(7)
        self.defl.setToolTip("Smooths brightness changes between frames (normal timelapse only)")
        self.tl_fmt = QtWidgets.QComboBox(); self.tl_fmt.addItems(TL_FORMATS)
        self.warn = note("MP4 files are small. Uncompressed AVI/TIFF are huge (1080p ≈ 6 MB per frame, 4K ≈ 25 MB).")
        self.b_tl = QtWidgets.QPushButton("🎬  Render timelapse video"); self.b_tl.setObjectName("primary")
        self.b_tl.clicked.connect(lambda: self.run_timelapse(False))
        g.addRow("Effect", self.effect); g.addRow("Comet length", slider_row(self.tl_comet)); g.addRow("", self.tl_gap)
        g.addRow("End hold", self.hold); g.addRow("Speed", self.fps); g.addRow("Size", self.res)
        g.addRow("", self.defl); g.addRow("Deflicker strength", self.defl_w)
        g.addRow("", self.hot_cb2); g.addRow("Noise reduction", self.nr_cb2)
        g.addRow("Save as", self.tl_fmt); g.addRow(self.warn); g.addRow(self.b_tl)
        tabs.addTab(tl, "🎞 Timelapse")
        self.effect_changed()

        # ---- meteors tab
        mt = QtWidgets.QWidget(); mf = QtWidgets.QFormLayout(mt); mf.setVerticalSpacing(8)
        mf.addRow(note("Finds shooting stars: short, bright streaks that show up in only one or two frames. "
                       "First click Find meteors, then use the buttons below."))
        self.m_sens = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.m_sens.setRange(1, 10); self.m_sens.setValue(5)
        self.m_sens.setToolTip("Higher finds fainter meteors but may add false detections")
        self.m_len = QtWidgets.QSpinBox(); self.m_len.setRange(10, 400); self.m_len.setValue(30); self.m_len.setSuffix(" px")
        self.m_len.setToolTip("Shortest streak to count (measured at 1600 px image width)")
        self.m_ignore = QtWidgets.QCheckBox("Ignore planes && satellites"); self.m_ignore.setChecked(True)
        self.m_ignore.setToolTip("Streaks that continue over 3 or more frames are not meteors")
        self.b_find = QtWidgets.QPushButton("☄  Find meteors"); self.b_find.setObjectName("primary")
        self.b_find.clicked.connect(self.run_meteor_search)
        self.m_summary = QtWidgets.QLabel("No search yet."); self.m_summary.setObjectName("muted"); self.m_summary.setWordWrap(True)
        self.meteor_list = QtWidgets.QListWidget(); self.meteor_list.setMinimumHeight(90); self.meteor_list.setMaximumHeight(130)
        self.meteor_list.currentItemChanged.connect(self.show_meteor)
        self.b_m_tick = QtWidgets.QPushButton("Tick only meteor frames"); self.b_m_tick.clicked.connect(self.tick_meteor_frames)
        self.b_m_export = QtWidgets.QPushButton("Export meteor photos…"); self.b_m_export.clicked.connect(self.export_meteors)
        self.b_m_pic = QtWidgets.QPushButton("Save meteors-only picture…"); self.b_m_pic.clicked.connect(self.save_meteor_picture)
        self.m_persist = QtWidgets.QDoubleSpinBox(); self.m_persist.setRange(0, 30); self.m_persist.setValue(2); self.m_persist.setSuffix(" s")
        self.m_persist.setToolTip("How long each meteor keeps glowing in the video after it appears")
        self.m_mark = QtWidgets.QCheckBox("Circle each meteor")
        self.b_m_video = QtWidgets.QPushButton("🎬  Timelapse with meteors highlighted…")
        self.b_m_video.clicked.connect(lambda: self.run_timelapse(True))
        mf.addRow("Sensitivity", slider_row(self.m_sens, "{}")); mf.addRow("Min. streak", self.m_len)
        mf.addRow("", self.m_ignore); mf.addRow(self.b_find); mf.addRow(self.m_summary); mf.addRow(self.meteor_list)
        mf.addRow(self.b_m_tick); mf.addRow(self.b_m_export); mf.addRow(self.b_m_pic)
        mf.addRow("Glow time", self.m_persist); mf.addRow("", self.m_mark); mf.addRow(self.b_m_video)
        mf.addRow(note("The video uses the speed, size and format chosen in the Timelapse tab."))
        tabs.addTab(mt, "☄ Meteors")
        self.meteor_result = None
        self.set_meteor_actions(False)

        create = QtWidgets.QGroupBox("2   Choose what to make")
        cv_ = QtWidgets.QVBoxLayout(create); cv_.addWidget(tabs)

        self.progress = QtWidgets.QProgressBar()
        self.status = QtWidgets.QLabel("Ready."); self.status.setWordWrap(True)
        self.cancel_btn = QtWidgets.QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(lambda: self.worker and self.worker.cancel())
        self.open_btn = QtWidgets.QPushButton("📂  Open output folder"); self.open_btn.hide()
        self.open_btn.clicked.connect(self.open_output)
        right_w = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right_w); rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(create, 1); rv.addWidget(self.status); rv.addWidget(self.progress)
        rr = QtWidgets.QHBoxLayout(); rr.addWidget(self.open_btn, 1); rr.addWidget(self.cancel_btn)
        rv.addLayout(rr)
        right_w.setFixedWidth(420)

        split = QtWidgets.QSplitter()
        split.addWidget(photos); split.addWidget(self.preview); split.addWidget(right_w)
        split.setStretchFactor(1, 1); split.setSizes([290, 610, 420]); split.setChildrenCollapsible(False)

        # ================= header & footer =================
        icon_lbl = QtWidgets.QLabel(); icon_lbl.setObjectName("header")
        icon_lbl.setPixmap(app_icon_pixmap(256).scaled(46, 46, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        title = QtWidgets.QLabel(APP_NAME); title.setObjectName("header")
        title.setStyleSheet("font-size:22px;font-weight:bold;color:#ffffff;background:#0b1020;")
        sub = QtWidgets.QLabel(TAGLINE); sub.setObjectName("header")
        sub.setStyleSheet("font-size:12px;color:#ffd88a;background:#0b1020;")
        tv = QtWidgets.QVBoxLayout(); tv.setSpacing(0); tv.addWidget(title); tv.addWidget(sub)
        header = QtWidgets.QWidget(); header.setStyleSheet("background:#0b1020;")
        hl = QtWidgets.QHBoxLayout(header); hl.setContentsMargins(16, 10, 16, 10)
        hl.addWidget(icon_lbl); hl.addSpacing(10); hl.addLayout(tv); hl.addStretch()
        footer = QtWidgets.QLabel(CREDIT); footer.setAlignment(QtCore.Qt.AlignCenter)
        footer.setStyleSheet("color:#9aa3b8;background:#0b1020;padding:7px;font-size:12px;")

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        body = QtWidgets.QWidget(); bl = QtWidgets.QVBoxLayout(body); bl.setContentsMargins(12, 8, 12, 8); bl.addWidget(split)
        v.addWidget(header); v.addWidget(body, 1); v.addWidget(footer)
        self.setCentralWidget(central)
        self.setAcceptDrops(True)
        self.refresh_preview()

    def effect_changed(self):
        trail = self.effect.currentIndex() != 0
        self.tl_comet.setEnabled(self.effect.currentIndex() == 2)
        self.tl_gap.setEnabled(trail)
        self.hold.setEnabled(trail)
        self.defl.setEnabled(not trail)
        self.defl_w.setEnabled(not trail)

    # ---- drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            paths += list_images(p) if os.path.isdir(p) else ([p] if is_supported(p) else [])
        self.add_paths(paths)

    # ---- list handling
    def add_paths(self, paths):
        existing = {self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())}
        for p in paths:
            if p in existing:
                continue
            it = QtWidgets.QListWidgetItem(os.path.basename(p))
            it.setData(QtCore.Qt.UserRole, p)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.list.addItem(it)
        self.sort_list()

    def sort_list(self):
        """Re-order the whole list (keeps ticked/unticked state)."""
        items = [(self.list.item(i).data(QtCore.Qt.UserRole), self.list.item(i).checkState())
                 for i in range(self.list.count())]
        mode = "date" if self.sort_combo.currentIndex() == 1 else "name"
        state = dict(items)
        self.list.blockSignals(True)
        self.list.clear()
        for p in sort_paths([p for p, _ in items], mode):
            it = QtWidgets.QListWidgetItem(os.path.basename(p))
            it.setData(QtCore.Qt.UserRole, p)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(state[p])
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.update_count()

    def add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose image folder")
        if d:
            self.add_paths(list_images(d))

    def add_files(self):
        pat = "Images (" + " ".join("*" + e for e in sorted(ALL_EXT)) + ")"
        fs, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose images", "", pat + ";;All files (*)")
        self.add_paths([f for f in fs if is_supported(f)])

    def set_checks(self, on):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self.update_count()

    def exclude_selected(self):
        for it in self.list.selectedItems():
            it.setCheckState(QtCore.Qt.Unchecked)
        self.update_count()

    def checked_files(self):
        return [self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == QtCore.Qt.Checked]

    def update_count(self):
        self.count_lbl.setText(f"{len(self.checked_files())} of {self.list.count()} frames ticked")

    # ---- preview
    def show_preview_of_current(self, cur, _prev):
        self.update_count()
        if cur is None:
            return
        try:
            self.show_image(load_preview(cur.data(QtCore.Qt.UserRole)))
        except Exception as e:
            self._preview_pix = None
            self._hint = f"Cannot preview this file ({e})"
            self.refresh_preview()

    def show_image(self, rgb16):
        self._preview_pix = QtGui.QPixmap.fromImage(to_qimage(rgb16))
        self.refresh_preview()

    def refresh_preview(self):
        size = self.preview.size()
        if size.width() < 10 or size.height() < 10:
            return
        if self._preview_pix:
            self.preview.setPixmap(self._preview_pix.scaled(size, QtCore.Qt.KeepAspectRatio,
                                                            QtCore.Qt.SmoothTransformation))
            return
        if self._sky is None:
            self._sky = QtGui.QPixmap.fromImage(rgb8_to_qimage(make_starry_sky(1280, 720, seed=7)))
        pm = self._sky.scaled(size, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        pm = pm.copy((pm.width() - size.width()) // 2, (pm.height() - size.height()) // 2,
                     size.width(), size.height())
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        f = QtGui.QFont(); f.setPointSize(20); f.setBold(True); p.setFont(f)
        p.setPen(QtGui.QColor("#ffffff"))
        p.drawText(QtCore.QRect(0, int(size.height() * 0.38), size.width(), 50), QtCore.Qt.AlignCenter, self._hint)
        f2 = QtGui.QFont(); f2.setPointSize(10); p.setFont(f2)
        p.setPen(QtGui.QColor("#ffd88a"))
        p.drawText(QtCore.QRect(0, int(size.height() * 0.38) + 50, size.width(), 30), QtCore.Qt.AlignCenter,
                   "JPG · PNG · TIFF · HEIC · DNG · RAW (CR2, NEF, ARW …)")
        p.end()
        self.preview.setPixmap(pm)

    # ---- jobs
    def set_busy(self, busy):
        self.cancel_btn.setEnabled(busy)
        self.b_trail.setEnabled(not busy)
        self.b_tl.setEnabled(not busy)
        self.b_find.setEnabled(not busy)

    def set_meteor_actions(self, on):
        for b in (self.b_m_tick, self.b_m_export, self.b_m_pic, self.b_m_video):
            b.setEnabled(on)

    # ---- meteors
    def run_meteor_search(self):
        def done(res):
            self.meteor_result = res
            ev = res["events"]
            self.meteor_list.clear()
            for e in sorted(ev, key=lambda e: e["frame"]):
                it = QtWidgets.QListWidgetItem(f"☄ Frame {e['frame'] + 1} · {os.path.basename(e['path'])} · {int(e['length'])} px")
                it.setData(QtCore.Qt.UserRole, e)
                self.meteor_list.addItem(it)
            msg = f"Found {len(ev)} meteor(s)"
            if res["rejected"]:
                msg += f" · ignored {res['rejected']} plane/satellite streak(s)"
            self.m_summary.setText(msg)
            self.status.setText("☄ " + msg)
            self.set_meteor_actions(len(ev) > 0)
            if not ev:
                QtWidgets.QMessageBox.information(self, "No meteors found",
                    "Nothing found. Try raising Sensitivity or lowering Min. streak.")
        self.start(detect_meteors, done, sensitivity=self.m_sens.value(), min_length=self.m_len.value(),
                   ignore_satellites=self.m_ignore.isChecked())

    def show_meteor(self, cur, _prev):
        if cur is None or not self.meteor_result:
            return
        e = cur.data(QtCore.Qt.UserRole)
        try:
            img = load_preview(e["path"])
            dw = self.meteor_result["det_size"][0]
            sc = img.shape[1] / dw
            x, y, w, h = e["bbox"]
            cv2.rectangle(img, (int(x * sc) - 14, int(y * sc) - 14), (int((x + w) * sc) + 14, int((y + h) * sc) + 14),
                          (65535, 54000, 15000), 2, cv2.LINE_AA)
            self.show_image(img)
        except Exception as ex:
            self.status.setText(f"Cannot preview: {ex}")

    def meteor_events(self):
        return self.meteor_result["events"] if self.meteor_result else []

    def tick_meteor_frames(self):
        paths = {e["path"] for e in self.meteor_events()}
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(QtCore.Qt.Checked if it.data(QtCore.Qt.UserRole) in paths else QtCore.Qt.Unchecked)
        self.update_count()

    def export_meteors(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder for the meteor photos")
        if d:
            self.start(export_meteor_frames, self.mark_saved, self.meteor_events(), d)

    def save_meteor_picture(self):
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save meteors-only picture", "meteors.png", "PNG (*.png)")
        if not out:
            return
        out = self.with_ext(out, ".png")

        def done(img):
            save_uncompressed(out, img, 16)
            self.show_image(img)
            self.mark_saved(out)
        self.start(make_meteor_composite, done, self.meteor_events(), self.meteor_result["det_size"])

    def start(self, fn, on_done, *args, **kw):
        if self.worker is not None and self.worker.isRunning():
            return
        files = self.checked_files()
        if not files:
            QtWidgets.QMessageBox.warning(self, "No photos", "Add some photos and tick the ones you want first.")
            return
        self.open_btn.hide()
        self.worker = Worker(fn, files, *args, **kw)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(lambda r: self.finish(on_done, r))
        self.worker.failed.connect(self.fail)
        self.set_busy(True)
        self.worker.start()

    def on_progress(self, i, n, msg):
        self.progress.setMaximum(max(n, 1)); self.progress.setValue(i); self.status.setText(msg)

    def finish(self, cb, res):
        self.set_busy(False)
        try:
            cb(res)
        except Exception as e:
            self.status.setText(f"Error: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def fail(self, msg):
        self.set_busy(False)
        self.status.setText(msg)
        if msg != "Cancelled":
            QtWidgets.QMessageBox.critical(self, "Error", msg)

    def mark_saved(self, path):
        self.last_output = path
        self.status.setText(f"✔ Saved: {path}")
        self.progress.setValue(self.progress.maximum())
        self.open_btn.show()

    def open_output(self):
        if self.last_output:
            folder = self.last_output if os.path.isdir(self.last_output) else os.path.dirname(self.last_output)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    @staticmethod
    def with_ext(path, ext):
        return path if os.path.splitext(path)[1].lower() == ext else path + ext

    def run_trails(self):
        _, ext, bits, flt = TRAIL_OUTPUTS[self.st_fmt.currentIndex()]
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save star trail", "startrail" + ext, flt)
        if not out:
            return
        out = self.with_ext(out, ext)
        mode = "lighten" if self.mode.currentIndex() == 0 else "comet"

        def done(img):
            save_uncompressed(out, img, bits)
            self.show_image(img)
            self.mark_saved(out)
        self.start(stack_star_trails, done, mode=mode, comet_length=self.comet.value() / 100,
                   gap_fill=self.gap.isChecked(), hot_pixels=self.hot_cb.isChecked(),
                   denoise=self.nr_cb.currentIndex())

    def run_timelapse(self, with_meteors=False):
        if with_meteors and not self.meteor_events():
            QtWidgets.QMessageBox.information(self, "No meteors", "Click 'Find meteors' first.")
            return
        fmt = TL_FMT_KEYS[self.tl_fmt.currentIndex()]
        if fmt == "tiff_seq":
            out = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder for TIFF frames")
        elif fmt == "mp4":
            out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save timelapse", "timelapse.mp4", "MP4 (*.mp4)")
            out = self.with_ext(out, ".mp4") if out else out
        else:
            out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save timelapse", "timelapse.avi", "AVI (*.avi)")
            out = self.with_ext(out, ".avi") if out else out
        if not out:
            return
        size = [None, (3840, 2160), (1920, 1080), (1280, 720)][self.res.currentIndex()]
        extra = {}
        if with_meteors:
            extra = dict(meteors={"by_path": _group_by_path(self.meteor_events()),
                                  "det_size": self.meteor_result["det_size"]},
                         meteor_persist=self.m_persist.value(), meteor_mark=self.m_mark.isChecked())
        self.start(render_timelapse, self.mark_saved, out,
                   fps=self.fps.value(), size=size, deflicker=self.defl.isChecked(),
                   deflicker_window=self.defl_w.value(), fmt=fmt,
                   trail_mode=TL_EFFECT_KEYS[self.effect.currentIndex()],
                   comet_length=self.tl_comet.value() / 100, gap_fill=self.tl_gap.isChecked(),
                   hold_seconds=self.hold.value(), hot_pixels=self.hot_cb2.isChecked(),
                   denoise=self.nr_cb2.currentIndex(), **extra)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QtGui.QIcon(app_icon_pixmap(256)))
    app.setStyleSheet(STYLE)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        report(traceback.format_exc())
        sys.exit(1)
