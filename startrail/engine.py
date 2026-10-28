"""Star-trail stacking and timelapse rendering. No GUI code here."""
from __future__ import annotations
import os
import subprocess
from typing import Callable, Sequence
import numpy as np
import cv2
import imageio_ffmpeg

from .imageio_all import load_image, save_uncompressed

Progress = Callable[[int, int, str], None]
Cancel = Callable[[], bool]


def _noop(*a): pass
def _never(): return False


def _fit_size(img: np.ndarray, size):
    if size is None:
        return img
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- star trails
def stack_star_trails(files: Sequence[str], mode: str = "lighten", comet_length: float = 0.5,
                      gap_fill: bool = False,
                      progress: Progress = _noop, cancel: Cancel = _never) -> np.ndarray:
    """
    mode: "lighten" (full trails) or "comet" (trails fade; comet_length 0..1, 1 = longest).
    gap_fill: also blend a 50/50 mix of consecutive frames to bridge shutter gaps.
    Streams one frame at a time, so memory use is ~3 frames regardless of count.
    """
    if not files:
        raise ValueError("No images selected")
    acc = None
    prev = None
    decay = 1.0 - (1.0 - comet_length) * 0.2 if mode == "comet" else 1.0
    # comet_length 1 -> decay 1.0 (no fade) ; 0 -> 0.8 per frame
    n = len(files)
    for i, f in enumerate(files):
        if cancel():
            raise InterruptedError("Cancelled")
        progress(i, n, f"Stacking {os.path.basename(f)}")
        img = load_image(f)
        if acc is None:
            acc = img.astype(np.uint16).copy()
        else:
            if img.shape != acc.shape:
                img = cv2.resize(img, (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_AREA)
            if mode == "comet":
                acc = np.maximum((acc.astype(np.float32) * decay).astype(np.uint16), img)
            else:
                acc = np.maximum(acc, img)
            if gap_fill and prev is not None:
                mid = ((prev.astype(np.uint32) + img.astype(np.uint32)) >> 1).astype(np.uint16)
                acc = np.maximum(acc, mid)
        prev = img
    progress(n, n, "Done")
    return acc


# ------------------------------------------------------------------ timelapse
def _luma(img16: np.ndarray) -> float:
    return float(img16[::8, ::8].mean())


def compute_deflicker_gains(files: Sequence[str], window: int = 7,
                            progress: Progress = _noop, cancel: Cancel = _never) -> list[float]:
    """Gain per frame so brightness follows a moving average (removes flicker)."""
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


def render_timelapse(files: Sequence[str], out_path: str, fps: float = 24.0,
                     size: tuple[int, int] | None = None, deflicker: bool = False,
                     deflicker_window: int = 7, fmt: str = "avi_raw",
                     progress: Progress = _noop, cancel: Cancel = _never) -> str:
    """
    fmt:
      "avi_raw"  - uncompressed AVI (rawvideo, 8-bit BGR24). Very large files, zero loss.
      "tiff_seq" - folder of uncompressed 16-bit TIFF frames (out_path is the folder).
    size: (w, h) output size; defaults to first frame size (made even for video).
    """
    if not files:
        raise ValueError("No images selected")
    gains = compute_deflicker_gains(files, deflicker_window, progress, cancel) if deflicker else None

    first = load_image(files[0])
    if size is None:
        size = (first.shape[1], first.shape[0])
    size = (size[0] // 2 * 2, size[1] // 2 * 2)

    proc = None
    if fmt == "avi_raw":
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
               "-c:v", "rawvideo", "-pix_fmt", "bgr24", "-an", out_path]
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=flags)
    elif fmt == "tiff_seq":
        os.makedirs(out_path, exist_ok=True)
    else:
        raise ValueError(f"Unknown format {fmt}")

    n = len(files)
    try:
        for i, f in enumerate(files):
            if cancel():
                raise InterruptedError("Cancelled")
            progress(i, n, f"Rendering frame {i+1}/{n}")
            img = first if i == 0 else load_image(f)
            img = _fit_size(img, size)
            if gains is not None:
                img = np.clip(img.astype(np.float32) * gains[i], 0, 65535).astype(np.uint16)
            if proc is not None:
                bgr8 = cv2.cvtColor((img >> 8).astype(np.uint8), cv2.COLOR_RGB2BGR)
                proc.stdin.write(np.ascontiguousarray(bgr8).tobytes())
            else:
                save_uncompressed(os.path.join(out_path, f"frame_{i+1:06d}.tif"), img, 16)
        if proc is not None:
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("FFmpeg failed to write the video")
    except BaseException:
        if proc is not None:
            try: proc.stdin.close()
            except Exception: pass
            proc.kill()
        raise
    progress(n, n, "Done")
    return out_path
