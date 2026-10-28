"""Universal image loading -> RGB uint16 arrays, and uncompressed saving."""
from __future__ import annotations
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


def list_images(folder: str) -> list[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if is_supported(f)]
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files


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
    """Save as uncompressed TIFF (16/8-bit) or BMP (8-bit). Format chosen by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = rgb16 if bits == 16 else (rgb16 >> 8).astype(np.uint8)
        # compression=1 -> no compression
        ok = cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                         [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        if not ok:
            Image.fromarray(arr).save(path, compression=None)
    elif ext == ".bmp":
        arr = (rgb16 >> 8).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        raise ValueError("Uncompressed output must be .tif/.tiff or .bmp")
    return path
