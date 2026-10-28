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
                      progress=_noop, cancel=_never):
    """Stack all frames into one image. Streams one frame at a time (low memory)."""
    if not files:
        raise ValueError("No images selected")
    acc = TrailAccumulator(mode, comet_length, gap_fill)
    n = len(files)
    for i, f in enumerate(files):
        if cancel():
            raise InterruptedError("Cancelled")
        progress(i, n, f"Stacking {os.path.basename(f)}")
        acc.add(load_image(f))
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
                     hold_seconds=0.0, progress=_noop, cancel=_never):
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
    try:
        for i, f in enumerate(files):
            if cancel():
                raise InterruptedError("Cancelled")
            progress(i, n, f"Rendering frame {i+1}/{n}")
            img = first if i == 0 else load_image(f)
            img = _fit_size(img, size)
            if acc is not None:
                img = acc.add(img)
            elif gains is not None:
                img = np.clip(img.astype(np.float32) * gains[i], 0, 65535).astype(np.uint16)
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


# ===== GUI =====
from PySide6 import QtCore, QtGui, QtWidgets

APP_NAME = "Astro Star Stack Studio"
CREDIT = "Created by Vaishnav KV"


def to_qimage(rgb16):
    rgb8 = np.ascontiguousarray((rgb16 >> 8).astype(np.uint8))
    h, w, _ = rgb8.shape
    return QtGui.QImage(rgb8.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


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
    ("PNG 16-bit", ".png", 16, "PNG (*.png)"),
    ("PNG 8-bit", ".png", 8, "PNG (*.png)"),
    ("TIFF 16-bit (uncompressed)", ".tif", 16, "TIFF (*.tif)"),
    ("TIFF 8-bit (uncompressed)", ".tif", 8, "TIFF (*.tif)"),
    ("BMP 8-bit (uncompressed)", ".bmp", 8, "BMP (*.bmp)"),
]
TL_FORMATS = ["MP4 (H.264, high quality)", "Uncompressed AVI (8-bit)", "TIFF sequence (16-bit, uncompressed)"]
TL_FMT_KEYS = ["mp4", "avi_raw", "tiff_seq"]
TL_EFFECTS = ["Normal timelapse", "Growing star trails", "Comet star trails"]
TL_EFFECT_KEYS = [None, "lighten", "comet"]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 820)
        self.worker = None

        # ---- left: file list
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.currentItemChanged.connect(self.show_preview_of_current)
        btn_folder = QtWidgets.QPushButton("Add folder…"); btn_folder.clicked.connect(self.add_folder)
        btn_files = QtWidgets.QPushButton("Add files…"); btn_files.clicked.connect(self.add_files)
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(lambda: (self.list.clear(), self.update_count()))
        btn_all = QtWidgets.QPushButton("Check all"); btn_all.clicked.connect(lambda: self.set_checks(True))
        btn_none = QtWidgets.QPushButton("Uncheck all"); btn_none.clicked.connect(lambda: self.set_checks(False))
        btn_sel_off = QtWidgets.QPushButton("Exclude selected"); btn_sel_off.clicked.connect(self.exclude_selected)
        self.count_lbl = QtWidgets.QLabel("0 frames")
        left = QtWidgets.QVBoxLayout()
        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(btn_folder); row1.addWidget(btn_files); row1.addWidget(btn_clear)
        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(btn_all); row2.addWidget(btn_none); row2.addWidget(btn_sel_off)
        left.addLayout(row1); left.addLayout(row2); left.addWidget(self.list); left.addWidget(self.count_lbl)
        left_w = QtWidgets.QWidget(); left_w.setLayout(left)

        # ---- centre: preview
        self.preview = QtWidgets.QLabel("Add images to begin.\nSupported: JPG, PNG, TIFF, HEIC/HEIF, DNG and most RAW formats")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(480, 360)
        self.preview.setStyleSheet("background:#111;color:#aaa;")
        self._preview_pix = None

        # ---- right: tabs
        tabs = QtWidgets.QTabWidget()

        st = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(st)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["Full trails (lighten)", "Comet (fading trails)"])
        self.comet = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.comet.setRange(0, 100); self.comet.setValue(60)
        self.gap = QtWidgets.QCheckBox("Fill gaps between frames")
        self.st_fmt = QtWidgets.QComboBox(); self.st_fmt.addItems([o[0] for o in TRAIL_OUTPUTS])
        self.b_trail = QtWidgets.QPushButton("Create star trail image…"); self.b_trail.clicked.connect(self.run_trails)
        f.addRow("Mode", self.mode); f.addRow("Comet length", self.comet); f.addRow("", self.gap)
        f.addRow("Save as", self.st_fmt); f.addRow(self.b_trail)
        tabs.addTab(st, "Star trails")

        tl = QtWidgets.QWidget(); g = QtWidgets.QFormLayout(tl)
        self.effect = QtWidgets.QComboBox(); self.effect.addItems(TL_EFFECTS)
        self.effect.currentIndexChanged.connect(self.effect_changed)
        self.tl_comet = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.tl_comet.setRange(0, 100); self.tl_comet.setValue(70)
        self.tl_gap = QtWidgets.QCheckBox("Fill gaps between frames")
        self.hold = QtWidgets.QDoubleSpinBox(); self.hold.setRange(0, 30); self.hold.setValue(2); self.hold.setSuffix(" s")
        self.fps = QtWidgets.QDoubleSpinBox(); self.fps.setRange(1, 120); self.fps.setValue(24)
        self.res = QtWidgets.QComboBox(); self.res.addItems(["Original size", "3840 x 2160", "1920 x 1080", "1280 x 720"])
        self.defl = QtWidgets.QCheckBox("Deflicker (normal timelapse only)")
        self.defl_w = QtWidgets.QSpinBox(); self.defl_w.setRange(3, 51); self.defl_w.setValue(7)
        self.tl_fmt = QtWidgets.QComboBox(); self.tl_fmt.addItems(TL_FORMATS)
        self.warn = QtWidgets.QLabel("Uncompressed video is very large (1080p ≈ 6 MB/frame,\n4K ≈ 25 MB/frame). MP4 is much smaller.")
        self.warn.setStyleSheet("color:#c80;")
        self.b_tl = QtWidgets.QPushButton("Render timelapse…"); self.b_tl.clicked.connect(self.run_timelapse)
        g.addRow("Effect", self.effect); g.addRow("Comet length", self.tl_comet); g.addRow("", self.tl_gap)
        g.addRow("Hold last frame", self.hold); g.addRow("Frame rate", self.fps); g.addRow("Resolution", self.res)
        g.addRow("", self.defl); g.addRow("Deflicker window", self.defl_w)
        g.addRow("Save as", self.tl_fmt); g.addRow(self.warn); g.addRow(self.b_tl)
        tabs.addTab(tl, "Timelapse")
        self.effect_changed()

        self.progress = QtWidgets.QProgressBar()
        self.status = QtWidgets.QLabel(""); self.status.setWordWrap(True)
        self.cancel_btn = QtWidgets.QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(lambda: self.worker and self.worker.cancel())
        right = QtWidgets.QVBoxLayout(); right.addWidget(tabs); right.addStretch()
        right.addWidget(self.status); right.addWidget(self.progress); right.addWidget(self.cancel_btn)
        right_w = QtWidgets.QWidget(); right_w.setLayout(right); right_w.setFixedWidth(340)

        split = QtWidgets.QSplitter()
        split.addWidget(left_w); split.addWidget(self.preview); split.addWidget(right_w)
        split.setStretchFactor(1, 1); split.setSizes([280, 660, 340])

        # ---- title bar + footer credit
        title = QtWidgets.QLabel(APP_NAME)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:20px;font-weight:bold;color:#e8eefc;background:#0b1020;padding:8px;")
        footer = QtWidgets.QLabel(CREDIT)
        footer.setAlignment(QtCore.Qt.AlignCenter)
        footer.setStyleSheet("color:#9aa3b8;background:#0b1020;padding:6px;font-size:12px;")
        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(title); v.addWidget(split, 1); v.addWidget(footer)
        self.setCentralWidget(central)
        self.setAcceptDrops(True)

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
        for p in sorted(paths, key=lambda x: os.path.basename(x).lower()):
            if p in existing:
                continue
            it = QtWidgets.QListWidgetItem(os.path.basename(p))
            it.setData(QtCore.Qt.UserRole, p)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.list.addItem(it)
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
        self.count_lbl.setText(f"{len(self.checked_files())} of {self.list.count()} frames selected")

    # ---- preview
    def show_preview_of_current(self, cur, _prev):
        self.update_count()
        if cur is None:
            return
        try:
            self.show_image(load_preview(cur.data(QtCore.Qt.UserRole)))
        except Exception as e:
            self.preview.setText(f"Cannot preview: {e}")

    def show_image(self, rgb16):
        self._preview_pix = QtGui.QPixmap.fromImage(to_qimage(rgb16))
        self.refresh_preview()

    def refresh_preview(self):
        if self._preview_pix:
            self.preview.setPixmap(self._preview_pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio,
                                                            QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh_preview()

    # ---- jobs
    def set_busy(self, busy):
        self.cancel_btn.setEnabled(busy)
        self.b_trail.setEnabled(not busy)
        self.b_tl.setEnabled(not busy)

    def start(self, fn, on_done, *args, **kw):
        if self.worker is not None and self.worker.isRunning():
            return
        files = self.checked_files()
        if not files:
            QtWidgets.QMessageBox.warning(self, "No images", "Add and check some images first.")
            return
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
            self.status.setText(f"Saved {out}")
        self.start(stack_star_trails, done, mode=mode, comet_length=self.comet.value() / 100,
                   gap_fill=self.gap.isChecked())

    def run_timelapse(self):
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
        self.start(render_timelapse, lambda r: self.status.setText(f"Saved {r}"), out,
                   fps=self.fps.value(), size=size, deflicker=self.defl.isChecked(),
                   deflicker_window=self.defl_w.value(), fmt=fmt,
                   trail_mode=TL_EFFECT_KEYS[self.effect.currentIndex()],
                   comet_length=self.tl_comet.value() / 100, gap_fill=self.tl_gap.isChecked(),
                   hold_seconds=self.hold.value())


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
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
