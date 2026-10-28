from __future__ import annotations
import os
import sys
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .imageio_all import list_images, is_supported, load_preview, save_uncompressed, ALL_EXT
from .engine import stack_star_trails, render_timelapse


def to_qimage(rgb16: np.ndarray) -> QtGui.QImage:
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

    def cancel(self): self._cancel = True

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda i, n, m: self.progress.emit(i, n, m),
                          cancel=lambda: self._cancel, **self.kwargs)
            self.done.emit(res)
        except InterruptedError:
            self.failed.emit("Cancelled")
        except Exception as e:  # noqa
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Star Trail Studio")
        self.resize(1250, 780)
        self.worker = None
        self.result = None

        # ---- left: file list
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.currentItemChanged.connect(self.show_preview_of_current)
        btn_folder = QtWidgets.QPushButton("Add folder…"); btn_folder.clicked.connect(self.add_folder)
        btn_files = QtWidgets.QPushButton("Add files…"); btn_files.clicked.connect(self.add_files)
        btn_clear = QtWidgets.QPushButton("Clear"); btn_clear.clicked.connect(self.list.clear)
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

        # star trails tab
        st = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(st)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["Full trails (lighten)", "Comet (fading trails)"])
        self.comet = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.comet.setRange(0, 100); self.comet.setValue(60)
        self.gap = QtWidgets.QCheckBox("Fill gaps between frames")
        self.st_fmt = QtWidgets.QComboBox(); self.st_fmt.addItems(["TIFF 16-bit (uncompressed)", "TIFF 8-bit (uncompressed)", "BMP 8-bit (uncompressed)"])
        b = QtWidgets.QPushButton("Create star trail…"); b.clicked.connect(self.run_trails)
        f.addRow("Mode", self.mode); f.addRow("Comet length", self.comet); f.addRow("", self.gap)
        f.addRow("Output", self.st_fmt); f.addRow(b)
        tabs.addTab(st, "Star trails")

        # timelapse tab
        tl = QtWidgets.QWidget(); g = QtWidgets.QFormLayout(tl)
        self.fps = QtWidgets.QDoubleSpinBox(); self.fps.setRange(1, 120); self.fps.setValue(24)
        self.res = QtWidgets.QComboBox(); self.res.addItems(["Original size", "3840 x 2160", "1920 x 1080", "1280 x 720"])
        self.defl = QtWidgets.QCheckBox("Deflicker"); self.defl_w = QtWidgets.QSpinBox(); self.defl_w.setRange(3, 51); self.defl_w.setValue(7)
        self.tl_fmt = QtWidgets.QComboBox(); self.tl_fmt.addItems(["Uncompressed AVI (8-bit)", "TIFF sequence (16-bit, uncompressed)"])
        warn = QtWidgets.QLabel("Uncompressed video is very large (1080p ≈ 6 MB/frame,\n4K ≈ 25 MB/frame). Check free disk space.")
        warn.setStyleSheet("color:#c80;")
        b2 = QtWidgets.QPushButton("Render timelapse…"); b2.clicked.connect(self.run_timelapse)
        g.addRow("Frame rate", self.fps); g.addRow("Resolution", self.res); g.addRow("", self.defl)
        g.addRow("Deflicker window", self.defl_w); g.addRow("Output", self.tl_fmt); g.addRow(warn); g.addRow(b2)
        tabs.addTab(tl, "Timelapse")

        self.progress = QtWidgets.QProgressBar()
        self.status = QtWidgets.QLabel("")
        self.cancel_btn = QtWidgets.QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(lambda: self.worker and self.worker.cancel())
        right = QtWidgets.QVBoxLayout(); right.addWidget(tabs); right.addStretch()
        right.addWidget(self.status); right.addWidget(self.progress); right.addWidget(self.cancel_btn)
        right_w = QtWidgets.QWidget(); right_w.setLayout(right); right_w.setFixedWidth(320)

        split = QtWidgets.QSplitter()
        split.addWidget(left_w); split.addWidget(self.preview); split.addWidget(right_w)
        split.setStretchFactor(1, 1); split.setSizes([280, 650, 320])
        self.setCentralWidget(split)
        self.setAcceptDrops(True)

    # ---- drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

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
            if p in existing: continue
            it = QtWidgets.QListWidgetItem(os.path.basename(p))
            it.setData(QtCore.Qt.UserRole, p)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.list.addItem(it)
        self.update_count()

    def add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose image folder")
        if d: self.add_paths(list_images(d))

    def add_files(self):
        pat = "Images (" + " ".join("*" + e for e in sorted(ALL_EXT)) + ")"
        fs, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose images", "", pat + ";;All files (*)")
        self.add_paths([f for f in fs if is_supported(f)])

    def set_checks(self, on):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self.update_count()

    def exclude_selected(self):
        for it in self.list.selectedItems(): it.setCheckState(QtCore.Qt.Unchecked)
        self.update_count()

    def checked_files(self):
        return [self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == QtCore.Qt.Checked]

    def update_count(self):
        self.count_lbl.setText(f"{len(self.checked_files())} of {self.list.count()} frames selected")

    # ---- preview
    def show_preview_of_current(self, cur, _prev):
        self.update_count()
        if cur is None: return
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
        super().resizeEvent(e); self.refresh_preview()

    # ---- jobs
    def start(self, fn, on_done, *args, **kw):
        files = self.checked_files()
        if not files:
            QtWidgets.QMessageBox.warning(self, "No images", "Add and check some images first."); return
        self.worker = Worker(fn, files, *args, **kw)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(lambda r: self.finish(on_done, r))
        self.worker.failed.connect(self.fail)
        self.cancel_btn.setEnabled(True)
        self.worker.start()

    def on_progress(self, i, n, msg):
        self.progress.setMaximum(max(n, 1)); self.progress.setValue(i); self.status.setText(msg)

    def finish(self, cb, res):
        self.cancel_btn.setEnabled(False); cb(res)

    def fail(self, msg):
        self.cancel_btn.setEnabled(False); self.status.setText(msg)
        if msg != "Cancelled": QtWidgets.QMessageBox.critical(self, "Error", msg)

    def run_trails(self):
        idx = self.st_fmt.currentIndex()
        ext, flt = [(".tif", "TIFF (*.tif)"), (".tif", "TIFF (*.tif)"), (".bmp", "BMP (*.bmp)")][idx]
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save star trail", "startrail" + ext, flt)
        if not out: return
        bits = 16 if idx == 0 else 8
        mode = "lighten" if self.mode.currentIndex() == 0 else "comet"

        def done(img):
            save_uncompressed(out, img, bits)
            self.show_image(img); self.status.setText(f"Saved {out}")
        self.start(stack_star_trails, done, mode=mode, comet_length=self.comet.value() / 100,
                   gap_fill=self.gap.isChecked())

    def run_timelapse(self):
        seq = self.tl_fmt.currentIndex() == 1
        if seq:
            out = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder for TIFF frames")
        else:
            out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save timelapse", "timelapse.avi", "AVI (*.avi)")
        if not out: return
        size = [None, (3840, 2160), (1920, 1080), (1280, 720)][self.res.currentIndex()]
        self.start(render_timelapse, lambda r: self.status.setText(f"Saved {r}"), out,
                   fps=self.fps.value(), size=size, deflicker=self.defl.isChecked(),
                   deflicker_window=self.defl_w.value(), fmt="tiff_seq" if seq else "avi_raw")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
