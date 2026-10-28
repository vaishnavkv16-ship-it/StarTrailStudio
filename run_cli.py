"""Command-line use:  python run_cli.py trails  <folder> out.tif [--comet 0.6]
                     python run_cli.py timelapse <folder> out.avi [--fps 24] [--deflicker]"""
import argparse, sys
from startrail.imageio_all import list_images, save_uncompressed
from startrail.engine import stack_star_trails, render_timelapse

def prog(i, n, msg): print(f"\r[{i}/{n}] {msg:60.60}", end="", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["trails", "timelapse"])
ap.add_argument("folder"); ap.add_argument("out")
ap.add_argument("--comet", type=float, default=None, help="0..1 comet length (omit for full trails)")
ap.add_argument("--gapfill", action="store_true")
ap.add_argument("--fps", type=float, default=24)
ap.add_argument("--deflicker", action="store_true")
ap.add_argument("--tiffseq", action="store_true")
a = ap.parse_args()
files = list_images(a.folder)
print(f"{len(files)} images found")
if a.cmd == "trails":
    img = stack_star_trails(files, "comet" if a.comet is not None else "lighten",
                            a.comet or 0.5, a.gapfill, progress=prog)
    save_uncompressed(a.out, img, 16)
else:
    render_timelapse(files, a.out, a.fps, deflicker=a.deflicker,
                     fmt="tiff_seq" if a.tiffseq else "avi_raw", progress=prog)
print("\nSaved", a.out)
