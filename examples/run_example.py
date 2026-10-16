"""
Runs the full pipeline on the bundled sample video (`example_video.tif`), so
the repo can be tried out with no dataset.

`example_video.tif` is a real excerpt (900 frames, cropped to 240x240 around
the cell -- verified to stay well inside that crop for the whole clip) from
one of the acquisitions used in the paper, saved as an OME-TIFF with a
proper T axis -- AICSImage reads it exactly like a .nd2 file, so
`pipeline.py` needs no changes to run on it. At the default FPS/ORIGINAL_FPS
ratio this gives 72 output frames (3.6s at 20 FPS).

Detection defaults to SAM 2 (`configs/default_params.py`'s own default,
itself seeded by a Hough circle fit -- see `cell_detection.py`'s
`detect_cell_sam`). If no checkpoint is configured (`config.SAM_CHECKPOINT`)
or SAM 2 / torch aren't installed, this falls back to plain Hough instead --
checked upfront here rather than relying on `run_full_pipeline`'s own
SAM-failure fallback, which goes to "ellipse" (a path with a known masking
artifact on this clip, see the project notes). Debug grids are turned off
(`compare_detections` always renders a Hough-vs-Ellipse comparison panel,
which needs an ellipse fit even when "hough" is the active method).
Everything else uses the defaults from `configs/default_params.py`.

`expected_output/` has the result of a reference run, so you can see what
this produces without running it yourself.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs import default_params as config
from pipeline import run_full_pipeline

EXAMPLE_VIDEO = os.path.join(os.path.dirname(__file__), "example_video.tif")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _resolve_detection_method() -> str:
    if config.DETECTION_METHOD != "sam":
        return config.DETECTION_METHOD
    try:
        import torch  # noqa: F401
        from sam2.build_sam import build_sam2  # noqa: F401
    except ImportError:
        print("[!] SAM 2 / torch not installed; falling back to Hough.")
        return "hough"
    if not os.path.isfile(config.SAM_CHECKPOINT):
        print(f"[!] SAM_CHECKPOINT not found ({config.SAM_CHECKPOINT}); falling back to Hough.")
        return "hough"
    return "sam"


def main() -> None:
    config.DETECTION_METHOD = _resolve_detection_method()
    config.DEBUG_DETECTION = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_info = run_full_pipeline(EXAMPLE_VIDEO, OUTPUT_DIR)

    if result_info is None:
        print("[!] Pipeline failed on the example video.")
        return

    print(f"\n[+] Done ({config.DETECTION_METHOD} detection). Output written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
