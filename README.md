# Preprocessing

Video registration pipeline for live-cell microscopy (.nd2): normalize raw
frames, detect and track the cell, register it with Lucas-Kanade optical
flow, and apply temporal CLAHE contrast enhancement. This is the
preprocessing step that produces the stabilized, contrast-enhanced videos
used as input for the optical flow methods (see `CompFlow_Github/` /
`CompFlow-OFV`).

Evaluation/metrics code (registration-quality IoU/MSD analysis, the
downstream density-flow optical-flow estimation and its statistics) is
**not** included here -- out of scope for preprocessing itself.

## Structure

```
image_utils.py       frame normalization, contrast adjustment (imadjust), temporal smoothing
cell_detection.py     cell localization -- Hough circles, ellipse fit (Canny + morphology), or SAM 2
registration.py        Lucas-Kanade registration: initial pass + iterative backward/forward refinement
clahe.py                temporal CLAHE (one LUT per spatial block, computed over the whole video)
pipeline.py             orchestrates the steps above for one .nd2 file
main.py                  loops over a folder of .nd2 files, calls pipeline.py on each

configs/
    default_params.py   all pipeline parameters (paths, FPS, detection method, registration,
                        CLAHE) -- edit this to change behavior, or point main.py at a different
                        module with the same names

examples/
    example_video.tif              short real sample clip, runnable with no dataset / SAM 2 checkpoint
    run_example.py                  runs run_full_pipeline() on it (see Example section below)
    preprocessing_walkthrough.ipynb  step-by-step notebook version, with plots at each stage
    expected_output/                reference output, for a look without running anything
```

## Requirements

```bash
pip install -r requirements.txt
```

`DETECTION_METHOD = "sam"` (the default in `configs/default_params.py`) additionally needs:
- `torch`
- SAM 2: `pip install "git+https://github.com/facebookresearch/sam2.git"`, plus a checkpoint
  (e.g. `sam2.1_hiera_small.pt`, from the [official SAM 2 releases](https://github.com/facebookresearch/sam2#model-description))
  -- set `SAM_CHECKPOINT` in `configs/default_params.py` to its path.

Switch `DETECTION_METHOD` to `"ellipse"` or `"hough"` in `configs/default_params.py` to avoid the
SAM 2 dependency entirely (classical CV, no checkpoint needed).

## Usage

1. Edit `configs/default_params.py`: at minimum `INPUT_DIR` (folder of `.nd2` files),
  `OUTPUT_DIR`, and `SAM_CHECKPOINT` if using SAM 2.
2. ```bash
   python3 main.py
   ```
   Each `.nd2` file at least `MIN_SIZE_MB` big gets its own subfolder under `OUTPUT_DIR`, containing:
   - `RAW_norm_<fps>fps.mp4` -- normalized, unregistered (sanity check)
   - `ORIGINAL_norm_<fps>fps.mp4` -- registered
   - `CLAHE_<fps>fps.mp4` -- registered + CLAHE + background mask (the final output)
   - `debug_01_...png` / `debug_02_...png` -- detection debug grids (if `DEBUG_DETECTION=True`)

Or call the pipeline directly on one file:

```python
from pipeline import run_full_pipeline
run_full_pipeline("video.nd2", "output_dir/")
```

## Example

`examples/` has a real sample clip (`example_video.tif`, cropped to 240x240
around the cell -- an excerpt from one of the acquisitions used in the
paper) so the pipeline can be tried without a dataset. It's an OME-TIFF with
a proper time axis, so `AICSImage` (and therefore `run_full_pipeline`) reads
it exactly like a `.nd2` file -- no code changes needed. At the default
FPS/ORIGINAL_FPS ratio it gives 72 output frames (3.6s at 20 FPS).

Detection defaults to SAM 2 (this repo's own default, see Requirements
above), same as the real pipeline. If no checkpoint is configured or SAM 2 /
torch aren't installed, both the example and the notebook fall back to
Hough automatically -- so it still runs with neither.

**Set to 900 frames (~87MB) to stay under GitHub's 100MB per-file limit** --
that's the practical ceiling for this crop size/compression; more frames
would risk the file failing to push.

```bash
python3 examples/run_example.py
```

Writes `RAW_norm_20fps.mp4` / `ORIGINAL_norm_20fps.mp4` / `CLAHE_20fps.mp4`
to `examples/output/` (see `examples/run_example.py` for the couple of
config overrides it uses, and why). `examples/expected_output/` has the
result of a reference run, if you just want to see what it produces.

For a step-by-step look at what each stage does -- with a plot after every
step (normalization, detection, before/after registration, CLAHE) -- open
`examples/preprocessing_walkthrough.ipynb`. It calls the same functions
individually instead of going through `run_full_pipeline` in one shot, and
already has its outputs rendered if you just want to read it.
