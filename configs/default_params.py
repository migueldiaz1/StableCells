# Folder of raw .nd2 files to process.
INPUT_DIR = r"/path/to/your/nd2/videos"
# Where each video's output subfolder (RAW/ORIGINAL/CLAHE .mp4 + debug grids) gets written.
OUTPUT_DIR = r"/path/to/output"

# Minimum .nd2 file size (MB) to process -- main.py skips anything smaller
# (cheap way to filter out short/aborted acquisitions).
MIN_SIZE_MB = 500

# Target frame rate of the output videos (downsampled from ORIGINAL_FPS).
FPS = 20
# Acquisition frame rate of the raw .nd2 (used to compute the downsampling ratio).
ORIGINAL_FPS = 250.0
# Median blur kernel size (px) applied to the tracking copy of each frame
# before running Lucas-Kanade on it (odd number; 1 = no blur).
MEDIAN_BLUR = 1
# Number of backward+forward registration refinement cycles after the
# initial forward pass (0 = skip refinement, use the initial pass only).
N_ITER = 10

# Whether to apply temporal CLAHE after registration.
CLAHE = True
# CLAHE spatial grid size (num_blocks x num_blocks tiles, each with its own LUT).
CLAHE_BLOCKS = 16
# CLAHE histogram clip limit (fraction of a block's pixel count used as the clipping threshold).
CLAHE_CLIP_LIMIT = 0.04

# False: blend the background with a Gaussian-blurred version of itself (soft, still has texture).
# True: replace the background with a single flat value (the median background intensity),
# constant across all frames -- removes any background flicker entirely.
MEDIAN_BACKGROUND = False

# Temporal moving-average window (frames) applied to the registered output
# before CLAHE, 1 = no smoothing.
TEMPORAL_WINDOW = 10

# Which region of the detected cell mask to search for Lucas-Kanade
# keypoints in: "full" (the whole mask), "interior" (mask eroded inward,
# away from the noisy membrane), or "ring" (a band around the contour --
# dilated exterior minus eroded interior).
KPS_MODE = "full"
# Dilation iterations used to build the "ring" KPS_MODE band.
KPS_DILATE_ITER = 8
# Erosion iterations used to build the "interior"/"ring" KPS_MODE regions.
KPS_ERODE_ITER = 8

# Multiplier applied to the detected cell radius to get the keypoint-search
# radius used during registration (1.0 = use the full detected radius).
KPS_RADIUS_FACTOR = 1.0

# Cell localization method: "sam" (SAM 2, most accurate, needs a checkpoint
# + torch), "ellipse" (Canny edges + morphology + ellipse fit, classical CV,
# no extra dependencies), or "hough" (Hough circle transform only,
# fastest/simplest, used internally as the seed for the other two methods).
DETECTION_METHOD = "sam"
# Path to the SAM 2 checkpoint (.pt) -- only used when DETECTION_METHOD = "sam".
SAM_CHECKPOINT = r"/path/to/sam2.1_hiera_small.pt"
# SAM 2 model config (Hydra yaml) -- ships inside the sam2 package itself.
SAM_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
# "cuda" or "cpu" for SAM 2 inference.
SAM_DEVICE = "cuda"

# Whether to save the detection debug grids (debug_01/debug_02 .png,
# Hough-vs-Ellipse / SAM comparison panels) for each processed video.
DEBUG_DETECTION = True
