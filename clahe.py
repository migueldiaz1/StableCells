
import cv2
import numpy as np


def compute_global_luts(
    tube: np.ndarray,
    num_blocks: int = 16,
    clip_limit: float = 0.01,
    bins: int = 256,
    mask_sam: np.ndarray = None,
) -> np.ndarray:
    T, H, W = tube.shape
    step_y  = H / num_blocks
    step_x  = W / num_blocks
    LUTs    = np.zeros((num_blocks, num_blocks, bins), dtype=np.float32)


    if mask_sam is not None:
        mask_bool = mask_sam.astype(bool)
    else:
        mask_bool = None

    for i in range(num_blocks):
        for j in range(num_blocks):
            y_start = int(i * step_y)
            y_end   = int((i + 1) * step_y) if i < num_blocks - 1 else H
            x_start = int(j * step_x)
            x_end   = int((j + 1) * step_x) if j < num_blocks - 1 else W


            block_tube = tube[:, y_start:y_end, x_start:x_end]

            if mask_bool is not None:

                block_mask = mask_bool[y_start:y_end, x_start:x_end]


                valid_pixels = block_tube[:, block_mask]

                if valid_pixels.size == 0:

                    LUTs[i, j, :] = np.linspace(0, 1, bins)
                    continue

                block_uint = (valid_pixels.ravel() * (bins - 1)).astype(np.uint32)
                n_pixels   = valid_pixels.size

            else:

                block_uint = (block_tube * (bins - 1)).astype(np.uint32)
                n_pixels   = block_tube.size

            hist, _ = np.histogram(block_uint, bins=bins, range=(0, bins))


            clip_threshold = int(clip_limit * n_pixels)
            if clip_threshold > 0:
                clipped = 0
                for b in range(bins):
                    if hist[b] > clip_threshold:
                        clipped  += hist[b] - clip_threshold
                        hist[b]   = clip_threshold

                hist += clipped // bins


            cdf = hist.cumsum()
            LUTs[i, j, :] = (cdf / cdf[-1]) if cdf[-1] > 0 else np.linspace(0, 1, bins)

    return LUTs


def apply_luts_bilinear(
    frame: np.ndarray,
    LUTs: np.ndarray,
    num_blocks: int = 16,
    bins: int = 256,
) -> np.ndarray:
    H, W    = frame.shape
    step_y  = H / num_blocks
    step_x  = W / num_blocks


    LUTs_padded = np.pad(LUTs, ((1, 1), (1, 1), (0, 0)), mode='edge')


    y_idx, x_idx = np.arange(H), np.arange(W)
    xv, yv = np.meshgrid(x_idx, y_idx)

    cy = (yv - step_y / 2.0) / step_y
    cx = (xv - step_x / 2.0) / step_x


    y1, x1 = np.floor(cy).astype(int), np.floor(cx).astype(int)
    ty, tx  = cy - y1, cx - x1


    y1_p, x1_p = y1 + 1, x1 + 1
    y2_p, x2_p = y1_p + 1, x1_p + 1


    cf_q = np.clip(frame * (bins - 1), 0, bins - 1).astype(int)


    val_tl = LUTs_padded[y1_p, x1_p, cf_q]
    val_tr = LUTs_padded[y1_p, x2_p, cf_q]
    val_bl = LUTs_padded[y2_p, x1_p, cf_q]
    val_br = LUTs_padded[y2_p, x2_p, cf_q]


    top    = val_tl * (1 - tx) + val_tr * tx
    bottom = val_bl * (1 - tx) + val_br * tx
    return top * (1 - ty) + bottom * ty


def build_clahe_mask(
    crop_side: int,
    half: int,
    hough_r: int,
    ellipse_info: tuple = None,
    dilate_iter: int = 3,
    blur_ksize: int = 51,
) -> np.ndarray:
    m_u8 = np.zeros((crop_side, crop_side), dtype=np.uint8)

    if ellipse_info is not None:

        _, axes, ang = ellipse_info
        cv2.ellipse(m_u8, (half, half), (int(axes[0] / 2), int(axes[1] / 2)), ang, 0, 360, 255, -1)
    else:
        cv2.circle(m_u8, (half, half), int(hough_r), 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m_u8   = cv2.dilate(m_u8, kernel, iterations=dilate_iter)
    mask_f = cv2.GaussianBlur(m_u8.astype(np.float32), (blur_ksize, blur_ksize), 0) / 255.0
    return mask_f


def apply_clahe_to_frames(
    frames: list,
    crop_side: int,
    half: int,
    hough_r: int,
    ellipse_info: tuple = None,
    num_blocks: int = 16,
    clip_limit: float = 0.04,
    erode_it: int = 3,
    mask_sam: np.ndarray = None,
) -> list:
    video_volume = np.array(frames)


    fixed_luts = compute_global_luts(
        video_volume,
        num_blocks=num_blocks,
        clip_limit=clip_limit,
        mask_sam=mask_sam,
    )


    fixed_mask = build_clahe_mask(crop_side, half, hough_r, ellipse_info=ellipse_info)


    frames_out = []
    for f in frames:
        clahe_frame = apply_luts_bilinear(f, fixed_luts, num_blocks=num_blocks)
        final_frame = (clahe_frame * fixed_mask) + (f * (1.0 - fixed_mask))
        frames_out.append(np.clip(final_frame, 0, 1))

    return frames_out
