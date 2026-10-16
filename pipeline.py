
import os
import cv2
import numpy as np
from aicsimageio import AICSImage

from configs import default_params as config
from image_utils import *
from registration import initial_pass, refine_iterative
from clahe import apply_clahe_to_frames
from cell_detection import *


def _write_video(
    path: str,
    frames: list[np.ndarray],
    side: int,
    output_fps: int = 30,
) -> None:
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (side, side),
    )
    for f in frames:
        frame_u8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR))
    writer.release()


def run_full_pipeline(nd2_path: str, output_dir: str) -> dict | None:
    """
    Runs the full pipeline for one .nd2 file:
      1. Global normalization
      2. Initial detection (Hough/Ellipse/SAM) to find KPs and stabilize
      3. Initial registration pass and iterative refinement
      4. Temporal CLAHE
      5. Background mask and final video write
    """
    filename = os.path.basename(nd2_path)

    reader = AICSImage(nd2_path)
    num_frames = reader.dims.T
    global_max = compute_global_max(reader, channel=0)
    border = 30

    print("[+] Saving raw video (unregistered, normalized only)...")
    raw_frames = []
    raw_ratio = config.ORIGINAL_FPS / config.FPS
    num_raw_frames = int(num_frames / raw_ratio)

    for idx in range(num_raw_frames):
        f_idx = int(idx * raw_ratio)
        f_raw = reader.get_image_data("YX", T=f_idx, C=0)
        f_norm = np.clip(f_raw.astype(np.float32) / global_max, 0, 1)
        raw_frames.append(f_norm)

    raw_video_path = os.path.join(output_dir, f"RAW_norm_{config.FPS}fps.mp4")
    h_c, w_c = raw_frames[0].shape

    raw_writer = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*"mp4v"), config.FPS, (w_c, h_c))
    for f in raw_frames:
        f_u8 = (f * 255).astype(np.uint8)
        raw_writer.write(cv2.cvtColor(f_u8, cv2.COLOR_GRAY2BGR))
    raw_writer.release()
    print(f"[+] Raw video saved to: {raw_video_path}")

    sam_predictor = None
    if config.DETECTION_METHOD == "sam":
        try:
            sam_predictor = load_sam(
                checkpoint=config.SAM_CHECKPOINT,
                model_type=getattr(config, "SAM_MODEL_CONFIG", "vit_s"),
                device=getattr(config, "SAM_DEVICE", "cuda"),
            )
        except Exception as e:
            print(f"[!] Error loading SAM: {e}. Falling back to 'ellipse'.")
            config.DETECTION_METHOD = "ellipse"

    det_frame_raw = reader.get_image_data("YX", T=0, C=0)
    det_frame_u8 = normalize_frame(det_frame_raw, global_max)
    det_frame_adj = imadjust(det_frame_u8, tol=1)

    if config.DETECTION_METHOD == "sam":
        guide_info = detect_cell_sam(det_frame_adj, sam_predictor, border=border,
                                     save_debug=config.DEBUG_DETECTION)
    elif config.DETECTION_METHOD == "ellipse":
        guide_info = detect_cell_ellipse(det_frame_adj, border=border, solidity_threshold=0.6,
                                         save_debug=config.DEBUG_DETECTION)
    else:
        guide_info = detect_cell_hough(det_frame_adj, border=border,
                                       save_debug=config.DEBUG_DETECTION)

    if guide_info is None:
        print(f"[!] Error: no cell detected in {filename}. Skipping.")
        return None

    method_used = guide_info.get("method", config.DETECTION_METHOD)
    print(f"[+] Initial detection (KP guide) completed using: {method_used.upper()}")

    if config.DEBUG_DETECTION:
        debug_path_init = os.path.join(output_dir, "debug_01_initial_guide_detection.png")
        compare_detections(det_frame_adj, output_path=debug_path_init, detection_info=guide_info)
        print(f"[DEBUG] Initial guide detection grid saved to: {debug_path_init}")

    ratio = config.ORIGINAL_FPS / config.FPS
    num_output_frames = int(num_frames / ratio)

    print("[+] Preparing ultra-conservative registration...")

    registration_info = guide_info.copy()
    restricted_radius = int(guide_info["r"] * config.KPS_RADIUS_FACTOR)
    registration_info["r"] = restricted_radius
    crop_side = guide_info["crop_side"]
    registration_info["half"] = crop_side // 2

    print(f"[DEBUG] KP search area reduced from {guide_info['r']} to {restricted_radius} pixels.")

    clean_frames, tracking_frames = initial_pass(
        reader, num_output_frames, ratio,
        detection_info=registration_info,
        global_max=global_max,
        border=border,
        median_blur=config.MEDIAN_BLUR,
        kps_mode=config.KPS_MODE,
        kps_dilate_iter=config.KPS_DILATE_ITER,
        kps_erode_iter=config.KPS_ERODE_ITER,
    )

    if config.N_ITER > 0:
        clean_frames, tracking_frames = refine_iterative(
            clean_frames=clean_frames,
            tracking_frames=tracking_frames,
            n_iter=config.N_ITER,
            median_blur=config.MEDIAN_BLUR,
            kps_mode=config.KPS_MODE,
            kps_dilate_iter=config.KPS_DILATE_ITER,
            kps_erode_iter=config.KPS_ERODE_ITER,
        )

    if hasattr(config, 'TEMPORAL_WINDOW') and config.TEMPORAL_WINDOW > 1:
        print(f"[+] Applying temporal filter (window N={config.TEMPORAL_WINDOW})...")
        clean_frames = apply_temporal_moving_filter(clean_frames, config.TEMPORAL_WINDOW)

    print("[+] Adapting initial mask to the crop...")
    ref_idx = len(clean_frames) // 2
    ref_frame_raw = clean_frames[ref_idx]
    ref_frame_u8 = (np.clip(ref_frame_raw, 0, 1) * 255).astype(np.uint8)
    ref_frame_adj = imadjust(ref_frame_u8, tol=1)

    final_info = guide_info.copy()
    final_info["cx"] = final_info["half"]
    final_info["cy"] = final_info["half"]

    if guide_info.get("_sam_mask") is not None:
        crop_x1, crop_y1 = guide_info["crop_x1"], guide_info["crop_y1"]
        crop_x2, crop_y2 = guide_info["crop_x2"], guide_info["crop_y2"]
        side = guide_info["crop_side"]

        full_mask = guide_info["_sam_mask"]
        padded_mask = cv2.copyMakeBorder(full_mask, border, border, border, border,
                                         cv2.BORDER_CONSTANT, value=0)

        crop_mask = padded_mask[max(0, crop_y1):min(padded_mask.shape[0], crop_y2),
                                max(0, crop_x1):min(padded_mask.shape[1], crop_x2)]

        pad_t = max(0, -crop_y1); pad_b = max(0, crop_y2 - padded_mask.shape[0])
        pad_l = max(0, -crop_x1); pad_r = max(0, crop_x2 - padded_mask.shape[1])
        crop_mask = cv2.copyMakeBorder(crop_mask, pad_t, pad_b, pad_l, pad_r,
                                       cv2.BORDER_CONSTANT, value=0)

        if crop_mask.shape[0] != side or crop_mask.shape[1] != side:
            crop_mask = cv2.resize(crop_mask, (side, side))

        final_info["_sam_mask"] = crop_mask

    if config.DEBUG_DETECTION:
        debug_path_fine = os.path.join(output_dir, "debug_02_fine_redetection_pure_adj.png")
        compare_detections(ref_frame_adj, output_path=debug_path_fine, detection_info=final_info)
        print(f"[DEBUG] Re-detection grid (pure + adjust) saved to: {debug_path_fine}")

    original_video_path = os.path.join(output_dir, f"ORIGINAL_norm_{config.FPS}fps.mp4")
    print(f"[+] Saving original video (registered and normalized) to: {original_video_path}")
    _write_video(original_video_path, clean_frames, final_info["crop_side"], output_fps=config.FPS)

    if config.CLAHE:
        print("[+] Applying CLAHE to the registered video...")
        base_frames = apply_clahe_to_frames(
            frames=clean_frames,
            crop_side=final_info["crop_side"],
            half=final_info["half"],
            hough_r=final_info["r"],
            ellipse_info=final_info.get("_ellipse"),
            num_blocks=config.CLAHE_BLOCKS,
            clip_limit=config.CLAHE_CLIP_LIMIT,
        )
    else:
        base_frames = clean_frames

    binary_mask = np.zeros_like(ref_frame_u8)
    if final_info.get("_sam_mask") is not None:
        sam_mask = final_info["_sam_mask"]
        if sam_mask.shape != ref_frame_u8.shape:
            sam_mask = cv2.resize(sam_mask, (ref_frame_u8.shape[1], ref_frame_u8.shape[0]))
        binary_mask = sam_mask
    elif final_info.get("_ellipse") is not None:
        cv2.ellipse(binary_mask, final_info["_ellipse"], 255, -1)
    else:
        cv2.circle(
            binary_mask,
            (int(final_info["cx"]), int(final_info["cy"])),
            int(final_info["r"]),
            255, -1,
        )

    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, dilation_kernel, iterations=1)

    use_median_background = getattr(config, "MEDIAN_BACKGROUND", False)

    if use_median_background:
        print("[+] Applying CONSTANT background mask (median of the exterior)...")
        dilation_kernel = np.ones((7, 7), np.uint8)
        dilated_mask = cv2.dilate(dilated_mask, dilation_kernel, iterations=6)
        background_mask_bool = dilated_mask == 0

        ref_frame_float = np.clip(base_frames[ref_idx], 0, 1)
        background_pixels = ref_frame_float[background_mask_bool]
        if background_pixels.size > 0:
            constant_background_value = float(np.median(background_pixels))
        else:
            constant_background_value = 0.0
        print(f"[+] Constant background value computed: {constant_background_value:.4f} "
             f"({constant_background_value * 255:.1f}/255)")

        dist_input_mask = (dilated_mask > 0).astype(np.uint8) * 255
        dist_map = cv2.distanceTransform(dist_input_mask, cv2.DIST_L2, 5)

        solid_margin = 20
        fade_width = 15

        soft_mask = np.clip((dist_map - solid_margin) / fade_width, 0, 1)

        flat_background = np.full_like(ref_frame_float, constant_background_value)

        final_frames = []
        for f in base_frames:
            blended_frame = (f * soft_mask) + (flat_background * (1.0 - soft_mask))
            blended_frame[dilated_mask == 0] = constant_background_value
            final_frames.append(np.clip(blended_frame, 0, 1))

    else:
        print("[+] Applying Gaussian background mask outside the cell to remove noise...")

        mask = cv2.GaussianBlur(dilated_mask, (51, 51), 0).astype(np.float32) / 255.0

        final_frames = []
        for f in base_frames:
            blurred_frame = cv2.GaussianBlur(f, (51, 51), 0)
            blended_frame = (f * mask) + (blurred_frame * (1.0 - mask))
            final_frames.append(blended_frame)

    final_video_path = os.path.join(output_dir, f"CLAHE_{config.FPS}fps.mp4")
    _write_video(final_video_path, final_frames, guide_info["crop_side"], output_fps=config.FPS)
    print(f"[+] Video saved to: {final_video_path}")

    return final_info
