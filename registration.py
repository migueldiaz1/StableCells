
import cv2
import numpy as np

from image_utils import imadjust, prepare_frame


try:
    from skimage.measure import ransac as _sk_ransac
    from skimage.transform import EuclideanTransform as _SkEuclidean
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False

import os
from configs import default_params as config


def _kabsch(src, dst):
    if len(src) < 2:
        return None, None
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = mu_d - R @ mu_s
    return R, t


def _estimate_euclidean_ransac(src, dst, residual_threshold=2.0, max_trials=1000):
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 2:
        return None

    if _HAVE_SKIMAGE:
        model, inliers = _sk_ransac(
            (src, dst), _SkEuclidean,
            min_samples=2, residual_threshold=residual_threshold, max_trials=max_trials,
        )
        if model is None or inliers is None or int(inliers.sum()) < 2:
            return None
        return model.params.astype(np.float32)


    N = len(src)
    if N == 2:
        R, t = _kabsch(src, dst)
    else:
        rng = np.random.default_rng(42)
        best_n, best_inl = 0, None
        trial, dyn_max = 0, float(max_trials)
        while trial < min(max_trials, int(np.ceil(dyn_max))):
            idx = rng.choice(N, size=2, replace=False)
            R, t = _kabsch(src[idx], dst[idx])
            if R is None:
                trial += 1
                continue
            res = np.linalg.norm(src @ R.T + t - dst, axis=1)
            inl = res < residual_threshold
            n = int(inl.sum())
            if n > best_n:
                best_n, best_inl = n, inl
                ratio = n / N
                if ratio >= 1.0:
                    break
                denom = np.log(1.0 - ratio ** 2)
                if denom < 0:
                    dyn_max = np.log(1.0 - 0.99) / denom
            trial += 1
        if best_inl is None or best_n < 2:
            R, t = _kabsch(src, dst)
        else:
            R, t = _kabsch(src[best_inl], dst[best_inl])

    if R is None:
        return None
    M = np.eye(3, dtype=np.float32)
    M[:2, :2] = R
    M[:2, 2] = t
    return M
def _save_debug_lk_match(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    status: np.ndarray,
    output_path: str,
    title: str = "LK matches",
) -> None:
    h, w = img_prev.shape[:2]
    if img_prev.ndim == 2:
        left  = cv2.cvtColor(img_prev, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(img_curr, cv2.COLOR_GRAY2BGR)
    else:
        left, right = img_prev.copy(), img_curr.copy()

    canvas = np.hstack([left, right])

    pts_prev = pts_prev.reshape(-1, 2)
    pts_curr = pts_curr.reshape(-1, 2)
    status   = status.reshape(-1)

    n_ok = 0
    for (p0, p1, st) in zip(pts_prev, pts_curr, status):
        x0, y0 = int(round(p0[0])),       int(round(p0[1]))
        x1, y1 = int(round(p1[0])) + w,   int(round(p1[1]))
        color = (0, 255, 0) if st == 1 else (0, 0, 255)
        cv2.circle(canvas, (x0, y0), 3, color, -1)
        cv2.circle(canvas, (x1, y1), 3, color, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        if st == 1:
            n_ok += 1

    cv2.putText(canvas, f"{title}  ok={n_ok}/{len(status)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (w, 0), (w, h), (200, 200, 200), 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

def _save_debug_kps_gftt(
    img_gray: np.ndarray,
    mask: np.ndarray,
    kps: np.ndarray,
    output_path: str,
    title: str = "GFTT keypoints",
) -> None:
    if img_gray.ndim == 2:
        canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img_gray.copy()


    if mask is not None:
        overlay = canvas.copy()
        overlay[mask > 0] = (255, 100, 0)
        canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)


    n = 0
    if kps is not None and len(kps) > 0:
        for p in kps.reshape(-1, 2):
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(canvas, (x, y), 4, (0, 0, 0), 1)
            n += 1

    cv2.putText(canvas, f"{title} (n={n})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


def _mask_from_detection_info(
    h: int,
    w: int,
    cx: int,
    cy: int,
    r: int,
    ellipse_info: tuple = None,
    sam_mask: np.ndarray = None,
    dilate_iter: int = 8,
    erode_iter: int = 8,
    kps_mode: str = "ring",
) -> np.ndarray:
    kernel_an = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if sam_mask is not None:
        if sam_mask.shape != (h, w):
            interior = np.zeros((h, w), dtype=np.uint8)
            hh, ww = sam_mask.shape
            interior[:hh, :ww] = sam_mask
        else:
            interior = sam_mask.copy()
        interior = (interior > 0).astype(np.uint8) * 255

    elif ellipse_info is not None:
        interior = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(interior, ellipse_info, 255, -1)

    else:
        interior = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(interior, (cx, cy), r, 255, -1)


    if kps_mode == "full":

        mask_kps = interior
    elif kps_mode == "interior":

        mask_kps = cv2.erode(interior, kernel_an, iterations=erode_iter)
    else:
        exterior        = cv2.dilate(interior, kernel_an, iterations=dilate_iter)
        interior_eroded = cv2.erode(interior,  kernel_an, iterations=erode_iter)
        mask_kps        = cv2.subtract(exterior, interior_eroded)

    return mask_kps


def initial_pass(
    reader, num_output_frames, ratio, detection_info, global_max,
    border, median_blur,
    kps_mode: str = "ring",
    kps_dilate_iter: int = 8,
    kps_erode_iter: int = 8,
) -> tuple:
    crop_x1, crop_y1 = detection_info["crop_x1"], detection_info["crop_y1"]
    crop_x2, crop_y2 = detection_info["crop_x2"], detection_info["crop_y2"]
    padded_size       = detection_info["padded_size"]

    clean_frames     = [None] * num_output_frames
    tracking_frames  = [None] * num_output_frames


    frame_0_raw    = reader.get_image_data("YX", T=0, C=0)
    _, img_0_track = prepare_frame(frame_0_raw, global_max, border)
    h_p, w_p       = img_0_track.shape


    cx_pad = detection_info["cx"] + border
    cy_pad = detection_info["cy"] + border
    r_det  = detection_info["r"]

    ellipse_pad = None
    if detection_info.get("_ellipse") is not None:
        (ecx, ecy), axes, ang = detection_info["_ellipse"]
        ellipse_pad = ((ecx + border, ecy + border), axes, ang)

    sam_mask_pad = None
    if detection_info.get("_sam_mask") is not None:
        sam_mask_pad = cv2.copyMakeBorder(
            detection_info["_sam_mask"],
            border, border, border, border,
            cv2.BORDER_CONSTANT, value=0,
        )

    texture_mask = _mask_from_detection_info(
        h_p, w_p, cx_pad, cy_pad, r_det,
        ellipse_info=ellipse_pad,
        sam_mask=sam_mask_pad,
        kps_mode=kps_mode,
        dilate_iter=kps_dilate_iter,
        erode_iter=kps_erode_iter,
    )


    p0 = cv2.goodFeaturesToTrack(
        img_0_track, mask=texture_mask, maxCorners=400, qualityLevel=0.03, minDistance=5
    )
    if p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(
            img_0_track, mask=None, maxCorners=400, qualityLevel=0.03, minDistance=5
        )
    if p0 is None:
        p0 = np.zeros((0, 1, 2), dtype=np.float32)

    print(f"[+] KPs at start of initial pass: {len(p0)}")

    _save_debug_kps_gftt(
    img_0_track, texture_mask, p0,
    output_path=os.path.join(config.OUTPUT_DIR, "kps_gftt_initial.png"),
    title="GFTT keypoints (frame 0)",
)


    p_prev         = p0.copy()
    prev_track_img = img_0_track
    M_total        = np.eye(3, dtype=np.float32)

    print(f"-> Running initial pass ({num_output_frames} frames, forward only)...")

    for loop_idx, idx in enumerate(range(num_output_frames)):
        center_frame = idx * ratio


        start_frame = max(0, int(round(center_frame - ratio)))
        end_frame   = min(reader.dims.T, int(round(center_frame + ratio)))
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        raw_block = reader.get_image_data("TYX", T=slice(start_frame, end_frame), C=0)
        win_size  = raw_block.shape[0]
        if win_size > 1:
            x       = np.linspace(-1, 1, win_size)
            weights = np.exp(-0.5 * (x / 0.5) ** 2)
            weights /= weights.sum()
            raw_frame = np.sum(
                raw_block * weights[:, np.newaxis, np.newaxis], axis=0
            ).astype(np.uint16)
        else:
            raw_frame = raw_block[0]

        vis_img, track_img = prepare_frame(raw_frame, global_max, border)

        if loop_idx == 0:

            M_final        = M_total
            registered_img = vis_img
        else:

            curr_adj_img = imadjust(track_img, tol=1)

            p_curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_track_img, curr_adj_img, p_prev, None)
            good_curr = p_curr[st == 1]
            good_prev = p_prev[st == 1]

            if loop_idx == 1:
                _save_debug_lk_match(
                    prev_track_img, curr_adj_img,
                    p_prev, p_curr, st,
                    output_path=os.path.join(
                        config.OUTPUT_DIR, f"lk_match_{loop_idx-1:04d}_{loop_idx:04d}.png"
                    ),
                    title=f"LK frame {loop_idx-1} -> {loop_idx}",
                )

            if len(good_curr) > 10:


                M_step = _estimate_euclidean_ransac(good_curr, good_prev)
                if M_step is not None:
                    M_total = M_total @ M_step

                p_prev = good_curr.reshape(-1, 1, 2)

            M_final = M_total

            registered_img = cv2.warpAffine(
                vis_img, M_final[:2, :], padded_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
            )
            prev_track_img = curr_adj_img


        reg_h, reg_w = registered_img.shape
        y1_c = max(0, crop_y1);  y2_c = min(reg_h, crop_y2)
        x1_c = max(0, crop_x1);  x2_c = min(reg_w, crop_x2)
        temp_crop = registered_img[y1_c:y2_c, x1_c:x2_c]

        pad_t = max(0, -crop_y1);      pad_b = max(0, crop_y2 - reg_h)
        pad_l = max(0, -crop_x1);      pad_r = max(0, crop_x2 - reg_w)
        final_crop = cv2.copyMakeBorder(
            temp_crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
        )

        clean_frames[idx]    = final_crop.astype(np.float32) / 255.0
        tracking_frames[idx] = cv2.medianBlur(final_crop, median_blur).astype(np.float32) / 255.0

        if loop_idx % 50 == 0:
            print(f"  -> Registered {loop_idx}/{num_output_frames}", end="\r")

    print()
    return clean_frames, tracking_frames


def _register_pass(
    reference_frames, tracking_frames, order_indices,
    median_blur,
    detection_info=None,
    kps_mode: str = "ring",
    kps_dilate_iter: int = 8,
    kps_erode_iter: int = 8,
) -> tuple:
    num_frames = len(reference_frames)
    tracking_frames_out  = [None] * num_frames
    reference_frames_out = [None] * num_frames

    first_idx     = order_indices[0]
    init_frame_u8  = (tracking_frames[first_idx] * 255).astype(np.uint8)
    init_frame_adj = imadjust(init_frame_u8, tol=1)
    init_h, init_w = init_frame_adj.shape


    if detection_info is not None:
        half      = detection_info.get("half", init_w // 2)
        r_det     = detection_info["r"]
        ellipse_c  = None
        if detection_info.get("_ellipse") is not None:
            _, axes, ang = detection_info["_ellipse"]
            ellipse_c = ((float(half), float(half)), axes, ang)


        sam_mask_crop = detection_info.get("_sam_mask")

        texture_mask_init = _mask_from_detection_info(
        init_h, init_w, half, half, r_det,
        ellipse_info=ellipse_c,
        sam_mask=sam_mask_crop,
        kps_mode=kps_mode,
        dilate_iter=kps_dilate_iter,
        erode_iter=kps_erode_iter)
    else:

        texture_mask_init = np.zeros((init_h, init_w), dtype=np.uint8)
        cv2.circle(texture_mask_init, (init_w // 2, init_h // 2), min(init_w, init_h) // 3, 255, -1)

    p_prev = cv2.goodFeaturesToTrack(
        init_frame_adj, mask=texture_mask_init, maxCorners=400, qualityLevel=0.03, minDistance=5
    )
    if p_prev is None or len(p_prev) < 10:
        p_prev = cv2.goodFeaturesToTrack(
            init_frame_adj, mask=None, maxCorners=400, qualityLevel=0.03, minDistance=5
        )
    if p_prev is None:
        p_prev = np.zeros((0, 1, 2), dtype=np.float32)

    print(f"    KPs at start of pass: {len(p_prev)}")

    prev_track_img = init_frame_adj
    M_total        = np.eye(3, dtype=np.float32)


    tracking_frames_out[first_idx]  = tracking_frames[first_idx].copy()
    reference_frames_out[first_idx] = reference_frames[first_idx].copy()

    for loop_idx, idx in enumerate(order_indices[1:], start=1):
        curr_frame_u8  = (tracking_frames[idx] * 255).astype(np.uint8)
        curr_frame_adj = imadjust(curr_frame_u8, tol=1)


        p_curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_track_img, curr_frame_adj, p_prev, None)
        good_curr = p_curr[st == 1]
        good_prev = p_prev[st == 1]

        if len(good_curr) > 10:


            M_step = _estimate_euclidean_ransac(good_curr, good_prev)
            if M_step is not None:
                M_total = M_total @ M_step

            p_prev = good_curr.reshape(-1, 1, 2)

        M_final = M_total

        crop_h, crop_w = curr_frame_u8.shape


        ref_u8     = (reference_frames[idx] * 255).astype(np.uint8)
        warped_ref = cv2.warpAffine(
            ref_u8, M_final[:2, :], (crop_w, crop_h),
            borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )

        reference_frames_out[idx] = cv2.medianBlur(
            warped_ref, median_blur
        ).astype(np.float32) / 255.0


        warped_track = cv2.warpAffine(
            curr_frame_u8, M_final[:2, :], (crop_w, crop_h),
            borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )
        tracking_frames_out[idx] = warped_track.astype(np.float32) / 255.0

        prev_track_img = curr_frame_adj

        if loop_idx % 50 == 0:
            print(f"    -> frame {loop_idx}/{len(order_indices)-1}", end="\r")

    return tracking_frames_out, reference_frames_out


def refine_iterative(
    clean_frames, tracking_frames, n_iter,
    median_blur,
    detection_info=None,
    kps_mode: str = "ring",
    kps_dilate_iter: int = 8,
    kps_erode_iter: int = 8,
) -> tuple:
    num_output_frames = len(clean_frames)
    forward_idx  = list(range(num_output_frames))
    backward_idx = list(range(num_output_frames - 1, -1, -1))

    reference_frames = [f.copy() for f in clean_frames]
    tk_frames         = [f.copy() for f in tracking_frames]

    for i in range(n_iter):
        print(f"\n=== Iteration {i + 1}/{n_iter} -- backward pass ===")
        tk_frames, reference_frames = _register_pass(
            reference_frames, tk_frames, backward_idx,
            median_blur=median_blur,
            detection_info=detection_info,
            kps_mode=kps_mode,
            kps_dilate_iter=kps_dilate_iter,
            kps_erode_iter=kps_erode_iter,
        )

        print(f"\n=== Iteration {i + 1}/{n_iter} -- forward pass ===")
        tk_frames, reference_frames = _register_pass(
            reference_frames, tk_frames, forward_idx,
            median_blur=median_blur,
            detection_info=detection_info,
            kps_mode=kps_mode,
            kps_dilate_iter=kps_dilate_iter,
            kps_erode_iter=kps_erode_iter,
        )

    return reference_frames, tk_frames
