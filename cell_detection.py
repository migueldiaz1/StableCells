
import cv2
import numpy as np


def detect_cell_hough(
    adj_frame: np.ndarray,
    border: int = 30,
    crop_factor: float = 2.8,
    save_debug: bool = False,
    target_w: int = 280,
) -> dict | None:

    h_p, w_p = adj_frame.shape
    f_blur = cv2.GaussianBlur(adj_frame, (11, 11), 0)

    circles = cv2.HoughCircles(
        f_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=int(w_p * 0.3),
        param1=60,
        param2=30,
        minRadius=int(w_p * 0.15),
        maxRadius=int(w_p * 0.45),
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    dists = np.sqrt((circles[:, 0] - w_p / 2) ** 2 + (circles[:, 1] - h_p / 2) ** 2)
    cx, cy, r = circles[np.argmin(dists)]

    result = _build_result_dict(cx, cy, r, border, crop_factor, w_p, h_p)

    if save_debug:

        canvas_cands = cv2.cvtColor(adj_frame.copy(), cv2.COLOR_GRAY2BGR)
        for ccx, ccy, cr in circles:
            cv2.circle(canvas_cands, (ccx, ccy), cr, (60, 220, 220), 1)
            cv2.circle(canvas_cands, (ccx, ccy), 3, (0, 0, 255), -1)


        canvas_res = cv2.cvtColor(adj_frame.copy(), cv2.COLOR_GRAY2BGR)
        cv2.circle(canvas_res, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(canvas_res, (cx, cy), 4, (0, 0, 255), -1)


        p1 = _add_header(_resize_bgr(adj_frame, target_w), "1. Adjusted frame")
        p2 = _add_header(_resize_bgr(f_blur, target_w),     "2. Gaussian blur (11x11)")

        cands_label = canvas_cands.copy()
        _draw_text(cands_label, f"{len(circles)} candidate(s)", (6, 16), color=(60, 220, 220))
        p3 = _add_header(_resize_bgr(cands_label, target_w), "3. Hough candidates")

        res_label = canvas_res.copy()
        _draw_text(res_label, f"r={r}px", (6, 16), color=(0, 255, 0))
        p4 = _add_header(_resize_bgr(res_label, target_w),  "4. Chosen circle")

        result["_debug"] = {"steps": [p1, p2, p3, p4]}

    return result


def detect_cell_ellipse(
    adj_frame: np.ndarray,
    border: int = 30,
    crop_factor: float = 2.8,
    solidity_threshold: float = 0.9,
    save_debug: bool = False,
    target_w: int = 280,
) -> dict | None:

    h_p, w_p = adj_frame.shape


    res_hough = detect_cell_hough(adj_frame, border, crop_factor,
                                  save_debug=save_debug, target_w=target_w)
    if not res_hough:
        return None


    mask_focus = np.zeros_like(adj_frame)
    cv2.circle(mask_focus,
               (res_hough["cx"], res_hough["cy"]),
               int(res_hough["r"] * 1.3), 255, -1)

    blur  = cv2.GaussianBlur(adj_frame, (11, 11), 0)
    edges = cv2.Canny(blur, 30, 80)
    edges_isolated = cv2.bitwise_and(edges, mask_focus)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges_isolated, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    def _build_ellipse_steps(final_result: dict) -> list[np.ndarray]:
        canvas_res = cv2.cvtColor(adj_frame.copy(), cv2.COLOR_GRAY2BGR)
        if final_result.get("method") == "ellipse":
            cv2.ellipse(canvas_res, final_result["_ellipse"], (0, 160, 255), 2)
            color, txt = (0, 160, 255), f"ELLIPSE (S:{final_result['solidity']:.2f})"
        else:
            cv2.circle(canvas_res,
                       (final_result["cx"], final_result["cy"]),
                       final_result["r"], (0, 255, 0), 2)
            color, txt = (0, 255, 0), "FALLBACK HOUGH"
        _draw_text(canvas_res, txt, (6, 18), color=color)

        p1 = _add_header(_resize_bgr(adj_frame,      target_w), "1. Frame")
        p2 = _add_header(_resize_bgr(edges_isolated, target_w), "2. Canny (Hough mask)")
        p3 = _add_header(_resize_bgr(closed,         target_w), "3. Hybrid closing")
        p4 = _add_header(_resize_bgr(canvas_res,     target_w), "4. Final result")
        return [p1, p2, p3, p4]


    if not contours:
        if save_debug:
            res_hough["_debug_ellipse"] = {"steps": _build_ellipse_steps(res_hough)}
        return res_hough

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best_contour = contours[0]

    if len(best_contour) < 5:
        if save_debug:
            res_hough["_debug_ellipse"] = {"steps": _build_ellipse_steps(res_hough)}
        return res_hough

    area      = cv2.contourArea(best_contour)
    hull      = cv2.convexHull(best_contour)
    hull_area = cv2.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0
    print("Solidity is: ", solidity)

    if solidity < solidity_threshold:
        if save_debug:
            res_hough["_debug_ellipse"] = {"steps": _build_ellipse_steps(res_hough)}
        return res_hough

    try:
        ellipse = cv2.fitEllipse(best_contour)
        (cx_e, cy_e), (axis_a, axis_b), _ = ellipse
        avg_r = int((axis_a + axis_b) / 4)

        result = _build_result_dict(int(cx_e), int(cy_e), avg_r, border, crop_factor, w_p, h_p,
                                    extra={"_ellipse": ellipse, "solidity": solidity, "method": "ellipse"})
        if save_debug:
            result["_debug_ellipse"] = {"steps": _build_ellipse_steps(result)}
            result["_debug"]         = res_hough.get("_debug")
        return result

    except Exception:
        if save_debug:
            res_hough["_debug_ellipse"] = {"steps": _build_ellipse_steps(res_hough)}
        return res_hough


def filter_largest_blob(mask_uint8: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 2:
        return mask_uint8

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    clean_mask = np.zeros_like(mask_uint8)
    clean_mask[labels == largest_label] = 255
    return clean_mask


def load_sam(
    checkpoint: str,
    model_type: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    device: str = "cuda",
):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and not torch.cuda.is_available():
        print("[SAM 2] CUDA not available, using CPU.")
        device = "cpu"

    print(f"[SAM 2] Loading model config={model_type} | weights={checkpoint} ({device})...")
    sam2_model = build_sam2(model_type, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("[SAM 2] Model loaded.")
    return predictor


def detect_cell_sam(
    adj_frame: np.ndarray,
    predictor,
    border: int = 40,
    crop_factor: float = 2.8,
    box_margin: float = 1.30,
    save_debug: bool = False,
    target_w: int = 280,
) -> dict | None:
    h_p, w_p = adj_frame.shape


    res_hough = detect_cell_hough(
        adj_frame, border=border, crop_factor=crop_factor,
        save_debug=save_debug, target_w=target_w,
    )
    if res_hough is None:
        return None

    cx_h, cy_h, r_h = res_hough["cx"], res_hough["cy"], res_hough["r"]


    side = int(r_h * box_margin)


    x1, y1 = max(0, cx_h - side), max(0, cy_h - side)
    x2, y2 = min(w_p, cx_h + side), min(h_p, cy_h + side)
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    offset = 5
    point_coords = np.array([
        [cx_h, cy_h],
        [x1 + offset, y1 + offset],
        [x2 - offset, y1 + offset],
        [x1 + offset, y2 - offset],
        [x2 - offset, y2 - offset]
    ], dtype=np.float32)

    point_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)


    img_rgb = cv2.cvtColor(adj_frame, cv2.COLOR_GRAY2RGB)
    predictor.set_image(img_rgb)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        print("[SAM 2] No mask obtained, falling back to Hough.")
        return res_hough


    expected_area = np.pi * r_h * r_h
    areas = [m.sum() for m in masks]
    diffs = [abs(a - expected_area) for a in areas]
    best_idx = int(np.argmin(diffs))

    raw_sam_mask = (masks[best_idx].astype(np.uint8)) * 255


    blob_mask = filter_largest_blob(raw_sam_mask)


    sam_mask = smooth_mask_morphology(blob_mask, ksize=7)


    M = cv2.moments(sam_mask)
    if M["m00"] == 0:
        return res_hough

    cx_sam = int(M["m10"] / M["m00"])
    cy_sam = int(M["m01"] / M["m00"])
    r_sam  = int(np.sqrt(M["m00"] / (255 * np.pi)))

    result = _build_result_dict(
        cx_sam, cy_sam, r_sam, border, crop_factor, w_p, h_p,
        extra={"_sam_mask": sam_mask, "method": "sam"},
    )


    if save_debug and res_hough.get("_debug"):
        result["_debug"] = res_hough["_debug"]

        canvas_sam = cv2.cvtColor(adj_frame.copy(), cv2.COLOR_GRAY2BGR)
        contours_sam, _ = cv2.findContours(
            sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas_sam, contours_sam, -1, (255, 100, 0), 2)
        _draw_text(canvas_sam, f"SAM 2 (area={int(M['m00']/255)}px)", (6, 18), color=(255, 100, 0))

        canvas_box = cv2.cvtColor(adj_frame.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas_box, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 200), 2)


        cv2.circle(canvas_box, (cx_h, cy_h), 4, (0, 255, 0), -1)
        cv2.circle(canvas_box, (int(x1+offset), int(y1+offset)), 4, (0, 0, 255), -1)
        cv2.circle(canvas_box, (int(x2-offset), int(y1+offset)), 4, (0, 0, 255), -1)
        cv2.circle(canvas_box, (int(x1+offset), int(y2-offset)), 4, (0, 0, 255), -1)
        cv2.circle(canvas_box, (int(x2-offset), int(y2-offset)), 4, (0, 0, 255), -1)

        _draw_text(canvas_box, "Box (1.2) + 5 points", (6, 18), color=(0, 200, 200))

        p1 = _add_header(_resize_bgr(adj_frame, target_w), "1. Frame")
        p2 = _add_header(_resize_bgr(canvas_box, target_w), "2. SAM 2 prompts")
        p3 = _add_header(_resize_bgr(sam_mask, target_w), "3. Clean SAM 2 mask")
        p4 = _add_header(_resize_bgr(canvas_sam, target_w), "4. SAM 2 contour")

        result["_debug_ellipse"] = {"steps": [p1, p2, p3, p4]}

    return result


import cv2
import numpy as np

def smooth_mask_morphology(binary_mask: np.ndarray, ksize: int = 7) -> np.ndarray:

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)


    smoothed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return smoothed_mask

def compare_detections(
    adj_frame: np.ndarray,
    output_path: str | None = None,
    detection_info: dict | None = None,
) -> np.ndarray:

    hough_steps = detection_info["_debug"]["steps"]
    ellipse_steps = detection_info["_debug_ellipse"]["steps"]

    SEP = 4

    def _row(p_left, p_right):
        pl, pr = _match_height(p_left, p_right)
        sep = np.zeros((pl.shape[0], SEP, 3), dtype=np.uint8)
        return np.hstack([pl, sep, pr])

    hough_block = np.vstack([
        _row(hough_steps[0], hough_steps[1]),
        np.zeros((SEP, hough_steps[0].shape[1] * 2 + SEP, 3), dtype=np.uint8),
        _row(hough_steps[2], hough_steps[3]),
    ])
    ellipse_block = np.vstack([
        _row(ellipse_steps[0], ellipse_steps[1]),
        np.zeros((SEP, ellipse_steps[0].shape[1] * 2 + SEP, 3), dtype=np.uint8),
        _row(ellipse_steps[2], ellipse_steps[3]),
    ])

    hough_block, ellipse_block = _match_height(hough_block, ellipse_block)

    center_sep = np.full((hough_block.shape[0], SEP * 2, 3), 60, dtype=np.uint8)
    body        = np.hstack([hough_block, center_sep, ellipse_block])

    width_h = hough_block.shape[1]
    width_e = ellipse_block.shape[1]
    header_h   = np.full((34, width_h,  3), (25, 45, 25), dtype=np.uint8)
    header_sep = np.full((34, SEP * 2,  3), 60,           dtype=np.uint8)
    header_e   = np.full((34, width_e,  3), (25, 25, 55), dtype=np.uint8)
    cv2.putText(header_h, "HOUGH CIRCLES",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 80), 1)
    cv2.putText(header_e, "ELLIPSE (Canny + morphology)",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 160, 255), 1)
    header = np.hstack([header_h, header_sep, header_e])

    grid = np.vstack([header, body])

    if output_path:
        cv2.imwrite(output_path, grid)

    return grid


def _build_result_dict(cx: int, cy: int, r: int,
                       border: int, crop_factor: float,
                       w_p: int, h_p: int,
                       extra: dict | None = None) -> dict:
    crop_side = int(r * crop_factor)
    if crop_side % 2 != 0:
        crop_side += 1
    half = crop_side // 2

    result = {
        "cx": cx, "cy": cy, "r": r,
        "crop_side": crop_side, "half": half,
        "crop_x1": int(cx + border - half),
        "crop_y1": int(cy + border - half),
        "crop_x2": int(cx + border - half) + crop_side,
        "crop_y2": int(cy + border - half) + crop_side,
        "padded_size": (w_p + 2 * border, h_p + 2 * border),
    }
    if extra:
        result.update(extra)
    return result


def _draw_text(img: np.ndarray, text: str, pos: tuple,
              color: tuple = (255, 255, 255), scale: float = 0.45) -> None:
    cv2.putText(img, text, (pos[0] + 1, pos[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def _add_header(img: np.ndarray, text: str,
                height: int = 26, bg=(40, 40, 40)) -> np.ndarray:
    h, w = img.shape[:2]
    header = np.full((height, w, 3), bg, dtype=np.uint8)
    cv2.putText(header, text, (6, height - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
    return np.vstack([header, img])


def _match_height(a: np.ndarray, b: np.ndarray):
    h = max(a.shape[0], b.shape[0])
    def _pad(img):
        diff = h - img.shape[0]
        if diff > 0:
            img = np.vstack([img, np.zeros((diff, img.shape[1], 3), dtype=np.uint8)])
        return img
    return _pad(a), _pad(b)


def _resize_bgr(img: np.ndarray, target_w: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (target_w, int(img.shape[0] * target_w / img.shape[1])))
