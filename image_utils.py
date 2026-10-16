
import bisect
import cv2
import numpy as np


def imadjust(src: np.ndarray, tol: float = 1, vout: tuple = (0, 255)) -> np.ndarray:
    assert len(src.shape) == 2, "Input image must be 2D"
    tol = max(0, min(100, tol))
    vin = [0, 255]

    if tol > 0:
        hist = np.histogram(src, bins=256, range=(0, 255))[0]
        cum  = np.cumsum(hist)
        total = src.shape[0] * src.shape[1]
        vin[0] = bisect.bisect_left(cum, total * tol / 100)
        vin[1] = bisect.bisect_left(cum, total * (100 - tol) / 100)

    scale = (vout[1] - vout[0]) / max(1, (vin[1] - vin[0]))
    vs = src.astype(np.float32) - vin[0]
    vs[vs < 0] = 0
    vd = vs * scale + 0.5 + vout[0]
    vd[vd > vout[1]] = vout[1]
    return vd.astype(np.uint8)


def normalize_frame(frame_u16: np.ndarray, global_max: float) -> np.ndarray:
    return (255.0 * (frame_u16.astype(np.float64) / global_max)).astype(np.uint8)

def apply_temporal_moving_filter(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if n <= 1:
        return frames

    num_frames = len(frames)

    stack = np.array(frames)
    result = np.zeros_like(stack)

    radius = n // 2

    for i in range(num_frames):

        start = max(0, i - radius)
        end = min(num_frames, i + radius + 1)


        result[i] = np.mean(stack[start:end], axis=0)

    return [f for f in result]


def prepare_frame(
    frame_u16: np.ndarray,
    global_max: float,
    border: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    frame_u8 = normalize_frame(frame_u16, global_max)
    adj_frame = imadjust(frame_u8, tol=1)

    frame_vis   = cv2.copyMakeBorder(frame_u8,  border, border, border, border, cv2.BORDER_REPLICATE)
    frame_track = cv2.copyMakeBorder(adj_frame, border, border, border, border, cv2.BORDER_REPLICATE)

    return frame_vis, frame_track


def compute_global_max(reader, channel: int = 0) -> float:
    full_volume = reader.get_image_data("TYX", C=channel)
    global_max = float(full_volume.max())
    del full_volume
    return global_max
