"""
Shared MediaPipe pose utilities for spinal / functional mobility analysis.

This module extracts the common pieces used across:
- Balance / SPPB stance
- Stand-and-reach
- Put-on-socks
- 4MWT
- Bending
- Cervical / thoracic rotation
- Getting up from floor
- 30s chair stand

Keep test-specific pipelines in notebooks or separate scripts.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import medfilt, savgol_filter


# MediaPipe Pose landmark indices
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30


MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def download_model(model_type: str = "lite", output_path: Optional[str] = None) -> str:
    """Download Pose Landmarker .task file if it does not already exist."""
    if model_type not in MODEL_URLS:
        raise ValueError(f"model_type must be one of {list(MODEL_URLS)}")
    if output_path is None:
        output_path = f"pose_landmarker_{model_type}.task"
    if not os.path.exists(output_path):
        print(f"Downloading Pose Landmarker ({model_type})...")
        urllib.request.urlretrieve(MODEL_URLS[model_type], output_path)
        print(f"Saved → {output_path}")
    return output_path


def create_landmarker(
    model_path: str,
    min_detection: float = 0.5,
    min_presence: float = 0.5,
    min_tracking: float = 0.5,
    num_poses: int = 1,
):
    """Create a VIDEO-mode PoseLandmarker."""
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection,
        min_pose_presence_confidence=min_presence,
        min_tracking_confidence=min_tracking,
    )
    return vision.PoseLandmarker.create_from_options(options)


def get_px(lm, w: int, h: int) -> np.ndarray:
    """2D pixel coordinates from a MediaPipe landmark."""
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def get_xyz(lm, w: int, h: int) -> np.ndarray:
    """Approximate 3D coordinates (z scaled by image width)."""
    try:
        return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float64)
    except Exception:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)


def visibility(lm, idx: int) -> float:
    try:
        return float(lm[idx].visibility)
    except Exception:
        return 0.0


def mid_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def fill_nan_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = np.isnan(a)
    if n.any() and (~n).any():
        a = a.copy()
        a[n] = np.interp(np.flatnonzero(n), np.flatnonzero(~n), a[~n])
    return a


def smooth_series(
    y: np.ndarray,
    fps: float,
    med_k: Optional[int] = None,
    sav_win: Optional[int] = None,
) -> np.ndarray:
    """Median filter + Savitzky-Golay smoothing."""
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return y
    if med_k is None:
        med_k = max(7, int(0.10 * fps) | 1)
    if med_k % 2 == 0:
        med_k += 1
    k = min(med_k, len(y) if len(y) % 2 == 1 else len(y) - 1)
    k = max(3, k)
    y = medfilt(y, kernel_size=k)

    if sav_win is None:
        sav_win = max(11, int(0.22 * fps) | 1)
    if sav_win >= len(y):
        sav_win = len(y) - 1 if len(y) % 2 == 0 else len(y)
        sav_win = max(5, sav_win)
    if sav_win % 2 == 0:
        sav_win -= 1
    try:
        y = savgol_filter(y, sav_win, 2)
    except Exception:
        pass
    return y


def joint_angle(a, b, c) -> float:
    """Interior angle at point b (0–180°)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def hip_flexion_deg(shoulder, hip, knee) -> float:
    """Higher value = more flexed (180 - interior angle)."""
    return max(0.0, 180.0 - joint_angle(shoulder, hip, knee))


def knee_flexion_deg(hip, knee, ankle) -> float:
    """Higher value = more flexed."""
    return max(0.0, 180.0 - joint_angle(hip, knee, ankle))


def trunk_flexion_from_vertical(shoulder, hip) -> float:
    """Angle of shoulder–hip vector from vertical. 0 ≈ upright."""
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return abs(float(np.degrees(np.arctan2(dx, -dy))))


def torso_elevation_deg(shoulder, hip) -> float:
    """0° ≈ lying horizontal, ~90° ≈ upright (lumbar / torso proxy)."""
    if np.any(np.isnan(shoulder)) or np.any(np.isnan(hip)):
        return np.nan
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    length = np.hypot(dx, dy)
    if length < 8.0:
        return np.nan
    vert = np.clip(-dy / length, -1.0, 1.0)
    return float(np.degrees(np.arcsin(vert)))


def cervical_yaw_deg(nose, left_sh, right_sh, scale_deg_per_px: float = 1.55) -> float:
    """Approximate cervical yaw from frontal view. + = left, - = right."""
    if np.any(np.isnan(nose)) or np.any(np.isnan(left_sh)) or np.any(np.isnan(right_sh)):
        return np.nan
    mid_sh_x = 0.5 * (left_sh[0] + right_sh[0])
    offset = nose[0] - mid_sh_x
    return float(offset * scale_deg_per_px)


def thoracic_yaw_deg(left_sh_xyz, right_sh_xyz) -> float:
    """Approximate thoracic rotation from shoulder-girdle orientation."""
    if np.any(np.isnan(left_sh_xyz)) or np.any(np.isnan(right_sh_xyz)):
        return np.nan
    dx = left_sh_xyz[0] - right_sh_xyz[0]
    dz = left_sh_xyz[2] - right_sh_xyz[2]
    return float(np.degrees(np.arctan2(dz, dx)))


class EMASmoother:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if new_val is None:
            return self.value
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1.0 - self.alpha) * self.value
        return self.value


def open_video(path: str):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, w, h, n


def make_writer(path: str, fps: float, w: int, h: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))
