"""
Shared landmark -> feature conversion.
Used by BOTH training (landmark_classifier.py) and the live app (app.py),
so the two can never drift apart.
"""
import numpy as np


def from_mediapipe(hand_landmarks, w, h):
    """MediaPipe hand landmarks + image size -> 63-dim feature vector (or None).

    - pixel coordinates (so non-square frames are not distorted)
    - translated so the wrist is the origin   -> position invariant
    - divided by the largest wrist distance   -> scale/distance invariant
    """
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks.landmark],
                   dtype=np.float32)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[:, :2], axis=1).max())
    if scale < 1e-6:
        return None
    return (pts / scale).reshape(-1)


def mirror(feat):
    """Left<->right mirrored copy of a feature vector (makes model hand-agnostic)."""
    f = np.asarray(feat, dtype=np.float32).reshape(21, 3).copy()
    f[:, 0] *= -1
    return f.reshape(-1)
