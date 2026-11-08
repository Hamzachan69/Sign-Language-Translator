"""
Shared landmark -> feature conversion.
Used by training (mlp_classifier.py, svm_classifier.py) AND the live app (app.py),
so training and inference can never drift apart.

Pipeline:  MediaPipe landmarks -> from_mediapipe() -> 63 raw numbers
           63 raw numbers      -> featurize()      -> 93 engineered numbers
"""
import numpy as np

# Finger chains (wrist first). Angles are measured at the 3 inner joints of each chain.
_CHAINS = [
    (0, 1, 2, 3, 4),       # thumb
    (0, 5, 6, 7, 8),       # index
    (0, 9, 10, 11, 12),    # middle
    (0, 13, 14, 15, 16),   # ring
    (0, 17, 18, 19, 20),   # pinky
]
_TIPS = [4, 8, 12, 16, 20]
_MCPS = [2, 5, 9, 13, 17]
N_RAW = 63
N_FEATURES = 63 + 10 + 15 + 5   # raw + tip-tip dists + joint angles + tip-to-knuckle dists


def from_mediapipe(hand_landmarks, w, h):
    """MediaPipe landmarks + image size -> 63-dim vector (or None).

    - pixel coordinates (non-square frames are not distorted)
    - wrist is the origin          -> position invariant
    - divided by max wrist distance -> scale / camera-distance invariant
    """
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks.landmark],
                   dtype=np.float32)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[:, :2], axis=1).max())
    if scale < 1e-6:
        return None
    return (pts / scale).reshape(-1)


def mirror(feat):
    """Left<->right mirrored copy of a 63-dim vector (makes the model hand-agnostic)."""
    f = np.asarray(feat, dtype=np.float32).reshape(21, 3).copy()
    f[:, 0] *= -1
    return f.reshape(-1)


def featurize(feat63):
    """63 raw numbers -> 93 numbers: raw + fingertip distances + joint angles + tip/knuckle distances.

    The extra features separate look-alike letters (M/N/S/T, U/V/R ...) because they
    describe finger BENDING and finger-to-finger contact directly.
    """
    p = np.asarray(feat63, dtype=np.float32).reshape(21, 3)

    # 1. distances between every pair of fingertips (10)
    tips = p[_TIPS]
    tip_d = [np.linalg.norm(tips[i] - tips[j]) for i in range(5) for j in range(i + 1, 5)]

    # 2. cosine of the bend angle at 3 joints per finger (15)
    cosines = []
    for chain in _CHAINS:
        for a, b, c in zip(chain[0:3], chain[1:4], chain[2:5]):
            v1, v2 = p[a] - p[b], p[c] - p[b]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            cosines.append(float(np.dot(v1, v2) / denom))

    # 3. fingertip to its own knuckle: how far each finger is curled/extended (5)
    curl = [np.linalg.norm(p[t] - p[m]) for t, m in zip(_TIPS, _MCPS)]

    return np.concatenate([p.reshape(-1), tip_d, cosines, curl]).astype(np.float32)


def featurize_many(X_raw):
    return np.vstack([featurize(r) for r in X_raw]).astype(np.float32)
