"""
Hand Tracker
============
Opens the webcam, detects one hand, and overlays:
  - 21 landmark dots (colour-coded per finger)
  - The full hand skeleton (connections between landmarks)
  - Landmark index labels
  - FPS counter

Controls:
  Q  →  Quit
"""

import cv2
import mediapipe as mp

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# Finger colours (BGR) — one per landmark group
FINGER_COLORS = {
    "wrist":      (255, 255, 255),   # white
    "thumb":      (0,   128, 255),   # orange
    "index":      (0,   255, 128),   # spring-green
    "middle":     (255, 0,   128),   # rose
    "ring":       (128, 0,   255),   # violet
    "pinky":      (255, 128, 0  ),   # sky-blue
}

# MediaPipe landmark index → finger name
LANDMARK_FINGER = {
    0:  "wrist",
    1:  "thumb",  2:  "thumb",  3:  "thumb",  4:  "thumb",
    5:  "index",  6:  "index",  7:  "index",  8:  "index",
    9:  "middle", 10: "middle", 11: "middle", 12: "middle",
    13: "ring",   14: "ring",   15: "ring",   16: "ring",
    17: "pinky",  18: "pinky",  19: "pinky",  20: "pinky",
}


def draw_skeleton(frame, hand_landmarks, w, h):
    """Draw connections (bones) and coloured landmark dots."""
    lm = hand_landmarks.landmark

    # Draw connections first (so dots render on top)
    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        x0 = int(lm[start_idx].x * w)
        y0 = int(lm[start_idx].y * h)
        x1 = int(lm[end_idx].x * w)
        y1 = int(lm[end_idx].y * h)
        cv2.line(frame, (x0, y0), (x1, y1), (200, 200, 200), 2, cv2.LINE_AA)

    # Draw landmark dots + index labels
    for idx, point in enumerate(lm):
        cx = int(point.x * w)
        cy = int(point.y * h)
        color = FINGER_COLORS[LANDMARK_FINGER[idx]]

        cv2.circle(frame, (cx, cy), 7,  color, -1,         cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 7,  (30, 30, 30), 1,   cv2.LINE_AA)  # dark border

        # Index label
        cv2.putText(
            frame, str(idx),
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            color, 1, cv2.LINE_AA,
        )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tick = cv2.getTickFrequency()

    with mp_hands.Hands(
        max_num_hands=1,          # track only one hand
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        prev_tick = cv2.getTickCount()

        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: Empty frame, skipping.")
                continue

            frame = cv2.flip(frame, 1)          # mirror so it feels natural
            h, w   = frame.shape[:2]

            # MediaPipe needs RGB
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            # ── Overlay ──────────────────────────────────────────────────
            if result.multi_hand_landmarks:
                draw_skeleton(frame, result.multi_hand_landmarks[0], w, h)

                # Handedness label (Left / Right)
                if result.multi_handedness:
                    label = result.multi_handedness[0].classification[0].label
                    score = result.multi_handedness[0].classification[0].score
                    cv2.putText(
                        frame, f"{label}  {score:.0%}",
                        (16, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 220), 2, cv2.LINE_AA,
                    )
            else:
                cv2.putText(
                    frame, "No hand detected",
                    (16, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (60, 60, 255), 2, cv2.LINE_AA,
                )

            # FPS counter
            now   = cv2.getTickCount()
            fps   = tick / (now - prev_tick)
            prev_tick = now
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (180, 255, 180), 2, cv2.LINE_AA,
            )

            cv2.imshow("Hand Tracker  –  press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

