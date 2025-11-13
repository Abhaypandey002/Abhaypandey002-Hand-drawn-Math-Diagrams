import math
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Return the Euclidean distance between two (x, y) points."""
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def detect_hand(frame: np.ndarray, hands):
    """Return the first detected MediaPipe hand landmarks and handedness label."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return None, None

    hand_landmarks = result.multi_hand_landmarks[0]
    handedness_label = result.multi_handedness[0].classification[0].label
    return hand_landmarks, handedness_label


def landmarks_to_pixels(hand_landmarks, frame_w: int, frame_h: int) -> Dict[int, Tuple[int, int]]:
    """Convert normalized MediaPipe landmarks to pixel coordinates."""
    if hand_landmarks is None:
        return {}

    landmark_dict: Dict[int, Tuple[int, int]] = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        px, py = int(lm.x * frame_w), int(lm.y * frame_h)
        landmark_dict[idx] = (px, py)
    return landmark_dict


def get_extended_fingers(hand_landmarks, handedness_label: str) -> Set[str]:
    """Return a set with the names of the fingers that are extended."""
    if hand_landmarks is None or handedness_label is None:
        return set()

    lm = hand_landmarks.landmark
    fingers: Set[str] = set()

    if lm[8].y < lm[6].y - 0.02:
        fingers.add("index")
    if lm[12].y < lm[10].y - 0.02:
        fingers.add("middle")
    if lm[16].y < lm[14].y - 0.02:
        fingers.add("ring")
    if lm[20].y < lm[18].y - 0.02:
        fingers.add("pinky")

    if handedness_label == "Right":
        if lm[4].x < lm[3].x - 0.02:
            fingers.add("thumb")
    else:
        if lm[4].x > lm[3].x + 0.02:
            fingers.add("thumb")

    return fingers


def determine_candidate_gesture(extended: Set[str]) -> str:
    """Determine the candidate gesture string from the extended finger set."""
    if not extended:
        return "idle"

    if "index" in extended and "middle" in extended and len(extended) <= 3:
        return "draw"
    if extended == {"index"}:
        return "pause"
    if len(extended) >= 4:
        return "calculate"
    return "idle"


def compute_triangle_metrics(points: List[Optional[Tuple[int, int]]]) -> Optional[Dict[str, Any]]:
    """Compute side lengths and Pythagorean relationships for three vertices."""
    if any(pt is None for pt in points):
        return None

    a, b, c = [np.array(pt, dtype=np.float32) for pt in points]
    ab = float(np.linalg.norm(a - b))
    bc = float(np.linalg.norm(b - c))
    ca = float(np.linalg.norm(c - a))

    lengths = [ab, bc, ca]
    sorted_lengths = sorted(lengths)
    side_a, side_b, side_c = sorted_lengths  # side_c is the hypotenuse candidate.
    c_from_legs = math.sqrt(side_a ** 2 + side_b ** 2)
    other_leg_from_hyp = math.sqrt(max(side_c ** 2 - side_a ** 2, 0.0))
    right_angle_match = abs(side_c ** 2 - (side_a ** 2 + side_b ** 2)) < 1e-3 * (side_c ** 2 + side_a ** 2 + side_b ** 2)

    return {
        "lengths": {
            "AB": ab,
            "BC": bc,
            "CA": ca,
        },
        "sorted_lengths": {
            "a": side_a,
            "b": side_b,
            "c": side_c,
        },
        "c_from_legs": c_from_legs,
        "other_leg_from_hyp": other_leg_from_hyp,
        "right_angle_match": right_angle_match,
    }


def overlay_text(frame: np.ndarray, text: str, y: int) -> None:
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def capture_numeric_input(window_name: str, prompt: str) -> Optional[float]:
    """
    Capture a positive float number from the keyboard using cv2.waitKey.
    Shows the typed characters on the screen. Finish with Enter. ESC cancels.
    """
    buffer = ""
    input_window = f"{window_name}_input"
    while True:
        dummy = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(dummy, prompt, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dummy, buffer, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(input_window, dummy)
        key = cv2.waitKey(50) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(input_window)
            return None
        if key in (13, 10):  # Enter
            cv2.destroyWindow(input_window)
            if buffer.strip() == "":
                return None
            try:
                return float(buffer)
            except ValueError:
                return None
        if key == 8:  # backspace
            buffer = buffer[:-1]
        elif 48 <= key <= 57 or key == ord('.'):
            buffer += chr(key)


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------


def main() -> None:
    window_name = "Abhay's project"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(window_name, 1280, 720)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    last_draw_point: Optional[Tuple[int, int]] = None
    smoothed_point: Optional[Tuple[int, int]] = None
    # We follow the "explicit vertex" approach described in the prompt: the user
    # hovers the fingertip over each triangle corner and presses keys 1, 2, and 3
    # to capture the coordinates. This keeps the implementation simple and
    # avoids heuristics for automatic vertex detection.
    vertex_points: Dict[int, Optional[Tuple[int, int]]] = {1: None, 2: None, 3: None}
    current_gesture = "idle"
    gesture_counter = 0
    calculation_triggered = False
    metrics = None
    # User-entered side lengths (in arbitrary units, e.g., cm)
    user_sides = {"a": None, "b": None, "c": None}

    print("Instructions: two fingers close = draw, single finger = pause, five fingers = calculate.")
    print("Place the fingertip at each vertex and press keys 1, 2, and 3 to register the triangle vertices.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
            frame = cv2.resize(frame, (frame_w, frame_h))

        hand_landmarks, handedness_label = detect_hand(frame, hands)
        extended = get_extended_fingers(hand_landmarks, handedness_label)
        candidate_gesture = determine_candidate_gesture(extended)
        if candidate_gesture == current_gesture:
            gesture_counter = 0
        else:
            gesture_counter += 1
            if gesture_counter >= 5:
                current_gesture = candidate_gesture
                gesture_counter = 0

        pixel_landmarks = landmarks_to_pixels(hand_landmarks, frame_w, frame_h)
        index_tip = pixel_landmarks.get(8)

        if current_gesture == "draw" and index_tip:
            ix, iy = index_tip
            current_point = (ix, iy)
            if smoothed_point is None:
                smoothed_point = current_point
            else:
                alpha = 0.3
                sx = int(alpha * current_point[0] + (1 - alpha) * smoothed_point[0])
                sy = int(alpha * current_point[1] + (1 - alpha) * smoothed_point[1])
                smoothed_point = (sx, sy)

            if last_draw_point is None:
                last_draw_point = smoothed_point
            else:
                cv2.line(canvas, last_draw_point, smoothed_point, (0, 255, 255), 4)
                last_draw_point = smoothed_point
        else:
            last_draw_point = None
            smoothed_point = None

        # Edge-triggered calculation mode.
        if current_gesture == "calculate" and not calculation_triggered:
            calculation_triggered = True
            metrics = compute_triangle_metrics([vertex_points[1], vertex_points[2], vertex_points[3]])
        elif current_gesture != "calculate":
            calculation_triggered = False

        display_canvas = canvas.copy()
        if all(vertex_points.values()):
            pts = [vertex_points[1], vertex_points[2], vertex_points[3]]
            cv2.line(display_canvas, pts[0], pts[1], (0, 255, 0), 3)
            cv2.line(display_canvas, pts[1], pts[2], (0, 255, 0), 3)
            cv2.line(display_canvas, pts[2], pts[0], (0, 255, 0), 3)

            ab = euclidean_distance(pts[0], pts[1])
            bc = euclidean_distance(pts[1], pts[2])
            ca = euclidean_distance(pts[2], pts[0])

            def mid(p, q):
                return int((p[0] + q[0]) / 2), int((p[1] + q[1]) / 2)

            cv2.putText(display_canvas, f"{ab:.1f}px", mid(pts[0], pts[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_canvas, f"{bc:.1f}px", mid(pts[1], pts[2]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_canvas, f"{ca:.1f}px", mid(pts[2], pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = cv2.addWeighted(frame, 1.0, display_canvas, 1.0, 0)

        # Draw vertex points and lines connecting them for clarity.
        for idx, point in vertex_points.items():
            if point:
                cv2.circle(combined, point, 10, (0, 165, 255), -1)
                cv2.putText(combined, f"V{idx}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        overlay_text(combined, f"Mode: {current_gesture}", 30)
        overlay_text(combined, f"Candidate gesture: {candidate_gesture}", 60)
        overlay_text(combined, "Two fingers together = draw", 95)
        overlay_text(combined, "Single finger up = pause drawing", 125)
        overlay_text(combined, "All/most fingers open = calculate third side", 155)
        overlay_text(combined, "Press 1/2/3 to save the triangle vertices at the fingertip", 185)
        overlay_text(combined, "Press R to reset canvas", 215)
        overlay_text(combined, "After triangle is set & metrics show:", 245)
        overlay_text(combined, "Press A to enter length of side a", 270)
        overlay_text(combined, "Press B to enter length of side b", 295)
        overlay_text(combined, "Press C to enter length of side c (hypotenuse)", 320)

        if metrics:
            lengths = metrics["lengths"]
            sorted_lengths = metrics["sorted_lengths"]
            overlay_text(combined, f"AB: {lengths['AB']:.1f} px", 350)
            overlay_text(combined, f"BC: {lengths['BC']:.1f} px", 375)
            overlay_text(combined, f"CA: {lengths['CA']:.1f} px", 400)
            overlay_text(combined, f"Hypotenuse candidate (c): {sorted_lengths['c']:.1f} px", 425)

            ua = user_sides["a"]
            ub = user_sides["b"]
            uc = user_sides["c"]
            overlay_text(combined, f"user a: {ua if ua is not None else '-'}", 455)
            overlay_text(combined, f"user b: {ub if ub is not None else '-'}", 480)
            overlay_text(combined, f"user c: {uc if uc is not None else '-'}", 505)

            calculated_side_text = "Waiting for at least two user sides..."
            if ua is not None and ub is not None and uc is None:
                c_val = math.sqrt(ua ** 2 + ub ** 2)
                user_sides["c"] = c_val
                uc = c_val
                calculated_side_text = f"Computed c from a & b: c = sqrt(a^2+b^2) = {c_val:.2f}"
            elif ua is not None and uc is not None and ub is None:
                if uc ** 2 > ua ** 2:
                    b_val = math.sqrt(uc ** 2 - ua ** 2)
                    user_sides["b"] = b_val
                    ub = b_val
                    calculated_side_text = f"Computed b from c & a: b = sqrt(c^2-a^2) = {b_val:.2f}"
                else:
                    calculated_side_text = "Invalid: c must be largest side (c^2 > a^2)."
            elif ub is not None and uc is not None and ua is None:
                if uc ** 2 > ub ** 2:
                    a_val = math.sqrt(uc ** 2 - ub ** 2)
                    user_sides["a"] = a_val
                    ua = a_val
                    calculated_side_text = f"Computed a from c & b: a = sqrt(c^2-b^2) = {a_val:.2f}"
                else:
                    calculated_side_text = "Invalid: c must be largest side (c^2 > b^2)."

            overlay_text(combined, calculated_side_text, 535)
            match_text = "Right angle confirmed" if metrics['right_angle_match'] else "Triangle deviates from right angle"
            overlay_text(combined, match_text, 565)
        else:
            overlay_text(combined, "Waiting for all three vertices before calculating...", 350)

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key in (ord('1'), ord('2'), ord('3')) and index_tip:
            vertex_points[int(chr(key))] = index_tip
        if key in (ord('a'), ord('A')):
            if metrics is not None:
                val = capture_numeric_input(window_name, "Enter length for side a:")
                if val is not None and val > 0:
                    user_sides["a"] = val
        if key in (ord('b'), ord('B')):
            if metrics is not None:
                val = capture_numeric_input(window_name, "Enter length for side b:")
                if val is not None and val > 0:
                    user_sides["b"] = val
        if key in (ord('c'), ord('C')):
            if metrics is not None:
                val = capture_numeric_input(window_name, "Enter length for side c (hypotenuse):")
                if val is not None and val > 0:
                    user_sides["c"] = val
        if key in (ord('r'), ord('R')):
            canvas = np.zeros_like(frame)
            vertex_points = {1: None, 2: None, 3: None}
            metrics = None
            last_draw_point = None
            smoothed_point = None
            user_sides = {"a": None, "b": None, "c": None}

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
