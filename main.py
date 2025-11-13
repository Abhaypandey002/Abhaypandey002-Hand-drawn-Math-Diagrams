import math
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Return the Euclidean distance between two (x, y) points."""
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def extract_hand_landmarks(frame: np.ndarray, hands) -> Optional[Dict[str, Tuple[int, int]]]:
    """Run MediaPipe Hands and return the landmark dictionary in pixel coordinates."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return None

    h, w, _ = frame.shape
    # Use the first detected hand for simplicity.
    hand_landmarks = result.multi_hand_landmarks[0]
    landmark_dict: Dict[str, Tuple[int, int]] = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        px, py = int(lm.x * w), int(lm.y * h)
        landmark_dict[idx] = (px, py)
    return landmark_dict


def fingers_extended(landmarks: Dict[str, Tuple[int, int]]) -> List[bool]:
    """Return which fingers are extended based on landmark positions."""
    if landmarks is None:
        return [False] * 5

    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]
    states: List[bool] = []

    # Thumb: compare x coordinate due to lateral movement.
    thumb_tip = landmarks.get(finger_tips[0])
    thumb_ip = landmarks.get(finger_pips[0])
    if thumb_tip and thumb_ip:
        states.append(thumb_tip[0] > thumb_ip[0])
    else:
        states.append(False)

    # Other fingers: compare y coordinate (tip lower value => extended).
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        tip_pt = landmarks.get(tip)
        pip_pt = landmarks.get(pip)
        if tip_pt and pip_pt:
            states.append(tip_pt[1] < pip_pt[1])
        else:
            states.append(False)
    return states


def determine_candidate_gesture(landmarks: Optional[Dict[str, Tuple[int, int]]],
                                finger_state: List[bool]) -> str:
    """Return the gesture keyword based on finger states and fingertip distance."""
    if not landmarks:
        return "idle"

    index_tip = landmarks.get(8)
    middle_tip = landmarks.get(12)
    thumb, index, middle, ring, pinky = finger_state

    if all(finger_state):
        return "calculate"

    if index and not middle and not ring and not pinky:
        return "pause"

    if index and middle and index_tip and middle_tip:
        if euclidean_distance(index_tip, middle_tip) < 40:
            return "draw"

    return "idle"


def update_mode(current_mode: str, candidate: str, counter: Dict[str, int], threshold: int = 5) -> str:
    """Stabilize gesture detection by requiring the same gesture for several frames."""
    if candidate == current_mode:
        counter[candidate] = counter.get(candidate, 0) + 1
        return current_mode

    counter[candidate] = counter.get(candidate, 0) + 1
    counter[current_mode] = 0
    if counter[candidate] >= threshold:
        return candidate
    return current_mode


def compute_triangle_metrics(points: List[Optional[Tuple[int, int]]]) -> Optional[Dict[str, any]]:
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


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------


def main() -> None:
    cv2.namedWindow("Abhay's project")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    canvas = None
    prev_point = None
    # We follow the "explicit vertex" approach described in the prompt: the user
    # hovers the fingertip over each triangle corner and presses keys 1, 2, and 3
    # to capture the coordinates. This keeps the implementation simple and
    # avoids heuristics for automatic vertex detection.
    vertex_points: Dict[int, Optional[Tuple[int, int]]] = {1: None, 2: None, 3: None}
    mode_counter: Dict[str, int] = {}
    active_mode = "idle"
    calculation_triggered = False
    metrics = None

    print("Instructions: two fingers close = draw, single finger = pause, five fingers = calculate.")
    print("Place the fingertip at each vertex and press keys 1, 2, and 3 to register the triangle vertices.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)

        landmarks = extract_hand_landmarks(frame, hands)
        finger_state = fingers_extended(landmarks) if landmarks else [False] * 5
        candidate_mode = determine_candidate_gesture(landmarks, finger_state)
        active_mode = update_mode(active_mode, candidate_mode, mode_counter)

        index_tip = landmarks.get(8) if landmarks else None

        if active_mode == "draw" and index_tip:
            if prev_point is None:
                prev_point = index_tip
            cv2.line(canvas, prev_point, index_tip, (0, 255, 255), 4)
            prev_point = index_tip
        else:
            prev_point = None

        # Edge-triggered calculation mode.
        if active_mode == "calculate" and not calculation_triggered:
            calculation_triggered = True
            metrics = compute_triangle_metrics([vertex_points[1], vertex_points[2], vertex_points[3]])
        elif active_mode != "calculate":
            calculation_triggered = False

        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

        # Draw vertex points and lines connecting them for clarity.
        for idx, point in vertex_points.items():
            if point:
                cv2.circle(combined, point, 10, (0, 165, 255), -1)
                cv2.putText(combined, f"V{idx}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if all(vertex_points.values()):
            pts = [vertex_points[1], vertex_points[2], vertex_points[3]]
            cv2.line(combined, pts[0], pts[1], (0, 255, 0), 2)
            cv2.line(combined, pts[1], pts[2], (0, 255, 0), 2)
            cv2.line(combined, pts[2], pts[0], (0, 255, 0), 2)

        mode_text = {
            "draw": "Mode: Draw (two fingers together)",
            "pause": "Mode: Pause (single finger)",
            "calculate": "Mode: Calculate (all fingers open)",
            "idle": "Mode: Idle / waiting for gesture",
        }.get(active_mode, "Mode: Idle")

        overlay_text(combined, mode_text, 30)
        overlay_text(combined, "Two fingers together = draw", 60)
        overlay_text(combined, "Single finger up = pause drawing", 90)
        overlay_text(combined, "All fingers open = calculate third side", 120)
        overlay_text(combined, "Press 1/2/3 to save the triangle vertices at the fingertip", 150)
        overlay_text(combined, "Press R to reset canvas", 180)

        if metrics:
            lengths = metrics["lengths"]
            sorted_lengths = metrics["sorted_lengths"]
            overlay_text(combined, f"AB: {lengths['AB']:.1f} px", 210)
            overlay_text(combined, f"BC: {lengths['BC']:.1f} px", 240)
            overlay_text(combined, f"CA: {lengths['CA']:.1f} px", 270)
            overlay_text(combined, f"Hypotenuse candidate (c): {sorted_lengths['c']:.1f} px", 300)
            overlay_text(combined, f"c^2 = a^2 + b^2 -> c = {metrics['c_from_legs']:.1f} px", 330)
            overlay_text(combined, f"Given c & a -> b = sqrt(c^2 - a^2) = {metrics['other_leg_from_hyp']:.1f} px", 360)
            match_text = "Right angle confirmed" if metrics['right_angle_match'] else "Triangle deviates from right angle"
            overlay_text(combined, match_text, 390)
        else:
            overlay_text(combined, "Waiting for all three vertices before calculating...", 210)

        cv2.imshow("Abhay's project", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key in (ord('1'), ord('2'), ord('3')) and index_tip:
            vertex_points[int(chr(key))] = index_tip
        if key in (ord('r'), ord('R')):
            canvas = np.zeros_like(frame)
            vertex_points = {1: None, 2: None, 3: None}
            metrics = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
