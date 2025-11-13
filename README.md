# Abhay's project – Hand Gesture Right-Angle Triangle Drawer

This beginner-friendly project combines Python, OpenCV, MediaPipe, and simple hand gestures to let you draw a right-angled triangle in the air and compute its third side using the Pythagorean theorem. The webcam feed is overlaid with a virtual canvas where your index fingertip becomes a pen.

## Prerequisites

- Python 3.9+
- A webcam connected to your computer
- CPU-only execution (no GPU required)

### Required packages

```bash
pip install opencv-python mediapipe numpy
```

## How to run

1. Clone or download this repository.
2. Install the dependencies listed above.
3. Run the application:
   ```bash
   python main.py
   ```

## Usage

- **Two fingers together (index + middle touching):** drawing mode. Your index fingertip acts like a pen on the canvas.
- **Single index finger up:** drawing pauses so you can reposition without drawing extra lines.
- **All five fingers open:** trigger the triangle calculation. The system measures two sides and computes the third using the Pythagorean theorem.
- **Marking vertices:** place your fingertip over each triangle corner and press keys `1`, `2`, and `3` respectively. This explicit marking keeps the geometry clean and easy to measure.
- **Enter real side lengths:** once the metrics block appears on screen, press `A`, `B`, or `C` to type the measured value for sides `AB`, `BC`, or `CA` (in any units you choose). The program maps those numbers to the actual sides you drew and uses only those values to compute the remaining side.
- **Reset:** press `R` to clear the canvas, stored vertices, and any previously entered side lengths.
- **Quit:** press `Q` to close the application window.

Once the three vertices are set and you open all fingers, the UI highlights the triangle, labels each side, and shows:

- Lengths of sides AB, BC, and CA.
- Which side is treated as the hypotenuse.
- Which user-entered sides are already known (e.g., real-world units you typed via `A`, `B`, `C`).
- The classic `c^2 = a^2 + b^2` / `b = sqrt(c^2 - a^2)` math, performed strictly on the user-entered numbers so that pixel distances are only used for visualization.

## Troubleshooting

- **Webcam not detected:** ensure another application is not using the camera. You can also try changing the index in `cv2.VideoCapture(0)` to `1` or `2` if you have multiple cameras.
- **Low performance:** reduce the capture resolution by lowering the `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT` values near the top of `main.py`.
- **Hand not detected:** make sure your hand is well lit and fully visible. Keep your palm facing the camera for the most reliable landmark tracking.

## Code overview

- `main.py` contains the full application:
  - Hand detection and landmark extraction using MediaPipe.
  - Gesture recognition logic for drawing, pausing, and calculating.
  - Canvas overlay for rendering the virtual pen strokes.
  - Vertex collection via keyboard shortcuts and triangle measurement utilities.
  - Pythagorean computations that determine the hypotenuse or unknown leg by pairing your typed values with the correct side labels (AB, BC, CA) rather than relying on raw pixel distances.

Feel free to explore and tweak the thresholds or UI text to learn more about computer vision and interactive math visualizations!
