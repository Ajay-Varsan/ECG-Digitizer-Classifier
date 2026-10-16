"""
ecg_utils.py
------------
Small shared helper functions used by BOTH train.py and predict.py,
so that the exact same preprocessing / cropping / lead-box logic is
used everywhere. Kept in one file to avoid duplicating code.

No machine learning happens here - just plain OpenCV image handling.
"""

import cv2
import numpy as np

IMG_SIZE = 224  # MobileNetV2's standard input size


# ---------------------------------------------------------------------
# 1. Detect and crop the main ECG waveform region
# ---------------------------------------------------------------------
def detect_ecg_region(image):
    """
    Very simple ECG-region detector (no training required).

    Most printed ECG reports have a pink/red grid in the waveform
    area, while the header/footer (patient name, hospital info, etc.)
    sits on plain white background. We use that color difference to
    find the waveform area:

        1. Convert the image to HSV.
        2. Threshold for the pink/red grid color.
        3. Take the bounding box of all matching pixels.
        4. If nothing is found (e.g. a black & white scan), fall back
           to using the whole image.

    Returns (x, y, w, h) of the detected region, in pixel coordinates.
    """
    h_img, w_img = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Pink / red ECG grid color range. This covers most printed ECG
    # paper colors without being so wide it grabs unrelated content.
    lower_red1 = np.array([0, 15, 120])
    upper_red1 = np.array([10, 120, 255])
    lower_red2 = np.array([160, 15, 120])
    upper_red2 = np.array([179, 120, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Close small gaps so the grid forms one solid blob instead of
    # thousands of tiny disconnected specks.
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(mask)

    if coords is None or len(coords) < 500:
        # Could not find a clear grid -> just use the whole image.
        return 0, 0, w_img, h_img

    x, y, w, h = cv2.boundingRect(coords)

    # Small margin so we don't clip waveform edges.
    margin = int(0.02 * max(w, h))
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(w_img - x, w + 2 * margin)
    h = min(h_img - y, h + 2 * margin)

    return x, y, w, h


# ---------------------------------------------------------------------
# 2. Build approximate bounding boxes for the 12 leads + rhythm strip
# ---------------------------------------------------------------------
def get_lead_boxes(x, y, w, h):
    """
    Build approximate bounding boxes for the 12 ECG leads plus the
    rhythm strip, based on the STANDARD 3x4 layout of a 12-lead ECG:

        I    aVR   V1   V4
        II   aVL   V2   V5
        III  aVF   V3   V6
        ------- Lead II rhythm strip -------

    These are NOT pixel-perfect / medically validated boxes - they are
    simple proportional boxes based on the detected ECG region (x, y,
    w, h), so they automatically scale to any image resolution.

    Returns a list of (label, x1, y1, x2, y2) tuples in pixel coords.
    """
    labels = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    n_rows = 3
    n_cols = 4

    # Reserve the bottom ~22% of the region for the rhythm strip.
    rhythm_h = int(h * 0.22)
    leads_h = h - rhythm_h

    row_h = leads_h // n_rows
    col_w = w // n_cols

    boxes = []
    for r in range(n_rows):
        for c in range(n_cols):
            x1 = x + c * col_w
            y1 = y + r * row_h
            x2 = x1 + col_w
            y2 = y1 + row_h
            boxes.append((labels[r][c], x1, y1, x2, y2))

    # Rhythm strip spans the full width at the bottom.
    boxes.append((
        "Lead II rhythm strip",
        x, y + leads_h, x + w, y + h,
    ))

    return boxes


def draw_lead_boxes(image, boxes):
    """Draw the lead bounding boxes + labels on a COPY of the image."""
    annotated = image.copy()
    for label, x1, y1, x2, y2 in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, label, (x1 + 4, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
        )
    return annotated


# ---------------------------------------------------------------------
# 3. Load + crop + resize an image for the CNN
# ---------------------------------------------------------------------
def load_and_crop(path, img_size=IMG_SIZE):
    """
    Read an image from disk, detect + crop the main ECG region, and
    resize it to (img_size, img_size). Returns a BGR uint8 array.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    x, y, w, h = detect_ecg_region(image)
    cropped = image[y:y + h, x:x + w]

    if cropped.size == 0:
        cropped = image  # safety fallback

    resized = cv2.resize(cropped, (img_size, img_size))
    return resized
