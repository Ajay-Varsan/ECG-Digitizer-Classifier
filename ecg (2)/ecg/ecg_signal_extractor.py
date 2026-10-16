"""
ecg_signal_extractor.py
-----------------------
Extracts 1D continuous digital time-series waveforms from 2D ECG lead images
using color filtering, vertical column skeletonization, and signal processing.
Calibrates horizontal time (seconds) and vertical amplitude (mV) based on
standard ECG paper specifications (25 mm/s, 10 mm/mV).
"""

import cv2
import numpy as np
from scipy import signal, ndimage

# Standard ECG paper calibration constants:
# Standard speed: 25 mm/s -> 1 second = 25 mm.
# Standard 1 mm small box = 0.04s (40 ms). 1 large box (5 mm) = 0.20s (200 ms).
# Standard sensitivity: 10 mm/mV -> 1 mV = 10 mm.


def extract_lead_signal(lead_image, sampling_rate_hz=250):
    """
    Extracts 1D continuous waveform y(t) from an ECG lead crop image.

    Parameters:
        lead_image: BGR numpy image of the lead region.
        sampling_rate_hz: Target effective sampling rate in Hz.

    Returns:
        dict containing:
            - 'signal_raw': 1D numpy array of raw pixel heights
            - 'signal_filtered': 1D numpy array of filtered waveform in normalized units
            - 'signal_mv': 1D numpy array calibrated approximately in millivolts
            - 'time_sec': 1D numpy array of timestamps (seconds)
            - 'trace_points': List of (x, y) coordinates on the original lead_image
            - 'baseline_y': Estimated isoelectric baseline Y pixel coordinate
            - 'px_per_sec': Estimated horizontal pixel density per second
            - 'px_per_mv': Estimated vertical pixel density per mV
            - 'lead_image': Cleaned/debug mask
    """
    if lead_image is None or lead_image.size == 0:
        return None

    h, w = lead_image.shape[:2]
    if w < 10 or h < 10:
        return None

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(lead_image, cv2.COLOR_BGR2GRAY)

    # 2. Extract dark ECG waveform trace
    # In printed/scanned clinical ECGs:
    # Grid lines are pink/red/light-gray (grayscale typically > 140).
    # The actual ECG trace is dark black/blue (grayscale typically < 115).
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    trace_thresh = min(thresh_val, 115)
    trace_mask = (gray < trace_thresh).astype(np.uint8) * 255

    # Remove isolated 1-pixel noise specks
    kernel = np.ones((2, 2), np.uint8)
    trace_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_OPEN, kernel)

    # If trace_mask is too sparse (very faint scan), fallback to adaptive
    if np.sum(trace_mask > 0) < w:
        trace_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10
        )

    # 3. Extract 1D Signal via Column-wise Center of Mass
    y_raw = np.full(w, np.nan, dtype=np.float32)
    top_margin = max(2, int(h * 0.03))
    bot_margin = max(2, int(h * 0.03))

    for x in range(w):
        col = trace_mask[top_margin:h - bot_margin, x]
        dark_y = np.where(col > 0)[0] + top_margin

        if len(dark_y) == 0:
            # Check darkest pixel in grayscale column
            col_gray = gray[top_margin:h - bot_margin, x]
            min_y = int(np.argmin(col_gray)) + top_margin
            if col_gray[min_y - top_margin] < 125:
                y_raw[x] = float(min_y)
            continue

        # If column is mostly dark (>45% of height), it is a vertical divider line -> skip column
        if len(dark_y) > 0.45 * h:
            continue

        weights = 255.0 - gray[dark_y, x].astype(np.float32)
        weights = np.maximum(weights, 1.0)
        y_val = np.sum(dark_y * weights) / np.sum(weights)
        y_raw[x] = float(y_val)

    # 4. Interpolate Missing / Skipped Columns
    import pandas as pd
    s = pd.Series(y_raw)
    y_interp = s.interpolate(method="linear").bfill().ffill().fillna(h / 2.0).values

    # 5. Baseline estimation via histogram mode of resting isoelectric segment
    # In clinical ECG, the trace rests on the isoelectric baseline between T and P waves
    hist, bin_edges = np.histogram(y_interp, bins=60)
    baseline_y = float(bin_edges[np.argmax(hist)])

    # Invert Y so upward deflection is positive voltage
    raw_centered = -(y_interp - baseline_y)

    # 6. Gentle Baseline Wander Removal (Preserving QRS, P, and T waveforms)
    # Only remove ultra-low frequency drift (> 1.2s period)
    win = int(w * 0.45)
    if win % 2 == 0:
        win += 1
    if len(raw_centered) > win:
        slow_drift = signal.medfilt(raw_centered, kernel_size=win)
        slow_drift = ndimage.gaussian_filter1d(slow_drift, sigma=max(5, int(w * 0.05)))
        filtered = raw_centered - slow_drift
    else:
        filtered = raw_centered.copy()

    # 7. Calibration to Time (seconds) and Voltage (mV)
    is_rhythm_strip = (w / max(1, h)) > 5.0
    duration_sec = 10.0 if is_rhythm_strip else 2.5
    px_per_sec = w / max(0.1, duration_sec)
    time_sec = np.linspace(0, duration_sec, w)

    # Standard clinical ECG: 10 mm = 1.0 mV. Standard grid box is ~1/3 to 1/4 of lead height
    # Peak-to-peak calibration
    amp_span = np.percentile(filtered, 99) - np.percentile(filtered, 1)
    px_per_mv = max(25.0, h * 0.30)
    signal_mv = filtered / px_per_mv

    # Reconstruct original 2D (x, y) trace points
    trace_points = []
    for x in range(w):
        orig_y = int(np.clip(baseline_y - filtered[x], 0, h - 1))
        trace_points.append((x, orig_y))

    return {
        "signal_raw": y_interp,
        "signal_filtered": filtered,
        "signal_mv": signal_mv,
        "time_sec": time_sec,
        "duration_sec": duration_sec,
        "trace_points": trace_points,
        "baseline_y": baseline_y,
        "px_per_sec": px_per_sec,
        "px_per_mv": px_per_mv,
        "width": w,
        "height": h,
    }


def extract_all_leads_signals(image, lead_boxes):
    """
    Extracts continuous 1D signals from all detected lead boxes and rhythm strip.

    Returns:
        dict mapping lead_name -> extracted signal dictionary
    """
    leads_data = {}
    for label, x1, y1, x2, y2 in lead_boxes:
        crop = image[y1:y2, x1:x2]
        sig_data = extract_lead_signal(crop)
        if sig_data is not None:
            sig_data["bbox"] = (x1, y1, x2, y2)
            sig_data["label"] = label
            leads_data[label] = sig_data

    return leads_data
