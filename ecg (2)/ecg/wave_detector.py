"""
wave_detector.py
----------------
High-precision detection of individual ECG waveform crests and troughs:
- P-peaks (Crests) & Inverted P (Troughs)
- Q-troughs
- R-peaks (Main Crests)
- S-troughs
- T-peaks (Crests) & Inverted T (Troughs)
- Flutter sawtooth crests & Fibrillatory micro-crests/troughs
- Anomalous complexes (Wide QRS / PVC, Prolonged PR, Pauses, ST shifts)

Computes bounding boxes for every single crest/trough and anomalous segment,
both in local lead coordinates and global image coordinates.
"""

import numpy as np
from scipy import signal
from disorder_palette import WAVE_FIDUCIAL_COLORS, DISORDER_COLORS, get_disorder_color


def detect_fiducials_and_anomalies(lead_sig_data, lead_label="Lead", global_box_offset=(0, 0)):
    """
    Analyzes an extracted 1D lead signal, finds all crests (peaks) and troughs (valleys),
    delineates P-Q-R-S-T waves, flags anomalies, and computes bounding boxes.

    Parameters:
        lead_sig_data: Output dict from ecg_signal_extractor.extract_lead_signal
        lead_label: Name of lead (e.g. "II", "V1", "Lead II rhythm strip")
        global_box_offset: (x_offset, y_offset) of the lead in the full ECG image

    Returns:
        dict containing:
            - 'crests': List of crest bounding boxes and metadata
            - 'troughs': List of trough bounding boxes and metadata
            - 'anomalies': List of anomaly bounding boxes and metadata
            - 'all_boxes': Combined list of all bounding boxes with colors and labels
            - 'intervals': Computed cardiac metrics (HR, RR_cv, PR_mean, QRS_mean, etc.)
            - 'diagnostics': Detailed waveform summary
    """
    if lead_sig_data is None:
        return None

    sig = lead_sig_data["signal_filtered"]
    sig_mv = lead_sig_data["signal_mv"]
    time_sec = lead_sig_data["time_sec"]
    baseline_y = lead_sig_data["baseline_y"]
    px_per_sec = lead_sig_data["px_per_sec"]
    px_per_mv = lead_sig_data["px_per_mv"]
    w = lead_sig_data["width"]
    h = lead_sig_data["height"]
    x_off, y_off = global_box_offset

    n_samples = len(sig)
    if n_samples < 20:
        return None

    # Normalization & dynamic thresholding
    std_val = np.std(sig)
    if std_val < 1e-5:
        std_val = 1.0

    # -------------------------------------------------------------------------
    # 1. Detect R-Peaks (Dominant Crests)
    # -------------------------------------------------------------------------
    # In standard ECG, R-peaks are sharp, high-prominence positive deflections.
    # We set minimum distance corresponding to max HR ~ 250 bpm (0.24s)
    min_r_dist = max(5, int(0.24 * px_per_sec))
    r_prominence = max(0.25 * np.max(np.abs(sig)), 1.2 * std_val)

    r_peaks, r_props = signal.find_peaks(
        sig,
        distance=min_r_dist,
        prominence=r_prominence,
        height=0.1 * np.max(sig) if np.max(sig) > 0 else None
    )

    # If inverted lead (like aVR), R peaks might be inverted (deep S-wave dominance)
    is_inverted_lead = "aVR" in lead_label or (len(r_peaks) == 0 and np.min(sig) < -2.5 * std_val)
    if len(r_peaks) == 0:
        # Try inverted peak search
        inv_peaks, _ = signal.find_peaks(-sig, distance=min_r_dist, prominence=r_prominence)
        if len(inv_peaks) > 0:
            r_peaks = inv_peaks

    # -------------------------------------------------------------------------
    # 2. Delineate Q-Troughs, S-Troughs, P-Crests, and T-Crests around each R-Peak
    # -------------------------------------------------------------------------
    crests = []
    troughs = []
    anomalies = []
    all_boxes = []

    # Window sizes in pixels
    win_q = max(2, int(0.06 * px_per_sec))    # ~60 ms before R
    win_s = max(2, int(0.08 * px_per_sec))    # ~80 ms after R
    win_p = max(5, int(0.22 * px_per_sec))    # ~220 ms before R
    win_t = max(5, int(0.40 * px_per_sec))    # ~400 ms after R

    complexes = []
    r_times = []
    rr_intervals_ms = []
    pr_intervals_ms = []
    qrs_durations_ms = []

    for i, r_idx in enumerate(r_peaks):
        r_time = time_sec[r_idx]
        r_times.append(r_time)
        r_amp_mv = sig_mv[r_idx]

        # Global pixel position of R-crest
        r_local_y = int(np.clip(baseline_y - sig[r_idx], 2, h - 2))
        r_box_w = max(6, int(0.04 * px_per_sec))
        r_box_h = max(8, int(abs(sig[r_idx]) * 0.4) + 6)

        r_box = {
            "type": "R_PEAK",
            "name": f"R-Crest #{i+1}",
            "kind": "crest",
            "index": int(r_idx),
            "time_sec": float(r_time),
            "amplitude_mv": float(r_amp_mv),
            "local_box": [
                max(0, r_idx - r_box_w // 2),
                max(0, r_local_y - r_box_h // 2),
                min(w, r_idx + r_box_w // 2),
                min(h, r_local_y + r_box_h // 2),
            ],
            "global_box": [
                max(0, x_off + r_idx - r_box_w // 2),
                max(0, y_off + r_local_y - r_box_h // 2),
                min(x_off + w, x_off + r_idx + r_box_w // 2),
                min(y_off + h, y_off + r_local_y + r_box_h // 2),
            ],
            "color_bgr": WAVE_FIDUCIAL_COLORS["R_PEAK"]["bgr"],
            "color_hex": WAVE_FIDUCIAL_COLORS["R_PEAK"]["hex"],
        }
        crests.append(r_box)
        all_boxes.append(r_box)

        # --- Q-Trough (local minimum immediately preceding R) ---
        q_start = max(0, r_idx - win_q)
        q_idx = q_start + np.argmin(sig[q_start:r_idx]) if r_idx > q_start else r_idx
        q_amp_mv = sig_mv[q_idx]
        q_local_y = int(np.clip(baseline_y - sig[q_idx], 2, h - 2))
        q_box_w = max(5, int(0.03 * px_per_sec))
        q_box_h = max(6, int(abs(sig[q_idx]) * 0.3) + 5)

        q_box = {
            "type": "Q_TROUGH",
            "name": f"Q-Trough #{i+1}",
            "kind": "trough",
            "index": int(q_idx),
            "time_sec": float(time_sec[q_idx]),
            "amplitude_mv": float(q_amp_mv),
            "local_box": [
                max(0, q_idx - q_box_w // 2),
                max(0, q_local_y - q_box_h // 2),
                min(w, q_idx + q_box_w // 2),
                min(h, q_local_y + q_box_h // 2),
            ],
            "global_box": [
                max(0, x_off + q_idx - q_box_w // 2),
                max(0, y_off + q_local_y - q_box_h // 2),
                min(x_off + w, x_off + q_idx + q_box_w // 2),
                min(y_off + h, y_off + q_local_y + q_box_h // 2),
            ],
            "color_bgr": WAVE_FIDUCIAL_COLORS["Q_TROUGH"]["bgr"],
            "color_hex": WAVE_FIDUCIAL_COLORS["Q_TROUGH"]["hex"],
        }
        troughs.append(q_box)
        all_boxes.append(q_box)

        # --- S-Trough (local minimum immediately following R) ---
        s_end = min(n_samples, r_idx + win_s)
        s_idx = r_idx + np.argmin(sig[r_idx:s_end]) if s_end > r_idx else r_idx
        s_amp_mv = sig_mv[s_idx]
        s_local_y = int(np.clip(baseline_y - sig[s_idx], 2, h - 2))
        s_box_w = max(5, int(0.03 * px_per_sec))
        s_box_h = max(6, int(abs(sig[s_idx]) * 0.3) + 5)

        s_box = {
            "type": "S_TROUGH",
            "name": f"S-Trough #{i+1}",
            "kind": "trough",
            "index": int(s_idx),
            "time_sec": float(time_sec[s_idx]),
            "amplitude_mv": float(s_amp_mv),
            "local_box": [
                max(0, s_idx - s_box_w // 2),
                max(0, s_local_y - s_box_h // 2),
                min(w, s_idx + s_box_w // 2),
                min(h, s_local_y + s_box_h // 2),
            ],
            "global_box": [
                max(0, x_off + s_idx - s_box_w // 2),
                max(0, y_off + s_local_y - s_box_h // 2),
                min(x_off + w, x_off + s_idx + s_box_w // 2),
                min(y_off + h, y_off + s_local_y + s_box_h // 2),
            ],
            "color_bgr": WAVE_FIDUCIAL_COLORS["S_TROUGH"]["bgr"],
            "color_hex": WAVE_FIDUCIAL_COLORS["S_TROUGH"]["hex"],
        }
        troughs.append(s_box)
        all_boxes.append(s_box)

        # QRS Duration (ms)
        qrs_dur_ms = (time_sec[s_idx] - time_sec[q_idx]) * 1000.0
        # If Q or S wasn't distinct, approximate using slope base
        if qrs_dur_ms < 40:
            qrs_dur_ms = 80.0
        qrs_durations_ms.append(qrs_dur_ms)

        # --- P-Wave (Crest or Inverted Trough in the pre-QRS zone) ---
        p_search_start = max(0, r_idx - win_p)
        p_search_end = max(0, q_idx - int(0.02 * px_per_sec))
        p_idx = None
        p_is_inverted = False

        if p_search_end > p_search_start + 2:
            p_segment = sig[p_search_start:p_search_end]
            max_p_amp = np.max(p_segment)
            min_p_amp = np.min(p_segment)

            if abs(max_p_amp) >= abs(min_p_amp) and max_p_amp > 0.05 * std_val:
                p_idx = p_search_start + np.argmax(p_segment)
                p_is_inverted = False
            elif abs(min_p_amp) > abs(max_p_amp) and min_p_amp < -0.05 * std_val:
                p_idx = p_search_start + np.argmin(p_segment)
                p_is_inverted = True

        if p_idx is not None:
            p_amp_mv = sig_mv[p_idx]
            p_local_y = int(np.clip(baseline_y - sig[p_idx], 2, h - 2))
            p_box_w = max(6, int(0.06 * px_per_sec))
            p_box_h = max(6, int(abs(sig[p_idx]) * 0.5) + 6)
            wave_key = "INVERTED_P_TROUGH" if p_is_inverted else "P_PEAK"

            p_box = {
                "type": wave_key,
                "name": f"{'Inverted P-Trough' if p_is_inverted else 'P-Crest'} #{i+1}",
                "kind": "trough" if p_is_inverted else "crest",
                "index": int(p_idx),
                "time_sec": float(time_sec[p_idx]),
                "amplitude_mv": float(p_amp_mv),
                "is_inverted": bool(p_is_inverted),
                "local_box": [
                    max(0, p_idx - p_box_w // 2),
                    max(0, p_local_y - p_box_h // 2),
                    min(w, p_idx + p_box_w // 2),
                    min(h, p_local_y + p_box_h // 2),
                ],
                "global_box": [
                    max(0, x_off + p_idx - p_box_w // 2),
                    max(0, y_off + p_local_y - p_box_h // 2),
                    min(x_off + w, x_off + p_idx + p_box_w // 2),
                    min(y_off + h, y_off + p_local_y + p_box_h // 2),
                ],
                "color_bgr": WAVE_FIDUCIAL_COLORS[wave_key]["bgr"],
                "color_hex": WAVE_FIDUCIAL_COLORS[wave_key]["hex"],
            }
            if p_is_inverted:
                troughs.append(p_box)
            else:
                crests.append(p_box)
            all_boxes.append(p_box)

            # PR Interval (ms) = time from P-peak start to Q/R start
            pr_ms = (time_sec[r_idx] - time_sec[p_idx]) * 1000.0
            pr_intervals_ms.append(pr_ms)

        # --- T-Wave (Crest or Inverted Trough in post-QRS zone) ---
        t_search_start = min(n_samples, s_idx + int(0.04 * px_per_sec))
        t_search_end = min(n_samples, r_idx + win_t)
        t_idx = None
        t_is_inverted = False

        if t_search_end > t_search_start + 3:
            t_segment = sig[t_search_start:t_search_end]
            max_t_amp = np.max(t_segment)
            min_t_amp = np.min(t_segment)

            if abs(max_t_amp) >= abs(min_t_amp) and max_t_amp > 0.08 * std_val:
                t_idx = t_search_start + np.argmax(t_segment)
                t_is_inverted = False
            elif abs(min_t_amp) > abs(max_t_amp) and min_t_amp < -0.08 * std_val:
                t_idx = t_search_start + np.argmin(t_segment)
                t_is_inverted = True

        if t_idx is not None:
            t_amp_mv = sig_mv[t_idx]
            t_local_y = int(np.clip(baseline_y - sig[t_idx], 2, h - 2))
            t_box_w = max(8, int(0.08 * px_per_sec))
            t_box_h = max(6, int(abs(sig[t_idx]) * 0.5) + 6)
            wave_key = "INVERTED_T_TROUGH" if t_is_inverted else "T_PEAK"

            t_box = {
                "type": wave_key,
                "name": f"{'Inverted T-Trough' if t_is_inverted else 'T-Crest'} #{i+1}",
                "kind": "trough" if t_is_inverted else "crest",
                "index": int(t_idx),
                "time_sec": float(time_sec[t_idx]),
                "amplitude_mv": float(t_amp_mv),
                "is_inverted": bool(t_is_inverted),
                "local_box": [
                    max(0, t_idx - t_box_w // 2),
                    max(0, t_local_y - t_box_h // 2),
                    min(w, t_idx + t_box_w // 2),
                    min(h, t_local_y + t_box_h // 2),
                ],
                "global_box": [
                    max(0, x_off + t_idx - t_box_w // 2),
                    max(0, y_off + t_local_y - t_box_h // 2),
                    min(x_off + w, x_off + t_idx + t_box_w // 2),
                    min(y_off + h, y_off + t_local_y + t_box_h // 2),
                ],
                "color_bgr": WAVE_FIDUCIAL_COLORS[wave_key]["bgr"],
                "color_hex": WAVE_FIDUCIAL_COLORS[wave_key]["hex"],
            }
            if t_is_inverted:
                troughs.append(t_box)
            else:
                crests.append(t_box)
            all_boxes.append(t_box)

        # Record complex info
        complexes.append({
            "r_idx": r_idx,
            "q_idx": q_idx,
            "s_idx": s_idx,
            "p_idx": p_idx,
            "t_idx": t_idx,
            "qrs_dur_ms": qrs_dur_ms,
            "p_is_inverted": p_is_inverted,
            "t_is_inverted": t_is_inverted,
        })

    # -------------------------------------------------------------------------
    # 3. Calculate Intervals & Rhythm Metrics
    # -------------------------------------------------------------------------
    for i in range(1, len(r_times)):
        rr_intervals_ms.append((r_times[i] - r_times[i - 1]) * 1000.0)

    mean_rr_ms = float(np.mean(rr_intervals_ms)) if len(rr_intervals_ms) > 0 else 800.0
    std_rr_ms = float(np.std(rr_intervals_ms)) if len(rr_intervals_ms) > 0 else 0.0
    rr_cv = (std_rr_ms / max(1.0, mean_rr_ms)) if len(rr_intervals_ms) > 0 else 0.0
    heart_rate_bpm = float(60000.0 / max(100.0, mean_rr_ms)) if mean_rr_ms > 0 else 0.0

    mean_pr_ms = float(np.mean(pr_intervals_ms)) if len(pr_intervals_ms) > 0 else 160.0
    mean_qrs_ms = float(np.mean(qrs_durations_ms)) if len(qrs_durations_ms) > 0 else 90.0

    # -------------------------------------------------------------------------
    # 4. Search for Flutter / Fibrillatory Baseline Crests & Troughs
    # -------------------------------------------------------------------------
    # Check baseline oscillations between QRS complexes
    # F-waves occur at 250-400 bpm -> interval ~ 0.15s - 0.24s
    min_f_dist = max(3, int(0.14 * px_per_sec))
    max_f_dist = max(5, int(0.26 * px_per_sec))
    flutter_peaks, _ = signal.find_peaks(sig, distance=min_f_dist, prominence=0.6 * std_val)
    
    # Exclude peaks that belong to R-peaks, P-waves, or T-waves
    known_wave_indices = set(r_peaks)
    for comp in complexes:
        if comp["p_idx"] is not None:
            known_wave_indices.add(comp["p_idx"])
        if comp["t_idx"] is not None:
            known_wave_indices.add(comp["t_idx"])

    non_r_peaks = [
        p for p in flutter_peaks
        if not any(abs(p - k) < int(0.06 * px_per_sec) for k in known_wave_indices)
    ]

    # If high density of rapid regular saw-tooth oscillations (~250-400 bpm)
    if len(non_r_peaks) >= 5:
        peak_diffs = np.diff(non_r_peaks)
        mean_diff = np.mean(peak_diffs)
        if len(peak_diffs) > 0 and min_f_dist <= mean_diff <= max_f_dist and np.std(peak_diffs) < 0.25 * mean_diff:
            # Flutter Sawtooth Crests Detected!
            for f_idx in non_r_peaks:
                f_time = time_sec[f_idx]
                f_local_y = int(np.clip(baseline_y - sig[f_idx], 2, h - 2))
                f_box_w = max(5, int(0.04 * px_per_sec))
                f_box_h = max(5, int(abs(sig[f_idx]) * 0.4) + 4)
                f_box = {
                    "type": "FLUTTER_CREST",
                    "name": f"Sawtooth F-Crest @ {f_time:.2f}s",
                    "kind": "crest",
                    "index": int(f_idx),
                    "time_sec": float(f_time),
                    "amplitude_mv": float(sig_mv[f_idx]),
                    "local_box": [
                        max(0, f_idx - f_box_w // 2),
                        max(0, f_local_y - f_box_h // 2),
                        min(w, f_idx + f_box_w // 2),
                        min(h, f_local_y + f_box_h // 2),
                    ],
                    "global_box": [
                        max(0, x_off + f_idx - f_box_w // 2),
                        max(0, y_off + f_local_y - f_box_h // 2),
                        min(x_off + w, x_off + f_idx + f_box_w // 2),
                        min(y_off + h, y_off + f_local_y + f_box_h // 2),
                    ],
                    "color_bgr": WAVE_FIDUCIAL_COLORS["FLUTTER_CREST"]["bgr"],
                    "color_hex": WAVE_FIDUCIAL_COLORS["FLUTTER_CREST"]["hex"],
                }
                crests.append(f_box)
                all_boxes.append(f_box)

    # -------------------------------------------------------------------------
    # 5. Detect and Bounding-Box Anomalies
    # -------------------------------------------------------------------------
    # A) Wide / Distorted / Premature QRS (PVC, Ectopic Beat)
    for idx_c, comp in enumerate(complexes):
        r_i = comp["r_idx"]
        q_i = comp["q_idx"]
        s_i = comp["s_idx"]
        dur = comp["qrs_dur_ms"]

        # Check if premature relative to preceding RR
        is_premature = False
        if idx_c > 0 and len(rr_intervals_ms) >= idx_c:
            current_rr = rr_intervals_ms[idx_c - 1]
            if current_rr < 0.75 * mean_rr_ms:
                is_premature = True

        # Wide QRS anomaly (>120ms - 140ms) or premature bizarre complex
        if dur > 130.0 or (is_premature and dur > 110.0):
            pvc_color_bgr = DISORDER_COLORS["Premature Ventricular Contraction (PVC)"]["bgr"]
            pvc_color_hex = DISORDER_COLORS["Premature Ventricular Contraction (PVC)"]["hex"]
            x1_a = max(0, q_i - int(0.04 * px_per_sec))
            x2_a = min(w, s_i + int(0.06 * px_per_sec))
            y1_a = max(0, int(baseline_y - max(sig[x1_a:x2_a])) - 6)
            y2_a = min(h, int(baseline_y - min(sig[x1_a:x2_a])) + 6)

            anomaly_box = {
                "type": "ANOMALY_PVC_COMPLEX",
                "name": f"Anomalous Wide/Premature QRS (PVC) #{idx_c+1}",
                "kind": "anomaly",
                "disorder_match": "Premature Ventricular Contraction (PVC)",
                "duration_ms": float(dur),
                "time_sec": float(time_sec[r_i]),
                "local_box": [x1_a, y1_a, x2_a, y2_a],
                "global_box": [x_off + x1_a, y_off + y1_a, x_off + x2_a, y_off + y2_a],
                "color_bgr": pvc_color_bgr,
                "color_hex": pvc_color_hex,
                "note": f"QRS width {dur:.0f}ms (>130ms) with distorted morphology",
            }
            anomalies.append(anomaly_box)
            all_boxes.append(anomaly_box)

    # B) Prolonged PR Interval (>200ms -> 1st Degree AV Block segment)
    for idx_c, comp in enumerate(complexes):
        p_i = comp["p_idx"]
        r_i = comp["r_idx"]
        if p_i is not None and r_i > p_i:
            pr_val = (time_sec[r_i] - time_sec[p_i]) * 1000.0
            if pr_val > 200.0:
                av_color_bgr = DISORDER_COLORS["First-Degree AV Block"]["bgr"]
                av_color_hex = DISORDER_COLORS["First-Degree AV Block"]["hex"]
                x1_a = max(0, p_i - 2)
                x2_a = min(w, r_i + 2)
                y1_a = max(0, int(min(baseline_y - sig[p_i], baseline_y - sig[r_i])) - 4)
                y2_a = min(h, int(max(baseline_y - sig[p_i], baseline_y - sig[r_i])) + 4)

                anomaly_box = {
                    "type": "ANOMALY_PROLONGED_PR",
                    "name": f"Prolonged PR Interval ({pr_val:.0f}ms) #{idx_c+1}",
                    "kind": "anomaly",
                    "disorder_match": "First-Degree AV Block",
                    "duration_ms": float(pr_val),
                    "time_sec": float(time_sec[p_i]),
                    "local_box": [x1_a, y1_a, x2_a, y2_a],
                    "global_box": [x_off + x1_a, y_off + y1_a, x_off + x2_a, y_off + y2_a],
                    "color_bgr": av_color_bgr,
                    "color_hex": av_color_hex,
                    "note": f"Prolonged PR {pr_val:.0f}ms > 200ms",
                }
                anomalies.append(anomaly_box)
                all_boxes.append(anomaly_box)

    # C) Rhythm Pauses / SA Arrest / Block Segment
    for idx_rr, rr_val in enumerate(rr_intervals_ms):
        if rr_val > 1.6 * mean_rr_ms and rr_val > 1200.0:
            pause_color_bgr = DISORDER_COLORS["Sinoatrial (SA) Arrest or Block"]["bgr"]
            pause_color_hex = DISORDER_COLORS["Sinoatrial (SA) Arrest or Block"]["hex"]
            r1_idx = r_peaks[idx_rr]
            r2_idx = r_peaks[idx_rr + 1]
            x1_a = r1_idx + int(0.08 * px_per_sec)
            x2_a = r2_idx - int(0.08 * px_per_sec)
            y1_a = max(0, int(baseline_y - 0.5 * px_per_mv))
            y2_a = min(h, int(baseline_y + 0.5 * px_per_mv))

            anomaly_box = {
                "type": "ANOMALY_PAUSE_BLOCK",
                "name": f"Rhythm Pause / SA Arrest ({rr_val/1000.0:.2f}s) #{idx_rr+1}",
                "kind": "anomaly",
                "disorder_match": "Sinoatrial (SA) Arrest or Block",
                "duration_ms": float(rr_val),
                "time_sec": float(time_sec[r1_idx]),
                "local_box": [x1_a, y1_a, x2_a, y2_a],
                "global_box": [x_off + x1_a, y_off + y1_a, x_off + x2_a, y_off + y2_a],
                "color_bgr": pause_color_bgr,
                "color_hex": pause_color_hex,
                "note": f"Pause length {rr_val/1000.0:.2f}s (> 1.6x baseline RR)",
            }
            anomalies.append(anomaly_box)
            all_boxes.append(anomaly_box)

    # D) Inverted P-Wave Anomaly (Junctional Rhythm / PJC)
    inverted_p_boxes = [b for b in troughs if b.get("is_inverted") and "P-Trough" in b.get("name", "")]
    if len(inverted_p_boxes) > 0:
        for idx_ip, ip in enumerate(inverted_p_boxes):
            junc_color_bgr = DISORDER_COLORS.get("Junctional Rhythm", {}).get("bgr", (0, 140, 255))
            junc_color_hex = DISORDER_COLORS.get("Junctional Rhythm", {}).get("hex", "#FF8C00")
            anom_junc = {
                "type": "ANOMALY_INVERTED_P",
                "name": f"Inverted Retrograde P-Wave #{idx_ip+1}",
                "kind": "anomaly",
                "disorder_match": "Junctional Rhythm",
                "time_sec": float(ip["time_sec"]),
                "local_box": ip["local_box"],
                "global_box": ip["global_box"],
                "color_bgr": junc_color_bgr,
                "color_hex": junc_color_hex,
                "note": "Negative P-wave deflection indicating junctional/retrograde conduction",
            }
            anomalies.append(anom_junc)
            all_boxes.append(anom_junc)

    # E) Pathological T-Wave Inversion (Ischemia / Ventricular Strain)
    # Exclude aVR and V1 where inverted T-wave is normal
    if "aVR" not in lead_label and "V1" not in lead_label:
        inverted_t_boxes = [b for b in troughs if b.get("is_inverted") and "T-Trough" in b.get("name", "")]
        for idx_it, it in enumerate(inverted_t_boxes):
            t_inv_color_bgr = DISORDER_COLORS.get("Myocardial Infarction", {}).get("bgr", (0, 0, 255))
            t_inv_color_hex = DISORDER_COLORS.get("Myocardial Infarction", {}).get("hex", "#FF1744")
            anom_t = {
                "type": "ANOMALY_INVERTED_T",
                "name": f"Pathological T-Wave Inversion #{idx_it+1}",
                "kind": "anomaly",
                "disorder_match": "Myocardial Infarction",
                "time_sec": float(it["time_sec"]),
                "local_box": it["local_box"],
                "global_box": it["global_box"],
                "color_bgr": t_inv_color_bgr,
                "color_hex": t_inv_color_hex,
                "note": f"Inverted T-wave ({it.get('amplitude_mv', 0):.2f} mV) in {lead_label} suggests myocardial ischemia or strain",
            }
            anomalies.append(anom_t)
            all_boxes.append(anom_t)

    # F) ST-Segment Elevation / Depression (STEMI / Ischemia)
    for idx_c, comp in enumerate(complexes):
        s_i = comp["s_idx"]
        t_i = comp["t_idx"]
        if s_i is not None and t_i is not None and t_i > s_i + int(0.04 * px_per_sec):
            # ST segment is evaluated ~60ms after S/J point
            st_start = min(n_samples - 1, s_i + int(0.03 * px_per_sec))
            st_end = max(st_start + 1, min(n_samples, s_i + int(0.09 * px_per_sec)))
            st_mean_mv = float(np.mean(sig_mv[st_start:st_end]))

            if st_mean_mv > 0.18: # Significant ST elevation (> 1.8 mm / 0.18 mV)
                stemi_color_bgr = (0, 0, 255)
                stemi_color_hex = "#FF1744"
                anom_st = {
                    "type": "ANOMALY_ST_ELEVATION",
                    "name": f"ST Elevation (+{st_mean_mv:.2f}mV) #{idx_c+1}",
                    "kind": "anomaly",
                    "disorder_match": "Myocardial Infarction",
                    "time_sec": float(time_sec[st_start]),
                    "local_box": [st_start, max(0, int(baseline_y - sig[st_start]) - 8), st_end, min(h, int(baseline_y) + 8)],
                    "global_box": [x_off + st_start, max(0, y_off + int(baseline_y - sig[st_start]) - 8), x_off + st_end, min(y_off + h, y_off + int(baseline_y) + 8)],
                    "color_bgr": stemi_color_bgr,
                    "color_hex": stemi_color_hex,
                    "note": f"ST elevation +{st_mean_mv:.2f} mV in {lead_label} - Acute STEMI injury pattern",
                }
                anomalies.append(anom_st)
                all_boxes.append(anom_st)
            elif st_mean_mv < -0.18: # Significant ST depression (< -0.18 mV)
                dep_color_bgr = (0, 165, 255)
                dep_color_hex = "#FFA500"
                anom_st_dep = {
                    "type": "ANOMALY_ST_DEPRESSION",
                    "name": f"ST Depression ({st_mean_mv:.2f}mV) #{idx_c+1}",
                    "kind": "anomaly",
                    "disorder_match": "History of MI",
                    "time_sec": float(time_sec[st_start]),
                    "local_box": [st_start, max(0, int(baseline_y) - 8), st_end, min(h, int(baseline_y - sig[st_start]) + 8)],
                    "global_box": [x_off + st_start, max(0, y_off + int(baseline_y) - 8), x_off + st_end, min(y_off + h, y_off + int(baseline_y - sig[st_start]) + 8)],
                    "color_bgr": dep_color_bgr,
                    "color_hex": dep_color_hex,
                    "note": f"ST depression {st_mean_mv:.2f} mV in {lead_label} - Subendocardial ischemia or reciprocal change",
                }
                anomalies.append(anom_st_dep)
                all_boxes.append(anom_st_dep)

    return {
        "lead_label": lead_label,
        "crests": crests,
        "troughs": troughs,
        "anomalies": anomalies,
        "all_boxes": all_boxes,
        "metrics": {
            "heart_rate_bpm": heart_rate_bpm,
            "mean_rr_ms": mean_rr_ms,
            "std_rr_ms": std_rr_ms,
            "rr_cv": rr_cv,
            "mean_pr_ms": mean_pr_ms,
            "mean_qrs_ms": mean_qrs_ms,
            "n_crests": len(crests),
            "n_troughs": len(troughs),
            "n_anomalies": len(anomalies),
            "r_peak_count": len(r_peaks),
            "p_wave_count": len([c for c in crests if "P-Crest" in c["name"]]),
            "inverted_p_count": len(inverted_p_boxes),
        }
    }
