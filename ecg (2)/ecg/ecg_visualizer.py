"""
ecg_visualizer.py
-----------------
Renders rich clinical visual annotations on ECG images:
- Lead bounding boxes and labels
- Individual bounding boxes for every Crest (P, R, T, Flutter) and Trough (Q, S, Inverted P/T)
- Color-coded Anomaly bounding boxes styled in each Disorder's unique color code
- High-visibility header banner with primary diagnosis, metrics, and color legend
"""

import cv2
import numpy as np
from disorder_palette import (
    DISORDER_COLORS,
    WAVE_FIDUCIAL_COLORS,
    get_disorder_color,
    get_fiducial_color,
)


def annotate_ecg_image(
    image,
    lead_boxes,
    all_fids_and_anomalies,
    primary_disorder_info,
    metrics,
    show_crests=True,
    show_troughs=True,
    show_anomalies=True,
    show_lead_boxes=True,
    show_legend=True,
):
    """
    Renders an annotated ECG image with bounding boxes for every crest, trough,
    anomaly, and lead region, along with a top clinical banner and color legend.

    Parameters:
        image: Original BGR image.
        lead_boxes: List of (label, x1, y1, x2, y2) tuples.
        all_fids_and_anomalies: List of box dicts from wave_detector.
        primary_disorder_info: Top ranked disorder info dict from arrhythmia_classifier.
        metrics: Cardiac metrics dictionary.
        show_*: Boolean flags to toggle layers.

    Returns:
        Annotated BGR numpy image.
    """
    h_orig, w_orig = image.shape[:2]
    annotated = image.copy()

    # 1. Draw Lead Regions (subtle semi-transparent lines)
    if show_lead_boxes and lead_boxes:
        lead_overlay = annotated.copy()
        for label, x1, y1, x2, y2 in lead_boxes:
            cv2.rectangle(lead_overlay, (x1, y1), (x2, y2), (0, 180, 0), 1)
            # Label badge
            badge_text = f" {label} "
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(lead_overlay, (x1, y1), (x1 + tw + 4, y1 + th + 6), (0, 100, 0), -1)
            cv2.putText(
                lead_overlay, badge_text, (x1 + 2, y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
        cv2.addWeighted(lead_overlay, 0.7, annotated, 0.3, 0, annotated)

    # 2. Draw Troughs (Valleys)
    if show_troughs:
        for item in all_fids_and_anomalies:
            if item.get("kind") == "trough":
                gx1, gy1, gx2, gy2 = item["global_box"]
                color = item.get("color_bgr", (255, 255, 0))
                # Draw trough box
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), color, 1)
                # Small corner mark
                cv2.drawMarker(
                    annotated, ((gx1 + gx2) // 2, (gy1 + gy2) // 2),
                    color, cv2.MARKER_TILTED_CROSS, 4, 1
                )

    # 3. Draw Crests (Peaks)
    if show_crests:
        for item in all_fids_and_anomalies:
            if item.get("kind") == "crest":
                gx1, gy1, gx2, gy2 = item["global_box"]
                color = item.get("color_bgr", (0, 255, 0))
                # Draw crest box
                is_r = item.get("type") == "R_PEAK"
                thick = 2 if is_r else 1
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), color, thick)
                # Small apex dot
                cv2.circle(annotated, ((gx1 + gx2) // 2, (gy1 + gy2) // 2), 2, color, -1)

    # 4. Draw Anomalous Regions (Thick Bounding Boxes with Disorder Unique Color)
    if show_anomalies:
        for item in all_fids_and_anomalies:
            if item.get("kind") == "anomaly":
                gx1, gy1, gx2, gy2 = item["global_box"]
                disorder_name = item.get("disorder_match", primary_disorder_info.get("disorder", "Anomaly"))
                disorder_color = get_disorder_color(disorder_name, "bgr")

                # Thick bounding box
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), disorder_color, 2)

                # Anomaly Label Badge
                anom_label = item.get("name", "Anomaly")
                if len(anom_label) > 28:
                    anom_label = anom_label[:25] + "..."
                (tw, th), _ = cv2.getTextSize(anom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                by1 = max(0, gy1 - th - 4)
                by2 = gy1
                bx1 = gx1
                bx2 = min(w_orig, gx1 + tw + 6)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), disorder_color, -1)
                cv2.putText(
                    annotated, anom_label, (bx1 + 3, by2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA
                )

    # 5. Render Top Diagnostic Banner & Metrics
    banner_height = 80
    legend_height = 45 if show_legend else 0
    total_new_h = h_orig + banner_height + legend_height
    canvas = np.zeros((total_new_h, w_orig, 3), dtype=np.uint8)

    # Dark slate background for header banner
    canvas[:banner_height, :] = (24, 28, 36)

    # Primary Disorder Info
    dis_name = primary_disorder_info.get("disorder", "ECG Analysis")
    dis_conf = primary_disorder_info.get("confidence_pct", 90.0)
    dis_color_bgr = primary_disorder_info.get("color_bgr", (0, 230, 118))
    hr_val = metrics.get("heart_rate_bpm", 75.0)
    rr_cv = metrics.get("rr_cv", 0.05) * 100.0
    pr_val = metrics.get("mean_pr_ms", 160.0)
    qrs_val = metrics.get("mean_qrs_ms", 90.0)
    n_crests = metrics.get("n_crests", 0)
    n_troughs = metrics.get("n_troughs", 0)
    n_anomalies = metrics.get("n_anomalies", 0)

    # Header Badge (Color coded to primary disorder)
    cv2.rectangle(canvas, (12, 10), (18, 42), dis_color_bgr, -1)
    cv2.putText(
        canvas, f"DISORDER: {dis_name.upper()}", (26, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas, f"Confidence: {dis_conf:.1f}%", (26, 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, dis_color_bgr, 1, cv2.LINE_AA
    )

    # Metrics Strip
    metrics_str = (
        f"HR: {hr_val:.0f} BPM  |  RR-CV: {rr_cv:.1f}%  |  "
        f"PR: {pr_val:.0f} ms  |  QRS: {qrs_val:.0f} ms  |  "
        f"Crests: {n_crests}  |  Troughs: {n_troughs}  |  Anomalies: {n_anomalies}"
    )
    cv2.putText(
        canvas, metrics_str, (12, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1, cv2.LINE_AA
    )

    # Place annotated ECG image in canvas
    canvas[banner_height:banner_height + h_orig, :w_orig] = annotated

    # 6. Render Bottom Color Legend
    if show_legend:
        leg_y_start = banner_height + h_orig
        canvas[leg_y_start:, :] = (18, 22, 28)

        legend_items = [
            ("R-Crest", WAVE_FIDUCIAL_COLORS["R_PEAK"]["bgr"]),
            ("P-Crest", WAVE_FIDUCIAL_COLORS["P_PEAK"]["bgr"]),
            ("T-Crest", WAVE_FIDUCIAL_COLORS["T_PEAK"]["bgr"]),
            ("Q-Trough", WAVE_FIDUCIAL_COLORS["Q_TROUGH"]["bgr"]),
            ("S-Trough", WAVE_FIDUCIAL_COLORS["S_TROUGH"]["bgr"]),
            ("Inv P/T", (180, 105, 255)),
            ("Anomaly (Disorder Color)", dis_color_bgr),
        ]

        curr_x = 15
        leg_y = leg_y_start + 26
        cv2.putText(canvas, "LEGEND:", (curr_x, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        curr_x += 70

        for lbl, col in legend_items:
            # Color swatch
            cv2.rectangle(canvas, (curr_x, leg_y - 10), (curr_x + 12, leg_y + 2), col, -1)
            cv2.rectangle(canvas, (curr_x, leg_y - 10), (curr_x + 12, leg_y + 2), (255, 255, 255), 1)
            curr_x += 16
            cv2.putText(canvas, lbl, (curr_x, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            curr_x += tw + 18

    return canvas
