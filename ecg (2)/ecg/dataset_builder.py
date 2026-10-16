"""
dataset_builder.py
------------------
Renders PTB-XL (.dat / .hea) WFDB clinical records into standard 12-lead ECG
grid JPEG images for 11 NEW disorder classes.

These 11 new rendered classes + the 4 original image folder classes = 15 total.

New classes extracted from PTB-XL (saved to dataset/rendered_ptbxl/<ClassName>/):
    1.  Atrial Fibrillation
    2.  Atrial Flutter
    3.  Sinus Tachycardia
    4.  Sinus Bradycardia
    5.  Complete Left Bundle Branch Block
    6.  Complete Right Bundle Branch Block
    7.  Incomplete Right Bundle Branch Block
    8.  First Degree AV Block
    9.  Left Anterior Fascicular Block
    10. Left Ventricular Hypertrophy
    11. Left Atrial Enlargement

Original 4 classes (image folders, untouched):
    - ECG Images of Myocardial Infarction Patients (240x12=2880)
    - ECG Images of Patient that have History of MI (172x12=2064)
    - ECG Images of Patient that have abnormal heartbeat (233x12=2796)
    - Normal Person ECG Images (284x12=3408)
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
import cv2

DATASET_DIR = "dataset"
OUTPUT_RENDER_DIR = os.path.join(DATASET_DIR, "rendered_ptbxl")

# 11 NEW class names rendered from PTB-XL signals
NEW_CLASSES = [
    "Atrial Fibrillation",
    "Atrial Flutter",
    "Sinus Tachycardia",
    "Sinus Bradycardia",
    "Complete Left Bundle Branch Block",
    "Complete Right Bundle Branch Block",
    "Incomplete Right Bundle Branch Block",
    "First Degree AV Block",
    "Left Anterior Fascicular Block",
    "Left Ventricular Hypertrophy",
    "Left Atrial Enlargement",
]

# Priority-ordered SCP code mapping.
# Each record is checked top-to-bottom; first match wins (one record = one class).
CLASS_SCP_PRIORITY = [
    ("Atrial Fibrillation",                  ["AFIB"]),
    ("Atrial Flutter",                       ["AFLT"]),
    ("Complete Left Bundle Branch Block",    ["CLBBB"]),
    ("Complete Right Bundle Branch Block",   ["CRBBB"]),
    ("Incomplete Right Bundle Branch Block", ["IRBBB"]),
    ("First Degree AV Block",                ["1AVB"]),
    ("Left Anterior Fascicular Block",       ["LAFB"]),
    ("Left Ventricular Hypertrophy",         ["LVH"]),
    ("Left Atrial Enlargement",              ["LAO/LAE"]),
    ("Sinus Tachycardia",                    ["STACH"]),
    ("Sinus Bradycardia",                    ["SBRAD"]),
]

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def setup_directories():
    for c in NEW_CLASSES:
        os.makedirs(os.path.join(OUTPUT_RENDER_DIR, c), exist_ok=True)


def map_record_to_new_class(scp_codes_dict):
    """
    Map a PTB-XL record to one of the 11 new classes using priority order.
    Returns class name string, or None if no match.
    """
    for class_name, scp_codes in CLASS_SCP_PRIORITY:
        for code in scp_codes:
            if code in scp_codes_dict:
                return class_name
    return None


def render_ecg_cv2(signals, out_path, img_w=1000, img_h=700):
    """
    Render a 12-lead ECG signal as a standard 3x4 medical ECG grid JPEG.
    signals shape: [num_samples, 12]  (millivolt values from wfdb.rdsamp)
    """
    canvas = np.full((img_h, img_w, 3), (242, 246, 252), dtype=np.uint8)

    # Minor grid (1mm equivalent)
    for x in range(0, img_w, 10):
        cv2.line(canvas, (x, 0), (x, img_h), (225, 230, 245), 1)
    for y in range(0, img_h, 10):
        cv2.line(canvas, (0, y), (img_w, y), (225, 230, 245), 1)

    # Major grid (5mm equivalent)
    for x in range(0, img_w, 50):
        cv2.line(canvas, (x, 0), (x, img_h), (200, 210, 235), 1)
    for y in range(0, img_h, 50):
        cv2.line(canvas, (0, y), (img_w, y), (200, 210, 235), 1)

    grid_layout = [
        ["I",   "aVR", "V1", "V4"],
        ["II",  "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    row_h = img_h / 3.0
    col_w = img_w / 4.0
    samples_per_lead = min(signals.shape[0], 1000)

    for r in range(3):
        for c in range(4):
            lead_name = grid_layout[r][c]
            lead_idx  = LEAD_NAMES.index(lead_name)
            raw_sig   = signals[:samples_per_lead, lead_idx]

            x_min = int(c * col_w)
            x_max = int((c + 1) * col_w)
            y_min = int(r * row_h)
            y_max = int((r + 1) * row_h)
            y_ctr = (y_min + y_max) / 2.0

            scale_y = (row_h * 0.35) / max(0.5, np.percentile(np.abs(raw_sig), 98))
            px_y = np.clip(y_ctr - (raw_sig * scale_y), y_min + 5, y_max - 5).astype(np.int32)
            px_x = np.linspace(x_min + 10, x_max - 10, len(raw_sig)).astype(np.int32)

            pts = np.column_stack((px_x, px_y)).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(20, 20, 20), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(canvas, lead_name, (x_min + 15, y_min + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 30, 20), 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (180, 180, 180), 1)

    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def build_rendered_dataset(max_samples_per_class=350, sampling_rate=100):
    """
    Parse PTB-XL, map records to 11 new disorder classes, render as JPEG images.

    Parameters
    ----------
    max_samples_per_class : int  - cap images per new class (for class balance)
    sampling_rate         : int  - 100 for LR records, 500 for HR records
    """
    print("=" * 65, flush=True)
    print("   PTB-XL SIGNAL -> 11 NEW DISORDER CLASS IMAGE RENDERER", flush=True)
    print("=" * 65, flush=True)

    setup_directories()

    csv_path = os.path.join(DATASET_DIR, "ptbxl_database.csv")
    scp_path = os.path.join(DATASET_DIR, "scp_statements.csv")

    if not os.path.exists(csv_path) or not os.path.exists(scp_path):
        print(f"[ERROR] Missing PTB-XL files:\n  {csv_path}\n  {scp_path}", flush=True)
        return

    print("\nLoading PTB-XL metadata ...", flush=True)
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes_dict"] = df["scp_codes"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
    )
    print(f"Total PTB-XL records loaded: {len(df)}", flush=True)

    df["new_class"] = df["scp_codes_dict"].apply(map_record_to_new_class)
    mapped_counts = df["new_class"].value_counts()

    print(f"\n  {'Class':<47} {'Available':>10} {'Will Render':>12}")
    print(f"  {'-'*70}")
    for cls in NEW_CLASSES:
        avail = mapped_counts.get(cls, 0)
        will  = min(avail, max_samples_per_class)
        print(f"  {cls:<47} {avail:>10} {will:>12}", flush=True)

    # Count already-rendered images to resume if interrupted
    class_counts = {}
    for c in NEW_CLASSES:
        existing = len([f for f in os.listdir(os.path.join(OUTPUT_RENDER_DIR, c))
                        if f.lower().endswith(".jpg")])
        class_counts[c] = existing

    print(f"\nRendering up to {max_samples_per_class} images per class ...", flush=True)
    total_rendered = 0
    skipped = 0

    for ecg_id, row in df.iterrows():
        target = row["new_class"]
        if not isinstance(target, str) or target not in class_counts:
            continue
        if class_counts[target] >= max_samples_per_class:
            continue

        rel_path = row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
        full_record_path = os.path.join(DATASET_DIR, rel_path)
        out_img_path = os.path.join(OUTPUT_RENDER_DIR, target, f"ptbxl_{ecg_id}.jpg")

        if os.path.exists(out_img_path):
            class_counts[target] += 1
            continue

        try:
            sig, _ = wfdb.rdsamp(full_record_path)
            render_ecg_cv2(sig, out_img_path)
            class_counts[target] += 1
            total_rendered += 1
            if total_rendered % 100 == 0:
                progress = " | ".join(f"{c[:18]}: {class_counts.get(c, 0)}" for c in NEW_CLASSES)
                print(f"  [{total_rendered:>4} rendered] {progress}", flush=True)
        except Exception:
            skipped += 1

        if all(class_counts.get(c, 0) >= max_samples_per_class for c in NEW_CLASSES):
            print("\n[+] All classes reached target - stopping early.", flush=True)
            break

    print(f"\n{'-'*65}", flush=True)
    print(f"[+] Done. New images: {total_rendered}  |  Failed/Skipped: {skipped}", flush=True)
    print("\nFinal image counts:", flush=True)
    for c in NEW_CLASSES:
        cnt = class_counts.get(c, 0)
        print(f"  {c:<47} {cnt:>4} images", flush=True)


if __name__ == "__main__":
    build_rendered_dataset(max_samples_per_class=350, sampling_rate=100)
