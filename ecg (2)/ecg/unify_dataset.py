"""
unify_dataset.py
----------------
Creates a clean, unified 'dataset/15_disorders' directory containing exactly
15 subfolders (one for each disorder), cleanly numbered and named.
Uses fast NTFS hardlinks (0 duplicate disk space, instantaneous, works natively in Windows).
"""

import os
import shutil

DATASET_ROOT = "dataset"
TARGET_DIR = os.path.join(DATASET_ROOT, "15_disorders")

# Mapping: (Source Subdirectory, Unified Destination Subdirectory)
CLASS_SOURCES = [
    # 4 Original Scanned ECG Classes
    (os.path.join(DATASET_ROOT, "Normal Person ECG Images (284x12=3408)"),
     "01_Normal_Sinus_Rhythm"),
    (os.path.join(DATASET_ROOT, "ECG Images of Myocardial Infarction Patients (240x12=2880)"),
     "02_Myocardial_Infarction"),
    (os.path.join(DATASET_ROOT, "ECG Images of Patient that have History of MI (172x12=2064)"),
     "03_History_of_MI"),
    (os.path.join(DATASET_ROOT, "ECG Images of Patient that have abnormal heartbeat (233x12=2796)"),
     "04_Abnormal_Heartbeat"),

    # 11 PTB-XL Rendered Classes
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Atrial Fibrillation"),
     "05_Atrial_Fibrillation"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Atrial Flutter"),
     "06_Atrial_Flutter"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Sinus Tachycardia"),
     "07_Sinus_Tachycardia"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Sinus Bradycardia"),
     "08_Sinus_Bradycardia"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Complete Left Bundle Branch Block"),
     "09_Complete_Left_Bundle_Branch_Block"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Complete Right Bundle Branch Block"),
     "10_Complete_Right_Bundle_Branch_Block"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Incomplete Right Bundle Branch Block"),
     "11_Incomplete_Right_Bundle_Branch_Block"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "First Degree AV Block"),
     "12_First_Degree_AV_Block"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Left Anterior Fascicular Block"),
     "13_Left_Anterior_Fascicular_Block"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Left Ventricular Hypertrophy"),
     "14_Left_Ventricular_Hypertrophy"),
    (os.path.join(DATASET_ROOT, "rendered_ptbxl", "Left Atrial Enlargement"),
     "15_Left_Atrial_Enlargement"),
]

def unify():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print("=" * 70)
    print(f"Creating Unified 15-Disorder Dataset in: {TARGET_DIR}")
    print("=" * 70)

    total_linked = 0
    for src_dir, dest_name in CLASS_SOURCES:
        dest_dir = os.path.join(TARGET_DIR, dest_name)
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            print(f"[!] Warning: Source directory not found: {src_dir}")
            continue

        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = 0
        for img_name in images:
            src_file = os.path.join(src_dir, img_name)
            dest_file = os.path.join(dest_dir, img_name)

            if not os.path.exists(dest_file):
                try:
                    os.link(src_file, dest_file)
                except Exception:
                    shutil.copy2(src_file, dest_file)
            count += 1

        total_linked += count
        print(f"  [OK] {dest_name:<42} : {count:>5} images")

    print("-" * 70)
    print(f"Total images organized across 15 folders: {total_linked}")
    print("=" * 70)

if __name__ == "__main__":
    unify()
