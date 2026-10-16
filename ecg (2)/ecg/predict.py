# -*- coding: utf-8 -*-
"""
predict.py
----------
ECG Analysis Pipeline with CNN as Primary Predictor.

1. Detect ECG region and 12-lead bounding boxes.
2. Extract 1D waveform signals, compute P/Q/R/S/T fiducial bounding boxes
   and morphological anomaly regions (supporting visual evidence).
3. Run the 15-class CNN model as the PRIMARY disorder classifier.
4. Map CNN output to clinical info (causes, color, category).
5. Render annotated image + save report and JSON.

Usage:
    python predict.py --image path/to/ecg.jpg [--out-dir outputs]
"""

# Clinical knowledge base: maps each of the 15 CNN class names to
# a short display name, category, unique color, and clinical causes.
CNN_CLASS_INFO = {
    "ECG Images of Myocardial Infarction Patients (240x12=2880)": {
        "short": "Myocardial Infarction",
        "category": "Ischemia / Infarction",
        "color_hex": "#FF1744",
        "color_bgr": (68, 23, 255),
        "causes": [
            "Acute coronary artery occlusion (plaque rupture)",
            "Thromboembolism in a coronary artery",
            "Coronary vasospasm (Prinzmetal angina)",
        ],
    },
    "ECG Images of Patient that have History of MI (172x12=2064)": {
        "short": "History of Myocardial Infarction",
        "category": "Ischemia / Infarction",
        "color_hex": "#FF6D00",
        "color_bgr": (0, 109, 255),
        "causes": [
            "Previous acute MI with residual scar tissue",
            "Chronic coronary artery disease",
            "Old Q-wave infarct with persistent ST-T changes",
        ],
    },
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": {
        "short": "Abnormal Heartbeat",
        "category": "Arrhythmia",
        "color_hex": "#AA00FF",
        "color_bgr": (255, 0, 170),
        "causes": [
            "Electrolyte imbalance (hypokalemia, hypomagnesemia)",
            "Structural heart disease",
            "Drug toxicity (digoxin, antiarrhythmics)",
        ],
    },
    "Normal Person ECG Images (284x12=3408)": {
        "short": "Normal Sinus Rhythm",
        "category": "Normal",
        "color_hex": "#00E676",
        "color_bgr": (118, 230, 0),
        "causes": ["Normal resting cardiac conduction"],
    },
    "Atrial Fibrillation": {
        "short": "Atrial Fibrillation",
        "category": "Supraventricular Arrhythmia",
        "color_hex": "#00B0FF",
        "color_bgr": (255, 176, 0),
        "causes": [
            "Hypertension and left atrial enlargement",
            "Valvular heart disease (mitral stenosis/regurgitation)",
            "Hyperthyroidism, alcohol excess, sleep apnea",
        ],
    },
    "Atrial Flutter": {
        "short": "Atrial Flutter",
        "category": "Supraventricular Arrhythmia",
        "color_hex": "#40C4FF",
        "color_bgr": (255, 196, 64),
        "causes": [
            "Right atrial enlargement or scarring",
            "Post-cardiac surgery",
            "Pulmonary embolism and COPD",
        ],
    },
    "Sinus Tachycardia": {
        "short": "Sinus Tachycardia",
        "category": "Sinus Rhythm Variant",
        "color_hex": "#FFD740",
        "color_bgr": (64, 215, 255),
        "causes": [
            "Fever, pain, anxiety, or exercise",
            "Hypovolemia, anemia, or sepsis",
            "Hyperthyroidism or stimulant use",
        ],
    },
    "Sinus Bradycardia": {
        "short": "Sinus Bradycardia",
        "category": "Sinus Rhythm Variant",
        "color_hex": "#69F0AE",
        "color_bgr": (174, 240, 105),
        "causes": [
            "Well-trained athletes (physiological)",
            "Beta-blockers, calcium channel blockers, digoxin",
            "Hypothyroidism or hypothermia",
        ],
    },
    "Complete Left Bundle Branch Block": {
        "short": "Complete LBBB",
        "category": "Conduction Disorder",
        "color_hex": "#E040FB",
        "color_bgr": (251, 64, 224),
        "causes": [
            "Coronary artery disease and acute MI",
            "Dilated cardiomyopathy",
            "Hypertension with left ventricular hypertrophy",
        ],
    },
    "Complete Right Bundle Branch Block": {
        "short": "Complete RBBB",
        "category": "Conduction Disorder",
        "color_hex": "#7C4DFF",
        "color_bgr": (255, 77, 124),
        "causes": [
            "Normal variant in young individuals",
            "Right ventricular overload (pulmonary embolism)",
            "Atrial septal defect",
        ],
    },
    "Incomplete Right Bundle Branch Block": {
        "short": "Incomplete RBBB",
        "category": "Conduction Disorder",
        "color_hex": "#B39DDB",
        "color_bgr": (219, 157, 179),
        "causes": [
            "Normal variant in athletes",
            "Mild right ventricular volume overload",
            "Atrial septal defect",
        ],
    },
    "First Degree AV Block": {
        "short": "1st Degree AV Block",
        "category": "Conduction Disorder",
        "color_hex": "#FF8A65",
        "color_bgr": (101, 138, 255),
        "causes": [
            "Increased vagal tone (athletes, elderly)",
            "Beta-blockers or digoxin",
            "Lyme disease",
        ],
    },
    "Left Anterior Fascicular Block": {
        "short": "Left Anterior Fascicular Block",
        "category": "Conduction Disorder",
        "color_hex": "#FFAB40",
        "color_bgr": (64, 171, 255),
        "causes": [
            "Coronary artery disease (most common)",
            "Hypertension and left ventricular hypertrophy",
            "Cardiomyopathy",
        ],
    },
    "Left Ventricular Hypertrophy": {
        "short": "Left Ventricular Hypertrophy",
        "category": "Structural",
        "color_hex": "#F06292",
        "color_bgr": (146, 98, 240),
        "causes": [
            "Systemic hypertension (most common cause)",
            "Aortic stenosis or regurgitation",
            "Hypertrophic cardiomyopathy",
        ],
    },
    "Left Atrial Enlargement": {
        "short": "Left Atrial Enlargement",
        "category": "Structural",
        "color_hex": "#80CBC4",
        "color_bgr": (196, 203, 128),
        "causes": [
            "Mitral valve disease",
            "Left ventricular dysfunction / heart failure",
            "Hypertension with diastolic dysfunction",
        ],
    },
}

import os
import json
import argparse
import cv2
import numpy as np

from ecg_utils import detect_ecg_region, get_lead_boxes
from ecg_signal_extractor import extract_all_leads_signals
from wave_detector import detect_fiducials_and_anomalies
from ecg_visualizer import annotate_ecg_image

OUTPUT_DIR       = "outputs"
CLASS_NAMES_PATH = "class_names.txt"
MODEL_PATH       = "model.keras"


def build_primary_from_cnn(cnn_class_name, cnn_confidence):
    """Build the primary_disorder dict from the CNN prediction."""
    info = CNN_CLASS_INFO.get(cnn_class_name, {})
    return {
        "disorder":          info.get("short", cnn_class_name),
        "full_class_name":   cnn_class_name,
        "category":          info.get("category", "Unknown"),
        "confidence_pct":    round(cnn_confidence * 100, 1),
        "color_hex":         info.get("color_hex", "#90A4AE"),
        "color_bgr":         info.get("color_bgr", (174, 164, 144)),
        "causes":            info.get("causes", ["Refer to cardiologist"]),
        "matching_criteria": ["15-class CNN image classifier (MobileNetV2)"],
        "source":            "CNN",
    }


def run_cnn(image, x, y, w, h, cnn_model, class_names):
    """Crop the ECG region and run the CNN. Returns (class_name, confidence, all_probs)."""
    import tensorflow as tf
    cropped = image[y:y + h, x:x + w]
    if cropped.size == 0:
        cropped = image
    resized = cv2.resize(cropped, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(rgb, axis=0)
    )
    probs   = cnn_model.predict(inp, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return (
        class_names[top_idx],
        float(probs[top_idx]),
        {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    )


def analyze_ecg_image(image_path, cnn_model=None, class_names=None):
    """
    Full ECG analysis pipeline.
    CNN is the primary disorder predictor.
    Waveform bounding boxes provide supporting visual evidence.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h_img, w_img = image.shape[:2]

    # 1. Detect ECG region and lead bounding boxes
    x, y, w, h = detect_ecg_region(image)
    lead_boxes  = get_lead_boxes(x, y, w, h)

    # 2. Extract 1D signals and compute fiducial + anomaly bounding boxes
    leads_signals = extract_all_leads_signals(image, lead_boxes)
    all_fids, all_crests, all_troughs, all_anomalies = [], [], [], []
    lead_metrics_list = []

    for label, sig_data in leads_signals.items():
        x1, y1, x2, y2 = sig_data["bbox"]
        det = detect_fiducials_and_anomalies(
            sig_data, lead_label=label, global_box_offset=(x1, y1)
        )
        if det:
            all_crests.extend(det["crests"])
            all_troughs.extend(det["troughs"])
            all_anomalies.extend(det["anomalies"])
            all_fids.extend(det["all_boxes"])
            lead_metrics_list.append(det["metrics"])

    # 3. Aggregate waveform metrics across all leads
    if lead_metrics_list:
        def _mean(vals, default):
            filtered = [v for v in vals if v > 0]
            return float(np.mean(filtered)) if filtered else default

        agg_metrics = {
            "heart_rate_bpm":   _mean([m["heart_rate_bpm"] for m in lead_metrics_list], 75.0),
            "mean_rr_ms":       _mean([m["mean_rr_ms"]     for m in lead_metrics_list], 800.0),
            "std_rr_ms":        float(np.mean([m["std_rr_ms"]  for m in lead_metrics_list])),
            "rr_cv":            float(np.mean([m["rr_cv"]      for m in lead_metrics_list])),
            "mean_pr_ms":       _mean([m["mean_pr_ms"]     for m in lead_metrics_list], 160.0),
            "mean_qrs_ms":      _mean([m["mean_qrs_ms"]    for m in lead_metrics_list], 90.0),
            "n_crests":         len(all_crests),
            "n_troughs":        len(all_troughs),
            "n_anomalies":      len(all_anomalies),
            "r_peak_count":     len([c for c in all_crests  if c.get("type") == "R_PEAK"]),
            "p_wave_count":     len([c for c in all_crests  if "P-Crest" in c.get("name", "")]),
            "inverted_p_count": len([t for t in all_troughs if t.get("is_inverted")]),
        }
    else:
        agg_metrics = {
            "heart_rate_bpm": 75.0, "mean_rr_ms": 800.0, "std_rr_ms": 40.0,
            "rr_cv": 0.05, "mean_pr_ms": 160.0, "mean_qrs_ms": 90.0,
            "n_crests": 0, "n_troughs": 0, "n_anomalies": 0,
        }

    # 4. CNN PRIMARY PREDICTION
    cnn_class, cnn_confidence, all_class_probs = None, 0.0, {}
    primary_disorder = None

    if cnn_model is not None and class_names is not None:
        try:
            cnn_class, cnn_confidence, all_class_probs = run_cnn(
                image, x, y, w, h, cnn_model, class_names
            )
            primary_disorder = build_primary_from_cnn(cnn_class, cnn_confidence)
        except Exception as e:
            print(f"[WARN] CNN prediction failed: {e}")

    if primary_disorder is None:
        primary_disorder = {
            "disorder":          "No CNN Model Loaded",
            "full_class_name":   "",
            "category":          "N/A",
            "confidence_pct":    0.0,
            "color_hex":         "#90A4AE",
            "color_bgr":         (174, 164, 144),
            "causes":            ["Load model.keras to enable CNN classification"],
            "matching_criteria": ["Waveform bounding boxes only"],
            "source":            "none",
        }

    # 5. Render annotated ECG image with bounding boxes
    annotated_img = annotate_ecg_image(
        image=image,
        lead_boxes=lead_boxes,
        all_fids_and_anomalies=all_fids,
        primary_disorder_info=primary_disorder,
        metrics=agg_metrics,
        show_crests=True,
        show_troughs=True,
        show_anomalies=True,
        show_lead_boxes=True,
        show_legend=True,
    )

    return {
        "primary_disorder": primary_disorder,
        "cnn_class":        cnn_class,
        "cnn_confidence":   cnn_confidence,
        "all_class_probs":  all_class_probs,
        "metrics":          agg_metrics,
        "counts": {
            "total_crests":    len(all_crests),
            "total_troughs":   len(all_troughs),
            "total_anomalies": len(all_anomalies),
            "r_peaks":         agg_metrics.get("r_peak_count", 0),
            "p_waves":         agg_metrics.get("p_wave_count", 0),
            "inverted_waves":  agg_metrics.get("inverted_p_count", 0),
            "leads_detected":  len(lead_boxes),
        },
        "all_crests":      all_crests,
        "all_troughs":     all_troughs,
        "all_anomalies":   all_anomalies,
        "lead_boxes":      lead_boxes,
        "annotated_image": annotated_img,
        "image_shape":     (h_img, w_img),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ECG Bounding Box Detector + 15-Class CNN Disorder Classifier"
    )
    parser.add_argument("--image",   required=True, help="Path to ECG image file.")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cnn_model, class_names = None, None
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
        try:
            import tensorflow as tf
            cnn_model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                class_names = [ln.strip() for ln in f if ln.strip()]
            print(f"[+] CNN model loaded ({len(class_names)} classes)")
        except Exception as e:
            print(f"[WARN] Could not load CNN model: {e}")

    print(f"\n=======================================================")
    print(f"  ECG BOUNDING BOX + 15-CLASS CNN DISORDER CLASSIFIER")
    print(f"=======================================================")
    print(f"Image: {args.image}")

    result  = analyze_ecg_image(args.image, cnn_model=cnn_model, class_names=class_names)
    primary = result["primary_disorder"]
    metrics = result["metrics"]
    counts  = result["counts"]

    print("\n---------------- PRIMARY CNN PREDICTION ----------------")
    print(f"Disorder     : {primary['disorder']} ({primary['category']})")
    print(f"CNN Class    : {result['cnn_class']}")
    print(f"Confidence   : {primary['confidence_pct']:.1f}%")
    print(f"Color Code   : {primary['color_hex']}")

    print("\nCardiac Waveform Metrics (from signal analysis):")
    print(f"  Heart Rate   : {metrics.get('heart_rate_bpm', 0):.0f} BPM")
    print(f"  R-R CV       : {metrics.get('rr_cv', 0)*100:.1f}%")
    print(f"  PR Interval  : {metrics.get('mean_pr_ms', 0):.0f} ms")
    print(f"  QRS Duration : {metrics.get('mean_qrs_ms', 0):.0f} ms")

    print("\nBounding Boxes Generated:")
    print(f"  Crests (Peaks)   : {counts['total_crests']} (R:{counts['r_peaks']} P:{counts['p_waves']})")
    print(f"  Troughs (Valleys): {counts['total_troughs']}")
    print(f"  Anomaly Regions  : {counts['total_anomalies']}")
    print(f"  Leads Tracked    : {counts['leads_detected']}")

    if result["all_class_probs"]:
        print("\nTop-5 CNN Class Probabilities:")
        sorted_probs = sorted(result["all_class_probs"].items(), key=lambda kv: -kv[1])[:5]
        for cls, prob in sorted_probs:
            short = CNN_CLASS_INFO.get(cls, {}).get("short", cls[:40])
            print(f"  {prob*100:>6.2f}%  {short}")

    print("\nPotential Clinical Causes:")
    for cause in primary.get("causes", []):
        print(f"  - {cause}")

    # Save annotated image
    out_img = os.path.join(args.out_dir, "prediction.jpg")
    cv2.imwrite(out_img, result["annotated_image"])
    print(f"\n[+] Annotated image saved: {out_img}")

    # Save text report
    out_rpt = os.path.join(args.out_dir, "report.txt")
    with open(out_rpt, "w", encoding="utf-8") as f:
        f.write("ECG CNN + BOUNDING BOX ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Primary Disorder : {primary['disorder']}\n")
        f.write(f"Category         : {primary['category']}\n")
        f.write(f"CNN Confidence   : {primary['confidence_pct']:.1f}%\n")
        f.write(f"Color Code       : {primary['color_hex']}\n\n")
        f.write("Cardiac Metrics:\n")
        f.write(f"  Heart Rate   : {metrics.get('heart_rate_bpm',0):.0f} BPM\n")
        f.write(f"  PR Interval  : {metrics.get('mean_pr_ms',0):.0f} ms\n")
        f.write(f"  QRS Duration : {metrics.get('mean_qrs_ms',0):.0f} ms\n\n")
        if result["all_class_probs"]:
            f.write("Top-5 CNN Probabilities:\n")
            for cls, prob in sorted(result["all_class_probs"].items(), key=lambda kv: -kv[1])[:5]:
                short = CNN_CLASS_INFO.get(cls, {}).get("short", cls[:40])
                f.write(f"  {prob*100:>6.2f}%  {short}\n")
        f.write("\nClinical Causes:\n")
        for c in primary.get("causes", []):
            f.write(f"  - {c}\n")
        f.write("\nBounding Boxes:\n")
        f.write(f"  Crests   : {counts['total_crests']}\n")
        f.write(f"  Troughs  : {counts['total_troughs']}\n")
        f.write(f"  Anomalies: {counts['total_anomalies']}\n")
    print(f"[+] Report saved: {out_rpt}")

    # Save JSON
    def jconv(o):
        if isinstance(o, (np.integer, np.int64, np.int32)): return int(o)
        if isinstance(o, (np.floating, np.float32, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    out_json = os.path.join(args.out_dir, "analysis_data.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "primary_disorder": primary["disorder"],
            "cnn_class":        result["cnn_class"],
            "confidence_pct":   primary["confidence_pct"],
            "color_hex":        primary["color_hex"],
            "all_class_probs":  result["all_class_probs"],
            "metrics":          metrics,
            "counts":           counts,
            "anomalies":        result["all_anomalies"],
        }, f, indent=2, default=jconv)
    print(f"[+] JSON saved: {out_json}\n")


if __name__ == "__main__":
    main()
