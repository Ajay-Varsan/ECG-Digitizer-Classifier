"""
arrhythmia_classifier.py
------------------------
Clinical Pattern Alignment & Multi-Arrhythmia Classification Engine.
Evaluates detected waveform metrics, crests, troughs, and intervals against
established clinical criteria for 18+ arrhythmia disorders from the clinical
guidelines, ranks disorder alignment, assigns confidence scores, and attaches
the corresponding clinical causes and unique color codes (excluding treatments).
"""

import numpy as np
from disorder_palette import DISORDER_COLORS, get_disorder_color

# Comprehensive Clinical Knowledge Base of Arrhythmia Criteria and Causes
ARRHYTHMIA_KNOWLEDGE_BASE = {
    "Normal Sinus Rhythm": {
        "category": "Normal",
        "hr_range": (60, 100),
        "rr_regularity": "Regular (CV < 0.10)",
        "p_wave": "Normal upright P wave preceding each QRS complex",
        "pr_interval": "Normal (120 - 200 ms)",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Normal resting cardiac conduction",
            "Adequate autonomic tone",
            "Absence of acute ischemia or electrolyte disturbances",
        ],
    },
    "Sinus Arrhythmia": {
        "category": "Sinus",
        "hr_range": (45, 100),
        "rr_regularity": "Irregular atrial and ventricular rhythms (CV >= 0.12)",
        "p_wave": "Normal P wave preceding each QRS complex",
        "pr_interval": "Normal (120 - 200 ms)",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Normal variation of sinus rhythm in athletes, children, and the elderly (respiratory sinus arrhythmia)",
            "Digoxin toxicity",
            "Inferior wall myocardial infarction",
            "Increased vagal tone",
        ],
    },
    "Sinus Tachycardia": {
        "category": "Sinus",
        "hr_range": (101, 160),
        "rr_regularity": "Regular (CV < 0.10)",
        "p_wave": "Normal P wave preceding each QRS complex",
        "pr_interval": "Normal (120 - 200 ms)",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Normal physiologic response to fever, exercise, anxiety, dehydration, or pain",
            "Hypovolemic or cardiogenic shock, left-sided heart failure, cardiac tamponade",
            "Hyperthyroidism and severe anemia",
            "Stimulants (epinephrine, atropine, caffeine, nicotine, alcohol)",
        ],
    },
    "Sinus Bradycardia": {
        "category": "Sinus",
        "hr_range": (20, 59),
        "rr_regularity": "Regular (CV < 0.10)",
        "p_wave": "Normal P wave preceding each QRS complex",
        "pr_interval": "Normal (120 - 200 ms)",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Physiologic in well-conditioned athletes and during sleep",
            "Increased intracranial pressure (Cushing's triad)",
            "Increased vagal tone (straining, vomiting, intubation, mechanical ventilation)",
            "Hypothermia, hypothyroidism, sick sinus syndrome",
            "Medications (beta-blockers, calcium channel blockers, digoxin)",
        ],
    },
    "Sinoatrial (SA) Arrest or Block": {
        "category": "Sinus",
        "hr_range": (0, 100),
        "rr_regularity": "Rhythm normal except for missing complex / sudden pause",
        "p_wave": "Normal P wave when present; absent during arrest pause",
        "pr_interval": "Normal when present",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Infection and acute myocarditis",
            "Coronary artery disease, degenerative conduction disease, acute inferior wall MI",
            "Excessive vagal stimulation, Valsalva maneuver, carotid sinus massage",
            "Digoxin, quinidine, or beta-blocker toxicity",
        ],
    },
    "Wandering Atrial Pacemaker": {
        "category": "Atrial",
        "hr_range": (50, 100),
        "rr_regularity": "Atrial and ventricular rhythms vary slightly",
        "p_wave": "P waves irregular with >= 3 changing configurations (upright, flat, inverted, biphasic)",
        "pr_interval": "Irregular / varying PR interval",
        "qrs_duration": "Uniform shape and normal duration",
        "causes": [
            "Rheumatic carditis due to inflammation involving SA node",
            "Digoxin toxicity",
            "Sick sinus syndrome",
            "Chronic obstructive pulmonary disease (COPD)",
        ],
    },
    "Premature Atrial Contraction (PAC)": {
        "category": "Atrial",
        "hr_range": (60, 120),
        "rr_regularity": "Underlying regular rhythm interrupted by premature ectopic beats",
        "p_wave": "Premature, abnormal-looking P wave differing from sinus P wave",
        "pr_interval": "Variable (often shortened or lengthened)",
        "qrs_duration": "Normal (< 120 ms), followed by non-compensatory pause",
        "causes": [
            "Stimulants (caffeine, nicotine, alcohol, amphetamines)",
            "Hyperthyroidism and acute emotional stress",
            "Chronic obstructive pulmonary disease (COPD) and hypoxemia",
            "Ischemic heart disease, valvular disease, infection",
        ],
    },
    "Paroxysmal Supraventricular Tachycardia (PSVT)": {
        "category": "Atrial",
        "hr_range": (161, 250),
        "rr_regularity": "Extremely regular, sudden onset and abrupt termination",
        "p_wave": "P waves regular but aberrant / hidden within preceding T wave",
        "pr_interval": "Difficult to measure / shortened",
        "qrs_duration": "Narrow and normal (< 120 ms)",
        "causes": [
            "Physical exertion, acute emotional stress, stimulants",
            "Intrinsic abnormality of AV conduction system (AVNRT / AVRT / WPW)",
            "Rheumatic heart disease and mitral valve prolapse",
            "Digoxin toxicity, caffeine, marijuana, CNS stimulants",
        ],
    },
    "Atrial Flutter": {
        "category": "Atrial",
        "hr_range": (60, 180),
        "rr_regularity": "Regular or regularly irregular (fixed 2:1, 3:1, 4:1 AV conduction block)",
        "p_wave": "Characteristic continuous saw-tooth (F-wave) baseline (atrial rate 250-400 bpm)",
        "pr_interval": "Not measurable",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Heart failure, tricuspid or mitral valve disease",
            "Pulmonary embolism, cor pulmonale, COPD",
            "Acute inferior wall MI, pericarditis, myocarditis",
            "Digoxin toxicity, post-cardiac surgery",
        ],
    },
    "Atrial Fibrillation": {
        "category": "Atrial",
        "hr_range": (60, 180),
        "rr_regularity": "Grossly irregular / irregularly irregular (high RR-CV > 0.20)",
        "p_wave": "No discernible P waves; chaotic erratic fibrillatory (f-wave) baseline (300-600 bpm)",
        "pr_interval": "Indiscernible",
        "qrs_duration": "Normal (< 120 ms) unless aberrant conduction present",
        "causes": [
            "Hypertension and hypertensive heart disease",
            "Heart failure, ischemic heart disease, coronary artery disease",
            "COPD, pulmonary embolism, thyrotoxicosis / hyperthyroidism",
            "Valvular heart disease (mitral stenosis), constrictive pericarditis, sepsis",
            "Post-coronary bypass or valve replacement surgery",
        ],
    },
    "Junctional Rhythm": {
        "category": "Junctional",
        "hr_range": (40, 60),
        "rr_regularity": "Regular atrial and ventricular rhythm",
        "p_wave": "P waves inverted preceding QRS, hidden within QRS, or retrograde following QRS",
        "pr_interval": "Shortened (< 120 ms) when preceding QRS",
        "qrs_duration": "Normal configuration and duration (< 120 ms)",
        "causes": [
            "Inferior wall MI or acute myocardial ischemia",
            "Hypoxia and excessive vagal stimulation",
            "Sick sinus syndrome and SA node dysfunction",
            "Acute rheumatic fever, post-valve surgery, digoxin toxicity",
        ],
    },
    "Premature Junctional Contraction (PJC)": {
        "category": "Junctional",
        "hr_range": (50, 90),
        "rr_regularity": "Irregular rhythm due to isolated premature beats",
        "p_wave": "Inverted P wave preceding, hidden within, or following premature QRS",
        "pr_interval": "Short (< 120 ms) when visible",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Myocardial infarction or acute ischemia",
            "Digoxin toxicity",
            "Excessive caffeine, tobacco, amphetamines",
            "Heart failure and autonomic imbalance",
        ],
    },
    "First-Degree AV Block": {
        "category": "AV Block",
        "hr_range": (50, 100),
        "rr_regularity": "Regular rhythm",
        "p_wave": "Normal P wave preceding each QRS complex (1:1 ratio)",
        "pr_interval": "Prolonged constant PR interval > 0.20 second (> 200 ms)",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Inferior wall MI, myocardial ischemia, or acute myocarditis",
            "Hypothyroidism, hypokalemia, hyperkalemia",
            "Digoxin toxicity",
            "Medications: quinidine, procainamide, beta-blockers, calcium channel blockers",
        ],
    },
    "Second-Degree AV Block Mobitz I (Wenckebach)": {
        "category": "AV Block",
        "hr_range": (40, 80),
        "rr_regularity": "Irregular ventricular rhythm (grouped beating)",
        "p_wave": "Normal sinus P waves; atrial rate exceeds ventricular rate",
        "pr_interval": "Progressively longer PR interval with each cycle until a QRS drops",
        "qrs_duration": "Normal (< 120 ms)",
        "causes": [
            "Severe coronary artery disease, anterior or inferior wall MI",
            "Acute myocarditis, rheumatic heart disease",
            "Digoxin toxicity, beta-blocker or calcium channel blocker excess",
            "Increased vagal tone",
        ],
    },
    "Third-Degree AV Block (Complete Heart Block)": {
        "category": "AV Block",
        "hr_range": (20, 50),
        "rr_regularity": "Regular atrial rate and regular slow ventricular escape rate (AV dissociation)",
        "p_wave": "Normal sinus P waves with no constant relation to QRS complexes",
        "pr_interval": "No constant PR interval (complete AV dissociation)",
        "qrs_duration": "Normal if junctional escape; wide and bizarre (> 140 ms) if ventricular escape",
        "causes": [
            "Inferior or anterior wall myocardial infarction",
            "Degenerative conduction system disease (Lenegre-Lev disease)",
            "Congenital heart block, rheumatic fever, Lyme disease",
            "Severe electrolyte imbalances, calcific aortic stenosis",
        ],
    },
    "Premature Ventricular Contraction (PVC)": {
        "category": "Ventricular",
        "hr_range": (50, 110),
        "rr_regularity": "Irregular rhythm due to premature wide complexes and compensatory pauses",
        "p_wave": "Absent or unrelated to the premature wide QRS",
        "pr_interval": "Absent for ectopic beats",
        "qrs_duration": "Wide and bizarre (> 130 - 140 ms), discordance with T wave",
        "causes": [
            "Heart failure, acute or previous myocardial ischemia / infarction",
            "Hypokalemia, hypomagnesemia, hypocalcemia, hypercapnia",
            "Drug toxicity (cardiac glycosides/digoxin, tricyclic antidepressants, aminophylline)",
            "Stimulants (caffeine, tobacco, alcohol, amphetamines)",
            "Psychological stress, anxiety, hypoxia",
        ],
    },
    "Ventricular Tachycardia": {
        "category": "Ventricular",
        "hr_range": (140, 220),
        "rr_regularity": "Regular or slightly irregular",
        "p_wave": "Not discernible or dissociated from QRS complexes",
        "pr_interval": "Absent",
        "qrs_duration": "Wide, bizarre, distorted QRS complexes (> 140 ms)",
        "causes": [
            "Acute myocardial ischemia, infarction, or ventricular aneurysm",
            "Coronary artery disease, dilated or hypertrophic cardiomyopathy",
            "Mitral valve prolapse, heart failure",
            "Electrolyte disturbances (severe hypokalemia, hypercalcemia)",
            "Pulmonary embolism, digoxin / quinidine / procainamide / epinephrine toxicity",
        ],
    },
    "Ventricular Fibrillation": {
        "category": "Ventricular",
        "hr_range": (0, 0),
        "rr_regularity": "Rapid, chaotic, completely irregular",
        "p_wave": "No discernible P waves",
        "pr_interval": "Absent",
        "qrs_duration": "No discernible QRS complexes; erratic fibrillatory undulations",
        "causes": [
            "Acute myocardial ischemia or massive infarction",
            "R-on-T phenomenon, untreated ventricular tachycardia",
            "Severe hypokalemia, hyperkalemia, hypocalcemia, severe acidosis, hypothermia",
            "Electric shock, lightning strike",
            "Digoxin, epinephrine, or quinidine toxicity",
        ],
    },
    "Asystole": {
        "category": "Lethal",
        "hr_range": (0, 0),
        "rr_regularity": "No electrical rhythm (flatline)",
        "p_wave": "No discernible P waves",
        "pr_interval": "Absent",
        "qrs_duration": "No discernible QRS complexes or T waves",
        "causes": [
            "End-stage myocardial ischemia or massive cardiac rupture",
            "Severe hypoxia, profound acidosis, hyperkalemia, hypokalemia",
            "Electric shock, severe hypothermia",
            "Massive pulmonary embolism, cardiac tamponade",
            "Cocaine overdose, prolonged cardiac arrest",
        ],
    },
    "Myocardial Infarction / Ischemia": {
        "category": "Ischemia",
        "hr_range": (50, 120),
        "rr_regularity": "Usually regular sinus rhythm",
        "p_wave": "Normal sinus P waves",
        "pr_interval": "Normal",
        "qrs_duration": "Pathological Q waves (>40ms width, >25% R amplitude) or ST elevation/depression",
        "causes": [
            "Acute coronary artery occlusion / plaque rupture / thrombosis",
            "Severe coronary spasm or microvascular ischemia",
            "Supply-demand mismatch (shock, severe anemia, severe tachycardia)",
            "Atherosclerotic cardiovascular disease",
        ],
    },
}


def score_pattern_alignment(metrics, all_boxes, cnn_prediction=None):
    """
    Evaluates detected waveform metrics (HR, RR-CV, PR, QRS, wave types, anomalies)
    against each disorder's clinical criteria and ranks alignment.

    Parameters:
        metrics: Aggregated metrics dictionary from wave_detector
        all_boxes: All detected crest, trough, and anomaly bounding boxes
        cnn_prediction: Optional top prediction string from CNN model

    Returns:
        list of sorted disorder alignment dictionaries:
            - 'disorder': Name of disorder
            - 'category': Clinical category
            - 'score': Alignment score [0.0 - 1.0]
            - 'confidence_pct': Formatted confidence percentage
            - 'is_primary': Boolean flag for top match
            - 'color_hex': Unique disorder color hex
            - 'color_bgr': Unique disorder color bgr
            - 'matching_criteria': List of clinical criteria matched
            - 'causes': Clinical causes
            - 'key_parameters': Table of key parameter comparisons
    """
    hr = metrics.get("heart_rate_bpm", 75.0)
    rr_cv = metrics.get("rr_cv", 0.05)
    mean_pr = metrics.get("mean_pr_ms", 160.0)
    mean_qrs = metrics.get("mean_qrs_ms", 90.0)
    n_anomalies = metrics.get("n_anomalies", 0)

    # Count specific fiducial types
    pvc_boxes = [b for b in all_boxes if "PVC" in b.get("name", "") or b.get("type") == "ANOMALY_PVC_COMPLEX"]
    pr_long_boxes = [b for b in all_boxes if b.get("type") == "ANOMALY_PROLONGED_PR"]
    pause_boxes = [b for b in all_boxes if b.get("type") == "ANOMALY_PAUSE_BLOCK"]
    flutter_boxes = [b for b in all_boxes if b.get("type") == "FLUTTER_CREST"]
    inverted_p_boxes = [b for b in all_boxes if b.get("is_inverted") and "P" in b.get("name", "")]

    scores = {}

    for disorder_name, info in ARRHYTHMIA_KNOWLEDGE_BASE.items():
        score = 0.0
        matched = []

        # 1. Heart Rate Alignment
        min_hr, max_hr = info["hr_range"]
        if min_hr == 0 and max_hr == 0:
            # Asystole or VFib
            if hr < 15.0 or metrics.get("r_peak_count", 0) == 0:
                score += 0.40
                matched.append("Absence of organized ventricular rate")
            else:
                score -= 0.30
        elif min_hr <= hr <= max_hr:
            score += 0.30
            matched.append(f"Heart rate {hr:.0f} bpm matches target range ({min_hr}-{max_hr} bpm)")
        else:
            # Partial credit if close
            dist = min(abs(hr - min_hr), abs(hr - max_hr))
            if dist < 15:
                score += 0.15
                matched.append(f"Heart rate {hr:.0f} bpm near boundary ({min_hr}-{max_hr} bpm)")

        # 2. Rhythm Regularity (RR-CV)
        if "irregular" in info["rr_regularity"].lower():
            if rr_cv >= 0.12 or len(pause_boxes) > 0:
                score += 0.25
                matched.append(f"Irregular R-R rhythm detected (RR-CV: {rr_cv*100:.1f}%)")
            else:
                score += 0.05
        else:
            # Regular rhythm expected
            if rr_cv < 0.12:
                score += 0.25
                matched.append(f"Regular R-R rhythm detected (RR-CV: {rr_cv*100:.1f}%)")
            else:
                score += 0.05

        # 3. Specific Waveform Morphology & Anomaly Signatures
        if disorder_name == "Sinus Tachycardia" and hr > 100 and mean_qrs < 120:
            score += 0.35
            matched.append("Regular tachycardia with normal narrow QRS complexes")

        elif disorder_name == "Sinus Bradycardia" and hr < 60 and mean_qrs < 120:
            score += 0.35
            matched.append("Regular bradycardia with normal narrow QRS complexes")

        elif disorder_name == "Sinus Arrhythmia" and rr_cv >= 0.12 and mean_qrs < 120:
            score += 0.30
            matched.append("Rhythm irregularity with preserved normal P-QRS morphology")

        elif disorder_name == "First-Degree AV Block" and (mean_pr > 200.0 or len(pr_long_boxes) > 0):
            score += 0.45
            matched.append(f"Prolonged PR interval ({mean_pr:.0f} ms > 200 ms)")

        elif disorder_name == "Premature Ventricular Contraction (PVC)" and (len(pvc_boxes) > 0 or mean_qrs > 130):
            score += 0.40
            matched.append(f"Detected {len(pvc_boxes)} wide bizarre / premature QRS complexes")

        elif disorder_name == "Atrial Flutter" and len(flutter_boxes) >= 4:
            score += 0.50
            matched.append(f"Detected {len(flutter_boxes)} characteristic sawtooth F-crests")

        elif disorder_name == "Atrial Fibrillation" and rr_cv >= 0.22:
            score += 0.40
            matched.append(f"High rhythm chaos (RR-CV {rr_cv*100:.1f}%) with erratic baseline")

        elif disorder_name == "Sinoatrial (SA) Arrest or Block" and len(pause_boxes) > 0:
            score += 0.45
            matched.append(f"Detected {len(pause_boxes)} sudden cardiac pause segments")

        elif disorder_name == "Junctional Rhythm" and (len(inverted_p_boxes) > 0 or (40 <= hr <= 60 and mean_pr < 120)):
            score += 0.35
            matched.append("Inverted / retrograde P-wave activity with junctional rate (40-60 bpm)")

        elif disorder_name == "Paroxysmal Supraventricular Tachycardia (PSVT)" and hr >= 160:
            score += 0.40
            matched.append(f"Extreme regular rate ({hr:.0f} bpm >= 160 bpm) with narrow QRS")

        elif disorder_name == "Ventricular Tachycardia" and hr >= 140 and mean_qrs >= 130:
            score += 0.50
            matched.append(f"Wide-complex tachycardia (HR: {hr:.0f} bpm, QRS: {mean_qrs:.0f} ms)")

        elif disorder_name == "Normal Sinus Rhythm":
            if 55 <= hr <= 105 and rr_cv < 0.18 and mean_pr <= 210 and mean_qrs < 120:
                score += 0.40
                matched.append("Cardiac intervals (HR, RR, PR, QRS) within normal clinical boundaries")

        # 4. Integrate CNN Model Prior (if provided)
        if cnn_prediction:
            cnn_lower = cnn_prediction.lower()
            if "myocardial" in cnn_lower or "history of mi" in cnn_lower or "mi" in cnn_lower:
                if disorder_name == "Myocardial Infarction / Ischemia":
                    score += 0.50
                    matched.append("Deep learning CNN detected diagnostic myocardial infarction / ischemic visual pattern")
            elif "normal" in cnn_lower:
                if disorder_name == "Normal Sinus Rhythm":
                    score += 0.50
                    matched.append("Deep learning CNN detected normal sinus rhythm visual pattern")
            elif "abnormal" in cnn_lower:
                if info["category"] in ["Sinus", "Atrial", "Junctional", "AV Block", "Ventricular"]:
                    score += 0.20
                    matched.append("Deep learning CNN identified abnormal rhythm visual features")

        # Clamp score to [0.05, 0.99]
        final_score = float(np.clip(score, 0.05, 0.98))
        scores[disorder_name] = {
            "score": final_score,
            "matched": matched,
        }

    # Sort disorders by alignment score descending
    sorted_disorders = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)

    results = []
    top_score = sorted_disorders[0][1]["score"]

    for rank, (name, data) in enumerate(sorted_disorders):
        is_primary = (rank == 0)
        # Normalize relative confidence
        conf_pct = (data["score"] / max(0.1, sum(d[1]["score"] for d in sorted_disorders[:5]))) * 100.0 if rank < 5 else (data["score"] * 100.0)
        conf_pct = float(np.clip(conf_pct if not is_primary else max(conf_pct, 65.0), 5.0, 98.5))

        color_hex = DISORDER_COLORS.get(name, {}).get("hex", "#FFD700")
        color_bgr = DISORDER_COLORS.get(name, {}).get("bgr", (0, 215, 255))
        causes = ARRHYTHMIA_KNOWLEDGE_BASE.get(name, {}).get("causes", [])
        category = ARRHYTHMIA_KNOWLEDGE_BASE.get(name, {}).get("category", "General")

        results.append({
            "disorder": name,
            "category": category,
            "score": float(data["score"]),
            "confidence_pct": float(conf_pct),
            "is_primary": bool(is_primary),
            "color_hex": color_hex,
            "color_bgr": color_bgr,
            "matching_criteria": data["matched"],
            "causes": causes,
        })

    return results
