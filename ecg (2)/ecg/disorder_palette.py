"""
disorder_palette.py
-------------------
Defines unique, visually distinct color codes (RGB, BGR, HEX) for all 18+
ECG arrhythmia disorders, individual crest/trough wave types (P, Q, R, S, T),
and provides helper utilities for color conversion and legend rendering.
"""

# Unique color codes for each Disorder (HEX and BGR for OpenCV)
# Format: "DISORDER_NAME": {"hex": "#RRGGBB", "bgr": (B, G, R), "rgb": (R, G, B), "category": "..."}
DISORDER_COLORS = {
    "Normal Sinus Rhythm": {
        "hex": "#00E676",
        "bgr": (118, 230, 0),
        "rgb": (0, 230, 118),
        "short": "NSR",
        "category": "Normal",
        "desc": "Regular rate 60-100 bpm, normal P waves preceding each QRS complex.",
    },
    "Sinus Arrhythmia": {
        "hex": "#00E5FF",
        "bgr": (255, 229, 0),
        "rgb": (0, 229, 255),
        "short": "Sinus Arrhythmia",
        "category": "Sinus",
        "desc": "Irregular atrial and ventricular rhythms. Normal P wave preceding each QRS complex.",
    },
    "Sinus Tachycardia": {
        "hex": "#FF3D00",
        "bgr": (0, 61, 255),
        "rgb": (255, 61, 0),
        "short": "Sinus Tachycardia",
        "category": "Sinus",
        "desc": "Regular rhythm, rate > 100 bpm, normal P wave preceding each QRS complex.",
    },
    "Sinus Bradycardia": {
        "hex": "#2979FF",
        "bgr": (255, 121, 41),
        "rgb": (41, 121, 255),
        "short": "Sinus Bradycardia",
        "category": "Sinus",
        "desc": "Regular rhythm, rate < 60 bpm, normal P wave preceding each QRS complex.",
    },
    "Sinoatrial (SA) Arrest or Block": {
        "hex": "#AA00FF",
        "bgr": (255, 0, 170),
        "rgb": (170, 0, 255),
        "short": "SA Arrest / Block",
        "category": "Sinus",
        "desc": "Normal rhythm except for missing P-QRS-T complexes. Pause not equal to multiple of previous rhythm.",
    },
    "Wandering Atrial Pacemaker": {
        "hex": "#00BFA5",
        "bgr": (165, 191, 0),
        "rgb": (0, 191, 165),
        "short": "Wandering Pacemaker",
        "category": "Atrial",
        "desc": "Atrial/ventricular rhythms vary slightly. Irregular PR interval, P waves with changing configurations.",
    },
    "Premature Atrial Contraction (PAC)": {
        "hex": "#76FF03",
        "bgr": (3, 255, 118),
        "rgb": (118, 255, 3),
        "short": "PAC",
        "category": "Atrial",
        "desc": "Premature abnormal-looking P waves differing in configuration from normal P waves.",
    },
    "Paroxysmal Supraventricular Tachycardia (PSVT)": {
        "hex": "#FF9100",
        "bgr": (0, 145, 255),
        "rgb": (255, 145, 0),
        "short": "PSVT",
        "category": "Atrial",
        "desc": "Regular rhythm, rate > 160 bpm (up to 250 bpm). Aberrant/hidden P waves, sudden onset.",
    },
    "Atrial Flutter": {
        "hex": "#FFD600",
        "bgr": (0, 214, 255),
        "rgb": (255, 214, 0),
        "short": "A-Flutter",
        "category": "Atrial",
        "desc": "Atrial rhythm regular 250-400 bpm. Characteristic saw-tooth shape P/F-waves.",
    },
    "Atrial Fibrillation": {
        "hex": "#FF1744",
        "bgr": (68, 23, 255),
        "rgb": (255, 23, 68),
        "short": "A-Fib",
        "category": "Atrial",
        "desc": "Grossly irregular rhythm, atrial rate 300-600 bpm. No distinct P waves, erratic fibrillatory baseline.",
    },
    "Junctional Rhythm": {
        "hex": "#78909C",
        "bgr": (156, 144, 120),
        "rgb": (120, 144, 156),
        "short": "Junctional Rhythm",
        "category": "Junctional",
        "desc": "Regular rate 40-60 bpm. P waves inverted, hidden within, or after QRS complex; PR < 0.12s.",
    },
    "Premature Junctional Contraction (PJC)": {
        "hex": "#F48FB1",
        "bgr": (177, 143, 244),
        "rgb": (244, 143, 177),
        "short": "PJC",
        "category": "Junctional",
        "desc": "Irregular rhythm from premature beats with inverted/hidden P waves and normal QRS.",
    },
    "First-Degree AV Block": {
        "hex": "#FF6E40",
        "bgr": (64, 110, 255),
        "rgb": (255, 110, 64),
        "short": "1st-Deg AV Block",
        "category": "AV Block",
        "desc": "Regular rhythm, prolonged constant PR interval > 0.20 second, normal QRS.",
    },
    "Second-Degree AV Block Mobitz I (Wenckebach)": {
        "hex": "#D50000",
        "bgr": (0, 0, 213),
        "rgb": (213, 0, 0),
        "short": "Mobitz I (Wenckebach)",
        "category": "AV Block",
        "desc": "PR interval progressively lengthens with each cycle until a QRS complex drops.",
    },
    "Third-Degree AV Block (Complete Heart Block)": {
        "hex": "#880E4F",
        "bgr": (79, 14, 136),
        "rgb": (136, 14, 79),
        "short": "3rd-Deg AV Block",
        "category": "AV Block",
        "desc": "Regular atrial rhythm independent of slow ventricular rhythm. No constant PR interval.",
    },
    "Premature Ventricular Contraction (PVC)": {
        "hex": "#D81B60",
        "bgr": (96, 27, 216),
        "rgb": (216, 27, 96),
        "short": "PVC",
        "category": "Ventricular",
        "desc": "Premature, wide and distorted QRS (>0.14s), followed by full compensatory pause.",
    },
    "Ventricular Tachycardia": {
        "hex": "#C2185B",
        "bgr": (91, 24, 194),
        "rgb": (194, 24, 91),
        "short": "V-Tach",
        "category": "Ventricular",
        "desc": "Ventricular rate 140-220 bpm. Wide, bizarre QRS complexes independent of P waves.",
    },
    "Ventricular Fibrillation": {
        "hex": "#4A148C",
        "bgr": (140, 20, 74),
        "rgb": (74, 20, 140),
        "short": "V-Fib",
        "category": "Ventricular",
        "desc": "Rapid, chaotic, irregular baseline without distinct QRS complexes or P waves.",
    },
    "Asystole": {
        "hex": "#616161",
        "bgr": (97, 97, 97),
        "rgb": (97, 97, 97),
        "short": "Asystole",
        "category": "Lethal",
        "desc": "Absence of electrical activity / flatline. No discernible P, QRS, or T waves.",
    },
    "Myocardial Infarction / Ischemia": {
        "hex": "#E65100",
        "bgr": (0, 81, 230),
        "rgb": (230, 81, 0),
        "short": "MI / Ischemia",
        "category": "Ischemia",
        "desc": "ST-segment elevation or depression, pathological Q waves, or T-wave inversion.",
    },
}

# Unique Colors for Individual Crests and Troughs
# Format: "WAVE_TYPE": {"bgr": (B, G, R), "hex": "#RRGGBB", "label": "..."}
WAVE_FIDUCIAL_COLORS = {
    # Crests (Peaks / Local Maxima)
    "R_PEAK": {
        "bgr": (0, 255, 0),       # Vibrant Green
        "hex": "#00FF00",
        "rgb": (0, 255, 0),
        "label": "R-Crest (Peak)",
        "kind": "crest",
    },
    "P_PEAK": {
        "bgr": (255, 191, 0),     # Deep Sky Blue
        "hex": "#00BFFF",
        "rgb": (0, 191, 255),
        "label": "P-Crest",
        "kind": "crest",
    },
    "T_PEAK": {
        "bgr": (255, 0, 200),     # Magenta / Violet
        "hex": "#C800FF",
        "rgb": (200, 0, 255),
        "label": "T-Crest",
        "kind": "crest",
    },
    "FLUTTER_CREST": {
        "bgr": (0, 215, 255),     # Gold
        "hex": "#FFD700",
        "rgb": (255, 215, 0),
        "label": "Flutter-Crest (F-Wave)",
        "kind": "crest",
    },
    # Troughs (Valleys / Local Minima)
    "Q_TROUGH": {
        "bgr": (255, 255, 0),     # Cyan
        "hex": "#00FFFF",
        "rgb": (0, 255, 255),
        "label": "Q-Trough",
        "kind": "trough",
    },
    "S_TROUGH": {
        "bgr": (0, 165, 255),     # Orange
        "hex": "#FFA500",
        "rgb": (255, 165, 0),
        "label": "S-Trough",
        "kind": "trough",
    },
    "INVERTED_P_TROUGH": {
        "bgr": (180, 105, 255),   # Hot Pink
        "hex": "#FF69B4",
        "rgb": (255, 105, 180),
        "label": "Inverted P-Trough",
        "kind": "trough",
    },
    "INVERTED_T_TROUGH": {
        "bgr": (205, 90, 106),    # Slate Blue
        "hex": "#6A5ACD",
        "rgb": (106, 90, 205),
        "label": "Inverted T-Trough",
        "kind": "trough",
    },
    "FIB_MICRO_CREST": {
        "bgr": (128, 0, 255),     # Electric Violet
        "hex": "#FF0080",
        "rgb": (255, 0, 128),
        "label": "Fib Micro-Crest (f-wave)",
        "kind": "crest",
    },
    "FIB_MICRO_TROUGH": {
        "bgr": (147, 20, 255),    # Deep Pink
        "hex": "#FF1493",
        "rgb": (255, 20, 147),
        "label": "Fib Micro-Trough",
        "kind": "trough",
    },
    # Anomalous Complex Segments
    "ANOMALY_PVC_COMPLEX": {
        "bgr": (0, 0, 255),       # Bright Red
        "hex": "#FF0000",
        "rgb": (255, 0, 0),
        "label": "Anomalous Wide QRS (PVC/Ectopic)",
        "kind": "anomaly",
    },
    "ANOMALY_PROLONGED_PR": {
        "bgr": (0, 140, 255),     # Dark Orange
        "hex": "#FF8C00",
        "rgb": (255, 140, 0),
        "label": "Prolonged PR Segment",
        "kind": "anomaly",
    },
    "ANOMALY_PAUSE_BLOCK": {
        "bgr": (255, 0, 128),     # Violet Red
        "hex": "#8000FF",
        "rgb": (128, 0, 255),
        "label": "Rhythm Pause / Dropped Beat",
        "kind": "anomaly",
    },
    "ANOMALY_ST_ELEVATION": {
        "bgr": (0, 69, 255),      # Red-Orange
        "hex": "#FF4500",
        "rgb": (255, 69, 0),
        "label": "ST Elevation / Depression (Ischemia)",
        "kind": "anomaly",
    },
}


def get_disorder_color(disorder_name, color_format="bgr"):
    """
    Look up color for a disorder name (case-insensitive fuzzy match).
    Returns (B, G, R) tuple or hex string.
    """
    for name, data in DISORDER_COLORS.items():
        if disorder_name.lower() in name.lower() or name.lower() in disorder_name.lower():
            return data.get(color_format.lower(), data["bgr"])
    # Default fallback
    if color_format.lower() == "hex":
        return "#FFD700"
    elif color_format.lower() == "rgb":
        return (255, 215, 0)
    return (0, 215, 255)


def get_fiducial_color(wave_name, color_format="bgr"):
    """Look up color for a wave fiducial type (R_PEAK, P_PEAK, Q_TROUGH, etc.)."""
    data = WAVE_FIDUCIAL_COLORS.get(wave_name.upper())
    if data is not None:
        return data.get(color_format.lower(), data["bgr"])
    # Default fallback
    if color_format.lower() == "hex":
        return "#00FFFF"
    return (255, 255, 0)
