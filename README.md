# CardioDigitize AI: 12-Lead ECG Digitizer, Bounding-Box Visualizer & 15-Disorder Classifier Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/TensorFlow%20%7C%20PyTorch-MobileNetV2%20%2B%20U--Net-orange.svg)](https://github.com/Ajay-Varsan/ECG-Digitizer-Classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end clinical AI application that bridges computer vision, signal processing, and deep neural networks to digitize raw 12-lead paper ECG reports, detect waveform fiducials and clinical anomalies with interactive bounding boxes, and classify conditions across 15 cardiac disorders.

---

## 🫀 Key Capabilities

1. **Adaptive 12-Lead Isolation Engine**:
   - Intelligently detects whether an ECG is a standard 3x4 layout (3 rows) or has a 4th-row rhythm strip.
   - Eliminates row misalignments and accurately places bounding boxes for all 12 clinical leads (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`).
2. **Interactive Clinical Bounding-Box Visualizer**:
   - **Lead Bounding Boxes**: Highlights individual lead boundaries with badges.
   - **Fiducial Bounding Boxes**: Automatically delineates individual **Crests (P, R, T peaks)** and **Troughs (Q, S valleys)**.
   - **Anomaly Bounding Boxes**: Color-coded bounding boxes highlighting morphological abnormalities (PVCs, STEMI ST-elevations, ST-depressions, prolonged PR, pathological T-wave inversions).
   - Real-time layer toggle checkboxes (Leads, Anomalies, Crests, Troughs).
3. **High-Fidelity 1D Time-Series Extraction**:
   - Advanced dark-trace separation suppressing background red/pink grid lines and vertical dividers.
   - Isoelectric baseline estimation via histogram mode calibration to preserve true P-Q-R-S-T waveforms.
   - Export full calibrated digital time-series to CSV (`Time_s`, microvolt potential values).
4. **Synchronized 12-Lead Interactive Waveform Studio**:
   - Plotly interactive waveform traces with zoom, pan, hover tooltips, and lead filtering.
   - **In-Trace Anomaly Shading & Markers**: Anomalous regions are shaded and tagged with diagnostic badges directly on the waveforms.
   - Detailed anomaly breakdown table summarizing lead, timestamp, duration/amplitude, and pathology note.
5. **15-Disorder Deep Learning Classifier (MobileNetV2)**:
   - Softmax probability distribution ranking across 15 disorders.
   - Clinical pathophysiology, causes, and diagnostic medical report export (.txt).

---

## 🏗️ Architecture & Pipeline Flow

```
[Raw ECG Image / Camera Photo / Paper Scan]
                     │
                     ▼
       [Adaptive Layout & Preprocessing]
         ├─ Pink/Red Grid & Lead Region Detection
         ├─ Dynamic Row Detection (3-Row vs 4-Row Rhythm Strip)
         └─ Pixel-Accurate 12-Lead Bounding Boxes
                     │
                     ▼
       [High-Fidelity 1D Signal Extraction]
         ├─ Vertical Divider Line & Grid Speck Suppression
         ├─ Mode-Based Isoelectric Baseline Calibration
         └─ 12-Lead Calibrated Time-Series Table (CSV Export)
                     │
                     ▼
       [Fiducial & Anomaly Detection Engine]
         ├─ P, Q, R, S, T Wave Detection (Crests & Troughs)
         ├─ ST-Segment Elevation / Depression (STEMI / Ischemia)
         ├─ Wide QRS Complexes (PVCs), Prolonged PR (AV Blocks)
         └─ Multi-Layer Bounding Box Rendering with Toggles
                     │
                     ▼
       [15-Disorder MobileNetV2 Deep CNN & Rhythm Telemetry]
         ├─ 15-Class Probability Distribution
         ├─ Heart Rate (BPM), RR mean & variability, PR & QRS intervals
         ├─ Pathophysiological Etiology & Top Differentials
         └─ Automated Diagnostic Medical Report Export (.txt)
```

---

## 📋 The 15 Disorder Classes

1. **Normal Sinus Rhythm**
2. **Myocardial Infarction**
3. **History of Myocardial Infarction**
4. **Abnormal Heartbeat**
5. **Atrial Fibrillation**
6. **Atrial Flutter**
7. **Sinus Tachycardia**
8. **Sinus Bradycardia**
9. **Complete Left Bundle Branch Block (LBBB)**
10. **Complete Right Bundle Branch Block (RBBB)**
11. **Incomplete Right Bundle Branch Block (IRBBB)**
12. **First Degree AV Block**
13. **Left Anterior Fascicular Block (LAFB)**
14. **Left Ventricular Hypertrophy (LVH)**
15. **Left Atrial Enlargement (LAE)**

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Ajay-Varsan/ECG-Digitizer-Classifier.git
cd ECG-Digitizer-Classifier
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the Application Studio
```bash
# Using Python
streamlit run app_combined.py

# Or on Windows using launcher script:
run_app.bat
# or in PowerShell:
.\run_app.ps1
```

Open your browser at **`http://localhost:8501`**.

---

## 🖥️ Studio Tabs & Features

- **Tab 1: 🖼️ 12-Lead Isolated Grid & Bounding Boxes**:
  - Live interactive checkboxes for toggling lead boxes, anomaly boxes, crests, and troughs.
  - One-click **Download Annotated 12-Lead ECG with Bounding Boxes (JPG)**.
- **Tab 2: 📋 Time-Series Table (Output)**:
  - Microvolt potential values per lead against continuous time.
  - One-click **Download Complete Time Series (CSV)**.
  - Telemetry statistics table (mean, std, min, max, peak-to-peak).
- **Tab 3: 📈 Interactive 12-Lead Studio**:
  - Stacked Plotly waveforms with shaded anomaly zones and diagnostic badges.
  - Breakdown table of detected anomalies per lead.
- **Tab 4: 🩺 15-Disorder Clinical Diagnosis**:
  - Ranked probability bar chart and clinical causes card.
- **Tab 5: 📄 Diagnostic Medical Report**:
  - Synthesized cardiology summary report with downloadable text format.

---

## 📄 License
This project is licensed under the MIT License.
