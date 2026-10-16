# CardioDigitize AI: Unified 12-Lead ECG Digitizer & 15-Disorder Classifier

This project integrates the **Open ECG Digitizer** and the **15-Disorder Deep ECG Classifier** into an end-to-end clinical AI application.

---

## 🏗️ Architecture & Pipeline Flow

```
[Raw ECG Image / Camera Photo / Paper Scan]
                     │
                     ▼
       [Open ECG Digitizer Pipeline]
         ├─ U-Net Semantic Segmentation
         ├─ Hough & Perspective Angle Detection
         ├─ Paper Perspective Un-warping & Grid Scaling
         ├─ 12-Lead Region Cropping & Layout Identifier
         └─ Trace Extraction (Voltage Conversion in µV)
                     │
                     ▼
       [12-Lead Time Series Table (DataFrame)]
         ├─ Columns: Time_s, I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
         ├─ Missing Value Interpolation (Linear / Nearest / Zero-fill)
         └─ One-Click CSV Export
                     │
                     ▼
       [Calibrated Standard 12-Lead ECG Reconstruction]
         └─ Standard 3x4 Medical Grid Renderer (calibrated µV/mV scaling)
                     │
                     ▼
       [15-Disorder MobileNetV2 Deep CNN & Rhythm Engine]
         ├─ Softmax Probability Distribution across 15 disorders
         ├─ Telemetry: Heart Rate (BPM), RR mean & variability, PR & QRS intervals
         ├─ Top Clinical Causes & Pathophysiological Breakdown
         └─ Comprehensive Medical Diagnostic Report Export (.txt)
```

---

## 💡 How the Dataset Discrepancy Was Solved

1. **The Challenge**:
   - The **Open ECG Digitizer** was trained on Hugging Face data (scanned and photographed paper ECGs with multiple grid types, perspectives, and lighting conditions) to output **raw continuous 12-lead time series signals**.
   - The **15-Disorder ECG Classifier** was trained on 15 disorder categories (4 original scanned image classes + 11 PTB-XL rendered classes).

2. **The Solution (The Bridge)**:
   - When an ECG image or photo is fed to the Open ECG Digitizer, it rectifies distortions and extracts the pure canonical time-series data for all 12 leads: `I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`.
   - This time-series table is presented to the user and can be inspected, searched, and downloaded as a CSV.
   - For classification, the digitized 12-lead time-series data is rendered using the exact calibrated 3x4 clinical grid synthesizer (`render_ecg_cv2`) that was used during PTB-XL model generation.
   - This feeds a clean, standardized, distortion-free 12-lead ECG representation directly into the 15-class MobileNetV2 deep learning classifier (`model.keras`).
   - The result is an accurate prediction of the disorder along with confidence scores, category, and medical causes.

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

## 🚀 How to Run the Application

### Option A: Using the Launcher Script (Double-click or terminal)
```bash
# Windows Command Prompt
run_app.bat

# PowerShell
.\run_app.ps1
```

### Option B: Using the Virtual Environment Directly
```powershell
& "ecg (2)\ecg\.venv\Scripts\python.exe" -m streamlit run app_combined.py --server.port 8501
```

Then open your browser at **`http://localhost:8501`**.

---

## 🖥️ Application Features

1. **Dual Input Mode**:
   - **Upload Custom ECG**: Upload any phone camera photo, scan, JPEG, PNG, or TIFF.
   - **Clinical Sample Gallery**: Select from pre-loaded clinical dataset samples covering all 15 disorders.
2. **Tab 1: Pipeline & Vision Stages**:
   - View the original image, perspective-aligned rectification, reconstructed calibrated grid, and U-Net feature probability maps.
3. **Tab 2: Time-Series Table**:
   - Interactive table showing `Time_s` and all 12 leads (`I` through `V6`).
   - One-click **Download Complete Time Series (CSV)**.
   - Lead statistical summary (mean, std, min, max, peak-to-peak amplitude in µV).
4. **Tab 3: Interactive 12-Lead Studio**:
   - Multi-lead Plotly waveforms with zoom, pan, hover tooltips, and lead filtering.
5. **Tab 4: 15-Disorder Clinical Diagnosis**:
   - Primary predicted disorder with color-coded category badge and confidence percentage.
   - Full 15-disorder horizontal probability distribution bar chart.
   - Clinical causes and etiology breakdown.
6. **Tab 5: Diagnostic Medical Report**:
   - Downloadable formatted text report summarizing all telemetry and diagnostic findings.
