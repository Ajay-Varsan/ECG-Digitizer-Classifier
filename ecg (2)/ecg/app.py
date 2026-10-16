"""
app.py
------
Interactive Clinical ECG Waveform & Arrhythmia Analysis Application (Streamlit).
Features:
- Upload any 12-lead ECG image or select from pre-loaded clinical dataset samples.
- Complete crest & trough fiducial detection (P, Q, R, S, T, Flutter, Fib waves).
- Interactive layer toggles for crests, troughs, anomaly bounding boxes, and lead regions.
- Multi-Arrhythmia clinical pattern alignment across 18+ disorder categories.
- Unique color coding for every single disorder.
- 1D Digital Waveform Trace viewer with fiducial peak/valley overlays.
- Clinical causes breakdown and printable diagnostic report export.
"""

import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf

from ecg_utils import detect_ecg_region, get_lead_boxes
from ecg_signal_extractor import extract_all_leads_signals, extract_lead_signal
from wave_detector import detect_fiducials_and_anomalies
from ecg_visualizer import annotate_ecg_image
from predict import analyze_ecg_image, CNN_CLASS_INFO
from disorder_palette import DISORDER_COLORS, WAVE_FIDUCIAL_COLORS, get_disorder_color

MODEL_PATH       = "model.keras"
CLASS_NAMES_PATH = "class_names.txt"

st.set_page_config(
    page_title="ECG Crest & Trough Anomaly Detector",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern medical UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00E676, #00E5FF, #2979FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #90A4AE;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1E2630;
        border: 1px solid #2C3847;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .disorder-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .cause-pill {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #00E5FF;
        padding: 6px 12px;
        margin: 4px 0px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_images():
    """Finds sample images from the local dataset directory."""
    samples = {}
    pattern = os.path.join("dataset", "**", "*.jpg")
    found = glob.glob(pattern, recursive=True)
    for p in found[:20]:
        cat_name = os.path.basename(os.path.dirname(p))
        file_name = os.path.basename(p)
        samples[f"[{cat_name[:18]}] {file_name}"] = p
    return samples


def main():
    st.markdown('<div class="main-header">🫀 ECG Waveform Crest/Trough & Arrhythmia Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Fiducial Peak/Valley Detection, Anomaly Bounding Boxes & Multi-Disorder Pattern Alignment</div>', unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.header("📁 Input ECG Image")
    samples = load_sample_images()
    input_mode = st.sidebar.radio("Select Input Source:", ["Choose Sample Dataset Image", "Upload Custom ECG Image"])

    image_path = None
    uploaded_file = None

    if input_mode == "Choose Sample Dataset Image":
        if samples:
            selected_sample = st.sidebar.selectbox("Select Sample:", list(samples.keys()))
            image_path = samples[selected_sample]
        else:
            st.sidebar.warning("No sample images found in dataset/ folder.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an ECG image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Bounding Box Layers")
    show_crests = st.sidebar.checkbox("Show Crests (P, R, T, Flutter peaks)", value=True)
    show_troughs = st.sidebar.checkbox("Show Troughs (Q, S, Inverted waves)", value=True)
    show_anomalies = st.sidebar.checkbox("Show Anomaly Boxes (Color-Coded)", value=True)
    show_leads = st.sidebar.checkbox("Show 12-Lead Region Boundaries", value=True)
    show_legend = st.sidebar.checkbox("Show Diagnostic Banner & Legend", value=True)

    st.sidebar.markdown("---")
    st.sidebar.header("🌈 Disorder Color Reference")
    with st.sidebar.expander("View All Disorder Colors"):
        for d_name, d_info in DISORDER_COLORS.items():
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-bottom:4px;'>"
                f"<div style='width:16px; height:16px; background-color:{d_info['hex']}; border-radius:3px; margin-right:8px;'></div>"
                f"<span style='font-size:0.85rem;'>{d_name}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Read image
    image = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif image_path is not None and os.path.exists(image_path):
        image = cv2.imread(image_path)

    if image is None:
        st.info("👈 Please select or upload an ECG report image in the sidebar to begin analysis.")
        return

    # Load CNN model (cached per session)
    cnn_model, class_names = None, None
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
        try:
            cnn_model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                class_names = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            pass

    # --- Processing Pipeline ---
    with st.spinner("Running CNN classifier and detecting waveform bounding boxes..."):
        # Save image to temp file so analyze_ecg_image can read it
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)

        result = analyze_ecg_image(tmp_path, cnn_model=cnn_model, class_names=class_names)
        os.unlink(tmp_path)

        primary       = result["primary_disorder"]
        agg_metrics   = result["metrics"]
        all_crests    = result["all_crests"]
        all_troughs   = result["all_troughs"]
        all_anomalies = result["all_anomalies"]
        all_fids      = result.get("all_anomalies", [])  # for tab3
        lead_boxes    = result["lead_boxes"]
        annotated_img = result["annotated_image"]
        all_class_probs = result.get("all_class_probs", {})

        # Also re-extract signals for waveform tab
        h_img, w_img = image.shape[:2]
        x, y, w, h = detect_ecg_region(image)
        leads_signals = extract_all_leads_signals(image, lead_boxes)

    # --- Main Display Layout ---
    col_left, col_right = st.columns([1.6, 1.0])

    with col_left:
        st.subheader("🖼️ Annotated ECG with Crest/Trough & Anomaly Bounding Boxes")
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Download Button
        success, encoded_img = cv2.imencode('.jpg', annotated_img)
        if success:
            st.download_button(
                label="📥 Download Annotated High-Res ECG Image",
                data=encoded_img.tobytes(),
                file_name="ecg_annotated_analysis.jpg",
                mime="image/jpeg",
            )

    with col_right:
        st.subheader("📋 Primary CNN Diagnosis")

        cnn_source = "CNN (15-class)" if cnn_model else "No model loaded"
        st.markdown(
            f"<div style='background-color:{primary['color_hex']}; color:#000; padding:12px 18px;"
            f"border-radius:8px; font-size:1.25rem; font-weight:700; margin-bottom:12px;'>"
            f"{primary['disorder']} <span style='font-size:0.85rem; opacity:0.8;'>({primary['category']})</span>"
            f"<span style='float:right;'>{primary['confidence_pct']:.1f}% confidence</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Metrics Grid
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Estimated Heart Rate", f"{agg_metrics['heart_rate_bpm']:.0f} BPM")
            st.metric("Mean PR Interval", f"{agg_metrics['mean_pr_ms']:.0f} ms")
            st.metric("Total Crests (Peaks)", f"{len(all_crests)}")
        with m_col2:
            st.metric("R-R Regularity (CV)", f"{agg_metrics['rr_cv']*100:.1f}%")
            st.metric("Mean QRS Duration", f"{agg_metrics['mean_qrs_ms']:.0f} ms")
            st.metric("Total Troughs (Valleys)", f"{len(all_troughs)}")

        st.markdown(f"**Detected Anomalous Regions:** `{len(all_anomalies)}` anomalies flagged")

        # Clinical Causes
        st.markdown("##### 🩺 Potential Clinical Causes:")
        for cause in primary.get("causes", []):
            st.markdown(f"<div class='cause-pill'>• {cause}</div>", unsafe_allow_html=True)

    # --- Waveform Trace Explorer & Multi-Disorder Alignment ---
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 1D Extracted Waveform Trace", "📊 Multi-Arrhythmia Alignment Ranking", "📦 Bounding Box Inspector"])

    with tab1:
        st.subheader("Interactive 1D Signal & Waveform Fiducials")
        lead_options = list(leads_signals.keys())
        selected_lead_name = st.selectbox("Select Lead to Inspect:", lead_options, index=0 if lead_options else 0)

        if selected_lead_name and selected_lead_name in leads_signals:
            sig_info = leads_signals[selected_lead_name]
            t = sig_info["time_sec"]
            v = sig_info["signal_mv"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=v, mode='lines', name=f'{selected_lead_name} Trace', line=dict(color='#00E5FF', width=2)))

            lead_fids = detect_fiducials_and_anomalies(sig_info, lead_label=selected_lead_name, global_box_offset=(0, 0))
            if lead_fids:
                r_items = [c for c in lead_fids["crests"] if c.get("type") == "R_PEAK"]
                if r_items:
                    fig.add_trace(go.Scatter(
                        x=[item["time_sec"] for item in r_items],
                        y=[item["amplitude_mv"] for item in r_items],
                        mode='markers+text', name='R-Crest (Peak)',
                        text=['R' for _ in r_items], textposition='top center',
                        marker=dict(color='#00FF00', size=9, symbol='triangle-up'),
                    ))
                p_items = [c for c in lead_fids["crests"] if "P-Crest" in c.get("name", "")]
                if p_items:
                    fig.add_trace(go.Scatter(
                        x=[item["time_sec"] for item in p_items],
                        y=[item["amplitude_mv"] for item in p_items],
                        mode='markers+text', name='P-Crest',
                        text=['P' for _ in p_items], textposition='top center',
                        marker=dict(color='#00BFFF', size=7, symbol='circle'),
                    ))
                q_items = [t for t in lead_fids["troughs"] if t.get("type") == "Q_TROUGH"]
                if q_items:
                    fig.add_trace(go.Scatter(
                        x=[item["time_sec"] for item in q_items],
                        y=[item["amplitude_mv"] for item in q_items],
                        mode='markers+text', name='Q-Trough',
                        text=['Q' for _ in q_items], textposition='bottom center',
                        marker=dict(color='#00FFFF', size=7, symbol='triangle-down'),
                    ))
                s_items = [t for t in lead_fids["troughs"] if t.get("type") == "S_TROUGH"]
                if s_items:
                    fig.add_trace(go.Scatter(
                        x=[item["time_sec"] for item in s_items],
                        y=[item["amplitude_mv"] for item in s_items],
                        mode='markers+text', name='S-Trough',
                        text=['S' for _ in s_items], textposition='bottom center',
                        marker=dict(color='#FFA500', size=7, symbol='triangle-down'),
                    ))

            fig.update_layout(
                title=f"Extracted Waveform for {selected_lead_name} (Time vs Amplitude)",
                xaxis_title="Time (seconds)", yaxis_title="Amplitude (mV)",
                template="plotly_dark", height=400, margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("CNN 15-Class Confidence Breakdown")
        if all_class_probs:
            sorted_probs = sorted(all_class_probs.items(), key=lambda kv: -kv[1])
            short_names = [CNN_CLASS_INFO.get(cls, {}).get("short", cls[:30]) for cls, _ in sorted_probs]
            prob_vals   = [p * 100 for _, p in sorted_probs]
            colors      = [CNN_CLASS_INFO.get(cls, {}).get("color_hex", "#90A4AE") for cls, _ in sorted_probs]

            bar_fig = go.Figure(go.Bar(
                x=prob_vals, y=short_names, orientation='h',
                marker_color=colors,
                text=[f"{p:.1f}%" for p in prob_vals],
                textposition='outside',
            ))
            bar_fig.update_layout(
                title="All 15 Class Probabilities (CNN)",
                xaxis_title="Probability (%)", yaxis_title="Disorder",
                template="plotly_dark", height=520,
                margin=dict(l=200, r=80, t=50, b=40),
                xaxis=dict(range=[0, 105]),
            )
            bar_fig.update_yaxes(autorange="reversed")
            st.plotly_chart(bar_fig, use_container_width=True)

            rank_data = [
                {"Rank": i+1, "Disorder": CNN_CLASS_INFO.get(cls, {}).get("short", cls),
                 "Category": CNN_CLASS_INFO.get(cls, {}).get("category", ""),
                 "Probability": f"{p*100:.2f}%",
                 "Color": CNN_CLASS_INFO.get(cls, {}).get("color_hex", "")}
                for i, (cls, p) in enumerate(sorted_probs)
            ]
            st.dataframe(pd.DataFrame(rank_data), use_container_width=True)
        else:
            st.warning("No CNN model loaded. Run train.py first to generate model.keras.")

    with tab3:
        st.subheader("Fiducial Bounding Box Telemetry")
        st.write(f"Total Crests: {len(all_crests)} | Total Troughs: {len(all_troughs)} | Total Anomalies: {len(all_anomalies)}")
        if all_anomalies:
            st.markdown("##### 🚨 Flagged Anomaly Bounding Boxes:")
            anom_df = pd.DataFrame([
                {
                    "Anomaly Name": a["name"],
                    "Disorder Match": a.get("disorder_match", "General"),
                    "Time (s)": f"{a.get('time_sec', 0):.2f}",
                    "Global Box [x1, y1, x2, y2]": str(a.get("global_box", [])),
                    "Note": a.get("note", ""),
                }
                for a in all_anomalies
            ])
            st.dataframe(anom_df, use_container_width=True)


if __name__ == "__main__":
    main()
