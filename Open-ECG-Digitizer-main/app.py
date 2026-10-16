import os
import io
import time
import glob
import subprocess
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from yacs.config import CfgNode as CN
from src.config.default import get_cfg
from src.model.inference_wrapper import InferenceWrapper
from src.digitize import decode_and_prepare_image, LEAD_NAMES
from src.dataset.hf_utils import (
    VAL_SHARDS,
    TRAIN_SHARDS,
    download_shard_file,
    extract_samples_from_parquet,
    update_unet_yaml,
)

# Page configuration
st.set_page_config(
    page_title="Open ECG Digitizer & Training Studio",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0b4f6c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #0b4f6c;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 10px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_digitizer_pipeline(config_file: str, weight_override: str = None, device_name: str = "cuda"):
    """Caches and loads the inference pipeline on GPU/CPU."""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    actual_device = device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    cfg.MODEL.KWARGS.device = actual_device
    cfg.MODEL.KWARGS.config.device = actual_device
    
    if weight_override and os.path.exists(weight_override):
        cfg.MODEL.KWARGS.config.SEGMENTATION_MODEL.weight_path = weight_override
        
    wrapper = InferenceWrapper(cfg.MODEL.KWARGS.config, device=actual_device)
    wrapper.eval()
    return wrapper


def render_ecg_plotly(canonical_tensor: torch.Tensor, sampling_rate_hz: float = 500.0):
    """Renders 12-lead ECG time series in an interactive Plotly chart with 12 stacked subplots."""
    data = canonical_tensor.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    n_leads, n_points = data.shape
    time_sec = np.arange(n_points) / sampling_rate_hz
    
    lead_labels = LEAD_NAMES[:n_leads]
    
    # 12 subplots (stacked vertically or 4x3)
    fig = make_subplots(
        rows=n_leads,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=lead_labels,
    )
    
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#3366cc", "#dc3912"
    ]
    
    for i in range(n_leads):
        y_vals = data[i]
        valid_mask = ~np.isnan(y_vals)
        fig.add_trace(
            go.Scatter(
                x=time_sec,
                y=y_vals,
                mode="lines",
                name=lead_labels[i],
                line=dict(color=colors[i % len(colors)], width=1.4),
                connectgaps=False,
                hovertemplate=f"<b>Lead {lead_labels[i]}</b><br>Time: %{{x:.3f}}s<br>Voltage: %{{y:.1f}} µV<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )
        if np.any(valid_mask):
            ymin, ymax = float(np.min(y_vals[valid_mask])), float(np.max(y_vals[valid_mask]))
            pad = max(abs(ymax - ymin) * 0.15, 40.0)
            fig.update_yaxes(range=[ymin - pad, ymax + pad], title_text=f"{lead_labels[i]} (µV)", row=i + 1, col=1)
        else:
            fig.update_yaxes(title_text=f"{lead_labels[i]} (µV)", row=i + 1, col=1)
        
    fig.update_xaxes(title_text="Time (seconds)", row=n_leads, col=1)
    fig.update_layout(
        height=150 * n_leads,
        margin=dict(l=60, r=30, t=40, b=50),
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor="#fafbfc",
        paper_bgcolor="#ffffff",
    )
    return fig, pd.DataFrame(data.T, columns=lead_labels, index=time_sec)


def render_clinical_ecg_plotly(canonical_tensor: torch.Tensor, sampling_rate_hz: float = 500.0):
    """Renders 12-lead ECG in the standard Clinical 3x4 layout plus continuous bottom rhythm strip."""
    data = canonical_tensor.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    n_leads, n_points = data.shape
    time_sec = np.arange(n_points) / sampling_rate_hz

    # 3x4 grid mapping: (row, col, lead_name, lead_index)
    grid_map = [
        (1, 1, "I", 0),   (1, 2, "aVR", 3), (1, 3, "V1", 6), (1, 4, "V4", 9),
        (2, 1, "II", 1),  (2, 2, "aVL", 4), (2, 3, "V2", 7), (2, 4, "V5", 10),
        (3, 1, "III", 2), (3, 2, "aVF", 5), (3, 3, "V3", 8), (3, 4, "V6", 11),
    ]

    specs = [
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [{"colspan": 4, "type": "xy"}, None, None, None],
    ]

    titles = [f"Lead {name}" for _, _, name, _ in grid_map] + ["Lead II Rhythm Strip (Full 10-Second Continuous Recording)"]

    fig = make_subplots(
        rows=4,
        cols=4,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
    )

    lead_colors = {
        "I": "#0b4f6c", "aVR": "#d95f02", "V1": "#7570b3", "V4": "#e7298a",
        "II": "#0b4f6c", "aVL": "#d95f02", "V2": "#7570b3", "V5": "#e7298a",
        "III": "#0b4f6c", "aVF": "#d95f02", "V3": "#7570b3", "V6": "#e7298a",
    }

    for r, c, name, idx in grid_map:
        if idx < n_leads:
            y = data[idx]
            mask = ~np.isnan(y)
            if np.any(mask):
                x_valid = time_sec[mask]
                y_valid = y[mask]
                fig.add_trace(
                    go.Scatter(
                        x=x_valid,
                        y=y_valid,
                        mode="lines",
                        name=f"Lead {name}",
                        line=dict(color=lead_colors.get(name, "#1f77b4"), width=1.5),
                        hovertemplate=f"<b>Lead {name}</b><br>Time: %{{x:.3f}}s<br>Voltage: %{{y:.1f}} µV<extra></extra>",
                    ),
                    row=r,
                    col=c,
                )
                ymin, ymax = float(np.min(y_valid)), float(np.max(y_valid))
                pad = max(abs(ymax - ymin) * 0.15, 50.0)
                fig.update_yaxes(range=[ymin - pad, ymax + pad], row=r, col=c, gridcolor="#eeeeee", zerolinecolor="#cccccc")
                fig.update_xaxes(row=r, col=c, gridcolor="#eeeeee")

    # Rhythm strip (Lead II is index 1)
    if n_leads > 1:
        y_rhythm = data[1]
        mask_rhythm = ~np.isnan(y_rhythm)
        if np.any(mask_rhythm):
            fig.add_trace(
                go.Scatter(
                    x=time_sec[mask_rhythm],
                    y=y_rhythm[mask_rhythm],
                    mode="lines",
                    name="Lead II Rhythm",
                    line=dict(color="#b2182b", width=1.3),
                    hovertemplate="<b>Lead II Rhythm</b><br>Time: %{x:.3f}s<br>Voltage: %{y:.1f} µV<extra></extra>",
                ),
                row=4,
                col=1,
            )
            ymin, ymax = float(np.min(y_rhythm[mask_rhythm])), float(np.max(y_rhythm[mask_rhythm]))
            pad = max(abs(ymax - ymin) * 0.15, 50.0)
            fig.update_yaxes(range=[ymin - pad, ymax + pad], title_text="Voltage (µV)", row=4, col=1, gridcolor="#eeeeee", zerolinecolor="#cccccc")
            fig.update_xaxes(title_text="Time (seconds)", range=[0, 10.0], row=4, col=1, gridcolor="#eeeeee")

    fig.update_layout(
        height=850,
        margin=dict(l=55, r=30, t=50, b=40),
        showlegend=False,
        plot_bgcolor="#fafbfc",
        paper_bgcolor="#ffffff",
    )
    return fig


# Sidebar Header
with st.sidebar:
    st.image("assets/pipeline-overview.svg", use_column_width=True)
    st.markdown("### 🫀 Open ECG Digitizer")
    st.caption("Deep Learning Paper ECG Digitization Platform")
    
    device_avail = "CUDA (GPU) 🟢" if torch.cuda.is_available() else "CPU 🟡"
    st.info(f"**Hardware:** {device_avail}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.caption(f"{gpu_name} ({vram_gb:.1f} GB VRAM)")

    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
            "⚡ Digitize ECG (Inference)",
            "📦 Hugging Face Dataset",
            "🎯 Model Training & Fine-Tuning",
            "⚙️ Configurations & YAML",
            "📖 Paper Citation",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("Based on npj Digital Medicine (2026)")


# -----------------------------------------------------------------------------------------
# TAB 1: DIGITIZE ECG (INFERENCE)
# -----------------------------------------------------------------------------------------
if menu == "⚡ Digitize ECG (Inference)":
    st.markdown('<div class="main-header">⚡ ECG Digitization & Trace Extraction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Extract calibrated 12-lead microvolt voltage–time series from scanned or photographed ECGs.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. Input ECG Image")
        input_mode = st.radio("Choose Input Method:", ["Select Sample Asset", "Upload Image File"])
        
        sample_assets = sorted(glob.glob("assets/*.png") + glob.glob("assets/*.jpg"))
        selected_img_path = None
        
        if input_mode == "Select Sample Asset":
            selected_sample = st.selectbox("Select Asset Image:", sample_assets)
            if selected_sample:
                selected_img_path = selected_sample
                st.image(selected_sample, caption="Selected Sample", use_column_width=True)
        else:
            uploaded_file = st.file_uploader("Upload ECG Image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                temp_dir = "sandbox/uploads"
                os.makedirs(temp_dir, exist_ok=True)
                selected_img_path = os.path.join(temp_dir, uploaded_file.name)
                with open(selected_img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.image(selected_img_path, caption="Uploaded ECG", use_column_width=True)

        st.subheader("2. Model & Pipeline Settings")
        available_weights = ["weights/unet_weights_07072025.pt"] + glob.glob("sandbox/*.pt")
        selected_weight = st.selectbox("Segmentation Weights:", available_weights)
        device_choice = st.selectbox("Inference Device:", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
        apply_dewarp = st.checkbox("Enable Experimental Dewarping", value=False)
        
        layout_dict = {
            "Auto-Detect (Smart Row Matching)": None,
            "Standard 3×4 + 1 Rhythm Lead (Most Clinical ECGs)": "standard_3x4_with_r1",
            "Standard 3×4 (No Rhythm Lead)": "standard_3x4",
            "Standard 3×4 + 3 Rhythm Leads": "standard_3x4_with_r3",
            "Standard 6×2 (Two Columns)": "standard_6x2",
            "Standard 12×1 (Single Column)": "standard_12x1",
            "Precordial 6×1": "precordial_6x1",
            "Cabrera 6×1 Limb": "cabrera_6x1_limb",
        }
        chosen_layout_label = st.selectbox(
            "ECG Lead Layout Format:",
            options=list(layout_dict.keys()),
            index=0,
            help="Select 'Standard 3x4 + 1 Rhythm Lead' if auto-detection misses the layout on phone photos."
        )
        selected_layout_override = layout_dict[chosen_layout_label]

        run_button = st.button("🚀 Run Digitization Pipeline", type="primary", use_container_width=True)

    with col2:
        st.subheader("3. Digitization Results")
        if run_button:
            if not selected_img_path or not os.path.exists(selected_img_path):
                st.error("Please select or upload a valid ECG image first.")
            else:
                progress_placeholder = st.empty()
                progress_placeholder.info("⏳ Initializing models and loading image...")
                
                try:
                    start_t = time.time()
                    wrapper = load_digitizer_pipeline(
                        "src/config/inference_wrapper.yml",
                        weight_override=selected_weight,
                        device_name=device_choice,
                    )
                    wrapper.apply_dewarping = apply_dewarp
                    
                    progress_placeholder.info("🔍 Segmenting leads, grid, and detecting perspective...")
                    image_tensor = decode_and_prepare_image(selected_img_path)
                    
                    with torch.no_grad():
                        result = wrapper(
                            image_tensor,
                            layout_should_include_substring=None,
                            override_layout=selected_layout_override,
                        )
                    
                    elapsed = time.time() - start_t
                    progress_placeholder.success(f"✅ Digitization complete in {elapsed:.2f} seconds!")
                    
                    # Store in session state
                    st.session_state["ecg_result"] = result
                    st.session_state["ecg_img_path"] = selected_img_path

                except Exception as e:
                    progress_placeholder.error(f"Inference error: {str(e)}")
                    st.exception(e)

        if "ecg_result" in st.session_state:
            res = st.session_state["ecg_result"]
            
            # Key Metrics Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Detected Layout", res.get("layout_name", "Unknown"))
            with m2:
                px_x = res.get("pixel_spacing_mm", {}).get("x", 0)
                st.metric("Grid X Spacing", f"{px_x:.4f} mm/px" if px_x else "N/A")
            with m3:
                px_y = res.get("pixel_spacing_mm", {}).get("y", 0)
                st.metric("Grid Y Spacing", f"{px_y:.4f} mm/px" if px_y else "N/A")
            with m4:
                canonical = res.get("signal", {}).get("canonical_lines", None)
                n_leads = canonical.shape[0] if canonical is not None else 0
                st.metric("Extracted Leads", f"{n_leads} Leads")

            # Intermediate Visualizations
            tab_signals, tab_alignment, tab_segmentation = st.tabs([
                "📊 12-Lead Digitized Traces",
                "📐 Perspective & Alignment",
                "🔬 U-Net Segmentation Channels",
            ])

            with tab_signals:
                if canonical is not None:
                    # View Mode Toggle
                    display_mode = st.radio(
                        "ECG Display Format:",
                        [
                            "📋 Clinical 3×4 Paper Layout (+ 10s Rhythm Strip)",
                            "📈 Synchronized 12-Lead Stacked View (0–10s)",
                        ],
                        horizontal=True,
                    )

                    st.info(
                        """
                        💡 **Clinical Format Explained:**
                        Standard paper ECGs record 12 leads across 4 sequential columns of 2.5 seconds each ($4 \\times 2.5\\text{s} = 10\\text{s}$ total paper width):
                        - **0.0 – 2.5s**: Leads I, II, III (Limb leads)
                        - **2.5 – 5.0s**: Leads aVR, aVL, aVF (Augmented leads)
                        - **5.0 – 7.5s**: Leads V1, V2, V3 (Septal/Anterior chest leads)
                        - **7.5 – 10.0s**: Leads V4, V5, V6 (Lateral chest leads)
                        - **0.0 – 10.0s (Bottom)**: Continuous Lead II Rhythm Strip (all 10 seconds continuous signal)
                        """
                    )

                    if "Clinical 3×4" in display_mode:
                        fig_clinical = render_clinical_ecg_plotly(canonical)
                        st.plotly_chart(fig_clinical, use_container_width=True)
                    else:
                        fig_stacked, _ = render_ecg_plotly(canonical)
                        st.plotly_chart(fig_stacked, use_container_width=True)

                    _, df_signals = render_ecg_plotly(canonical)

                    # Export buttons
                    csv_bytes = df_signals.to_csv(index_label="time_seconds").encode("utf-8")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            label="📥 Download Time-Series CSV (µV)",
                            data=csv_bytes,
                            file_name="digitized_ecg_signals.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with c2:
                        # Quick Preview Table
                        with st.expander("Peek Raw Time-Series Data (First 10 rows)"):
                            st.dataframe(df_signals.head(10))
                else:
                    st.warning("No canonical lead signals were extracted.")

            with tab_alignment:
                c_orig, c_aligned = st.columns(2)
                with c_orig:
                    st.markdown("**Original Image & Corner Points**")
                    orig_img = res["input_image"].squeeze().permute(1, 2, 0).cpu().numpy()
                    fig_corners, ax = plt.subplots(figsize=(6, 5))
                    ax.imshow(orig_img * 0.999)
                    sp = res["source_points"].cpu().numpy()
                    ax.scatter(sp[:, 0], sp[:, 1], c="red", s=30, label="Grid Boundary Points")
                    ax.axis("off")
                    st.pyplot(fig_corners)
                with c_aligned:
                    st.markdown("**Corrected & Resampled Image**")
                    aligned_img = res["aligned"]["image"].squeeze().permute(1, 2, 0).cpu().numpy()
                    st.image(np.clip(aligned_img, 0, 1), use_column_width=True)

            with tab_segmentation:
                cs1, cs2, cs3 = st.columns(3)
                aligned = res["aligned"]
                with cs1:
                    st.markdown("**ECG Signal Probability**")
                    sig_map = aligned["signal_prob"].squeeze().cpu().numpy()
                    st.image(np.clip(sig_map, 0, 1), use_column_width=True)
                with cs2:
                    st.markdown("**Grid Probability**")
                    grid_map = aligned["grid_prob"].squeeze().cpu().numpy()
                    st.image(np.clip(grid_map, 0, 1), use_column_width=True)
                with cs3:
                    st.markdown("**Text & Background Probability**")
                    text_map = aligned["text_prob"].squeeze().cpu().numpy()
                    st.image(np.clip(text_map, 0, 1), use_column_width=True)


# -----------------------------------------------------------------------------------------
# TAB 2: HUGGING FACE DATASET
# -----------------------------------------------------------------------------------------
elif menu == "📦 Hugging Face Dataset":
    st.markdown('<div class="main-header">📦 Hugging Face Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Connects to **`Ahus-AIM/Open-ECG-Digitizer-Development-Dataset`**.
        Contains 4,112 train and 217 val synthetic & realistic ECG scans with pixel-level ground truth annotations
        across multiple grid layouts, colours, perspective distortions, and noise.
        """
    )

    t_overview, t_download, t_inspector = st.tabs([
        "📋 Dataset Overview",
        "📥 Download & Unpack Shards",
        "🖼️ Local Sample Gallery",
    ])

    with t_overview:
        st.markdown(
            """
            ### Characteristics
            - **Trace Annotations:** Pixel-level binary masks of ECG signal traces
            - **Grid Annotations:** Coordinate-level grid markings and color variations
            - **Distortions:** Rotations up to 45°, perspective warp, non-uniform illumination, scanner folds, shadows
            - **Format in Repo:** Parquet shards containing raw image bytes, mask bytes, and WFDB signal records (.dat / .hea).
            """
        )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.info(
                f"""
                **Validation Split:**
                - Examples: 217
                - Total Size: ~3.8 GB across 4 parquet shards
                - Shards: `{VAL_SHARDS[0]}` to `{VAL_SHARDS[-1]}`
                """
            )
        with col_t2:
            st.info(
                f"""
                **Training Split:**
                - Examples: 4,112
                - Total Size: ~70 GB across 68 parquet shards
                - Shards: `{TRAIN_SHARDS[0]}` to `{TRAIN_SHARDS[-1]}`
                """
            )

    with t_download:
        st.subheader("Download Shards & Extract for Training")
        st.caption("Each shard contains ~50–60 full-resolution ECG images and corresponding pixel masks.")
        
        split_to_dl = st.radio("Select Split:", ["Validation (val)", "Training (train)"], horizontal=True)
        shards_list = VAL_SHARDS if "Validation" in split_to_dl else TRAIN_SHARDS
        chosen_shard = st.selectbox("Select Parquet Shard:", shards_list)
        
        target_dir = st.text_input(
            "Extract Samples Destination Directory:",
            value="datasets/val" if "Validation" in split_to_dl else "datasets/train",
        )
        max_extract = st.number_input("Max Samples to Extract from this Shard (0 = all in shard):", min_value=0, max_value=60, value=25)
        
        dl_btn = st.button("📥 Download Shard & Extract Images + Masks", type="primary")
        
        if dl_btn:
            with st.status(f"Processing shard {chosen_shard}...", expanded=True) as status:
                st.write("1. Downloading parquet file from Hugging Face...")
                shard_path = download_shard_file(chosen_shard, output_dir="dataset_cache")
                st.write(f"Parquet saved at: `{shard_path}` ({os.path.getsize(shard_path) / (1024**2):.1f} MB)")
                
                st.write(f"2. Unpacking PNG image and mask pairs into `{target_dir}`...")
                limit = max_extract if max_extract > 0 else None
                saved_count = extract_samples_from_parquet(
                    shard_path,
                    output_dir=target_dir,
                    max_samples=limit,
                )
                status.update(label=f"Extracted {saved_count} samples successfully!", state="complete")
                st.success(f"Extracted {saved_count} ECG scan & mask pairs to `{target_dir}`!")

    with t_inspector:
        st.subheader("Browse Extracted Local Dataset")
        local_train = sorted(glob.glob("datasets/train/*.png"))
        local_val = sorted(glob.glob("datasets/val/*.png"))
        
        # Filter only scan files (not mask files)
        train_scans = [f for f in local_train if not f.endswith("_mask.png")]
        val_scans = [f for f in local_val if not f.endswith("_mask.png")]
        
        st.write(f"**Local Dataset Status:** {len(train_scans)} Train scans | {len(val_scans)} Val scans")
        
        folder_choice = st.radio("Browse Folder:", ["Train", "Val"], horizontal=True)
        scans_to_browse = train_scans if folder_choice == "Train" else val_scans
        
        if scans_to_browse:
            idx = st.slider("Select Sample Index:", 0, len(scans_to_browse) - 1, 0)
            sample_scan_path = scans_to_browse[idx]
            sample_mask_path = sample_scan_path.replace(".png", "_mask.png")
            
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown(f"**ECG Scan:** `{os.path.basename(sample_scan_path)}`")
                st.image(sample_scan_path, use_column_width=True)
            with c_s2:
                st.markdown(f"**Ground Truth Segmentation Mask:** `{os.path.basename(sample_mask_path)}`")
                if os.path.exists(sample_mask_path):
                    st.image(sample_mask_path, use_column_width=True)
                else:
                    st.warning("Mask file not found.")
        else:
            st.info("No extracted samples found in selected folder yet. Use the download tab above to fetch samples!")


# -----------------------------------------------------------------------------------------
# TAB 3: MODEL TRAINING & FINE-TUNING
# -----------------------------------------------------------------------------------------
elif menu == "🎯 Model Training & Fine-Tuning":
    st.markdown('<div class="main-header">🎯 U-Net Segmentation Network Training</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Train or fine-tune the 4-channel U-Net segmentation network on your dataset.</div>',
        unsafe_allow_html=True,
    )

    col_t_left, col_t_right = st.columns([1, 1])

    with col_t_left:
        st.subheader("1. Training Hyperparameters")
        
        train_path = st.text_input("Train Dataset Directory:", value=os.path.abspath("datasets/train").replace("\\", "/"))
        val_path = st.text_input("Val Dataset Directory:", value=os.path.abspath("datasets/val").replace("\\", "/"))
        
        # Verify samples count
        n_train = len([f for f in glob.glob(os.path.join(train_path, "*.png")) if not f.endswith("_mask.png")])
        n_val = len([f for f in glob.glob(os.path.join(val_path, "*.png")) if not f.endswith("_mask.png")])
        
        st.caption(f"📊 Detected: {n_train} Train pairs, {n_val} Val pairs")
        
        hp_col1, hp_col2 = st.columns(2)
        with hp_col1:
            epochs = st.slider("Epochs:", min_value=1, max_value=50, value=2)
            batch_size = st.selectbox("Batch Size:", [1, 2, 4, 8], index=1)
            learning_rate = st.select_slider(
                "Learning Rate:",
                options=[0.0001, 0.0005, 0.001, 0.002, 0.0037, 0.005, 0.01],
                value=0.001,
            )
        with hp_col2:
            device_train = st.selectbox("Training Device:", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
            init_weights = st.radio("Weight Initialization:", ["Fine-tune from Pretrained Weights", "Train from Scratch"])
            optimizer_name = st.selectbox("Optimizer:", ["src.optimizer.adammuon.AdamMuon", "torch.optim.AdamW"])

        st.subheader("2. Start Training Job")
        start_train_btn = st.button("🔥 Launch Model Training", type="primary", use_container_width=True)

    with col_t_right:
        st.subheader("3. Live Training Monitor")
        
        chart_placeholder = st.empty()
        log_placeholder = st.empty()
        
        if start_train_btn:
            if n_train == 0 or n_val == 0:
                st.error("Cannot train: Dataset folder has 0 samples. Please download/extract samples in the Dataset tab first!")
            else:
                # Update unet.yml
                update_unet_yaml(
                    train_dir=train_path,
                    val_dir=val_path,
                    test_dir=val_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=device_train,
                    compile_model=False,
                )
                
                os.makedirs("sandbox", exist_ok=True)
                
                with st.spinner("Training in progress..."):
                    # Run training via subprocess
                    cmd = [
                        "python", "-m", "src.train",
                        "--config", "src/config/unet.yml"
                    ]
                    
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    
                    logs = []
                    train_losses = []
                    val_losses = []
                    
                    for line in iter(process.stdout.readline, ""):
                        logs.append(line)
                        # Display latest log lines
                        log_placeholder.code("".join(logs[-12:]))
                        
                        # Parse epoch metrics
                        if "train_loss" in line and "val_loss" in line:
                            import ast
                            try:
                                parts = line.split(":", 1)
                                if len(parts) == 2:
                                    dict_str = parts[1].strip()
                                    metrics_dict = ast.literal_eval(dict_str)
                                    train_losses.append(metrics_dict.get("train_loss", 0))
                                    val_losses.append(metrics_dict.get("val_loss", 0))
                                    
                                    # Update loss curve
                                    df_loss = pd.DataFrame({
                                        "Epoch": range(1, len(train_losses) + 1),
                                        "Train Loss": train_losses,
                                        "Validation Loss": val_losses,
                                    })
                                    chart_placeholder.line_chart(df_loss.set_index("Epoch"))
                            except Exception:
                                pass
                                
                    process.wait()
                    
                    if process.returncode == 0:
                        st.success("🎉 Training complete! Checkpoints saved to `sandbox/` directory.")
                    else:
                        st.error(f"Training ended with exit code {process.returncode}")


# -----------------------------------------------------------------------------------------
# TAB 4: CONFIGURATIONS & YAML
# -----------------------------------------------------------------------------------------
elif menu == "⚙️ Configurations & YAML":
    st.markdown('<div class="main-header">⚙️ Configuration & YAML Manager</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect or edit model architectures and inference pipelines.</div>', unsafe_allow_html=True)

    config_choice = st.selectbox(
        "Select YAML Config to View/Edit:",
        [
            "src/config/unet.yml",
            "src/config/inference_wrapper.yml",
            "src/config/lead_layouts_reduced.yml",
            "src/config/lead_name_unet.yml",
        ],
    )
    
    if os.path.exists(config_choice):
        with open(config_choice, "r") as f:
            yaml_content = f.read()
            
        edited_content = st.text_area("Config Content:", value=yaml_content, height=450)
        
        c_save, c_reset = st.columns([1, 1])
        with c_save:
            if st.button("💾 Save Changes to YAML", type="primary"):
                with open(config_choice, "w") as f:
                    f.write(edited_content)
                st.success(f"Saved `{config_choice}` successfully!")


# -----------------------------------------------------------------------------------------
# TAB 5: PAPER CITATION
# -----------------------------------------------------------------------------------------
elif menu == "📖 Paper Citation":
    st.markdown('<div class="main-header">📖 Mandatory Citation & References</div>', unsafe_allow_html=True)
    st.markdown(
        """
        If you use this software, code, or dataset in your research or applications, please cite the original study:
        """
    )
    st.markdown(
        """
        > **Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research**  
        > *Elias Stenhede, Agnar Martin Bjørnstad, Arian Ranjbar*  
        > **npj Digital Medicine (2026)**. DOI: [10.1038/s41746-025-02327-1](https://doi.org/10.1038/s41746-025-02327-1)
        """
    )

    bibtex_str = """@article{stenhede_digitizing_2026,
  title        = {Digitizing Paper {ECGs} at Scale: An Open-Source Algorithm for Clinical Research},
  author       = {Stenhede, Elias and Bjørnstad, Agnar Martin and Ranjbar, Arian},
  journal      = {npj Digital Medicine},
  year         = {2026},
  doi          = {10.1038/s41746-025-02327-1},
  url          = {https://doi.org/10.1038/s41746-025-02327-1},
  shorttitle   = {Digitizing Paper {ECGs} at Scale}
}"""

    st.code(bibtex_str, language="bibtex")
