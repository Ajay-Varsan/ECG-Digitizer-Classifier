import sys
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.io import decode_image

# Add paths
root_dir = os.path.abspath(".")
digitizer_path = os.path.join(root_dir, "Open-ECG-Digitizer-main")
ecg_path = os.path.join(root_dir, "ecg (2)", "ecg")

sys.path.insert(0, digitizer_path)
sys.path.insert(0, ecg_path)

from src.config.default import get_cfg
from src.model.inference_wrapper import InferenceWrapper
from predict import run_cnn, CNN_CLASS_INFO
from arrhythmia_classifier import ARRHYTHMIA_KNOWLEDGE_BASE
import keras

print("1. Loading Digitizer...")
cfg = get_cfg()
cfg.merge_from_file(os.path.join(digitizer_path, "src", "config", "inference_wrapper.yml"))
cfg.MODEL.KWARGS.device = "cpu"
cfg.MODEL.KWARGS.config.device = "cpu"
cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER.KWARGS.device = "cpu"
# Fix relative paths
cfg.MODEL.KWARGS.config.SEGMENTATION_MODEL.weight_path = os.path.join(digitizer_path, "weights", "unet_weights_07072025.pt")
cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER.config_path = os.path.join(digitizer_path, "src", "config", "lead_layouts_all.yml")
cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER.unet_config_path = os.path.join(digitizer_path, "src", "config", "lead_name_unet.yml")
cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER.unet_weight_path = os.path.join(digitizer_path, "weights", "lead_name_unet_weights_07072025.pt")

wrapper = InferenceWrapper(cfg.MODEL.KWARGS.config, device="cpu")
wrapper.eval()
print("   Digitizer loaded successfully!")

print("2. Loading 15-class ECG CNN model...")
cnn = keras.models.load_model(os.path.join(ecg_path, "model.keras"))
with open(os.path.join(ecg_path, "class_names.txt")) as f:
    class_names = [line.strip() for line in f if line.strip()]
print(f"   Loaded CNN with {len(class_names)} classes.")

print("3. Running digitizer on sample image...")
sample_img_path = os.path.join(digitizer_path, "assets", "visual_abstract-img0.png")
img_tensor = decode_image(sample_img_path, mode="RGB").unsqueeze(0)
out = wrapper(img_tensor)

canonical = out["signal"].get("canonical_lines")
print("   Digitizer output shape:", canonical.shape if canonical is not None else None)

# Convert canonical to time-series DataFrame
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
signals_np = canonical.squeeze().cpu().numpy()  # shape (12, 5000)

# Replace NaNs by linear interpolation
for i in range(signals_np.shape[0]):
    series = pd.Series(signals_np[i])
    signals_np[i] = series.interpolate(method="linear").bfill().ffill().fillna(0).values

# Transpose to (5000, 12)
signals_t = signals_np.T

# Create time series table
sampling_rate_hz = 500.0
time_sec = np.arange(len(signals_t)) / sampling_rate_hz
df_table = pd.DataFrame(signals_t, columns=lead_names)
df_table.insert(0, "Time_s", time_sec)
print(f"   Created Time-Series Table: {df_table.shape} rows x cols")
print(df_table.head(3))

print("4. Rendering 12-lead standard grid image from digitized signals...")
from dataset_builder import render_ecg_cv2
render_dir = os.path.join(root_dir, "scratch")
os.makedirs(render_dir, exist_ok=True)
rendered_img_path = os.path.join(render_dir, "rendered_from_digitizer.jpg")
render_ecg_cv2(signals_t, rendered_img_path)
print(f"   Saved rendered standard ECG image to {rendered_img_path}")

print("5. Classifying ECG into 15 disorders...")
rendered_bgr = cv2.imread(rendered_img_path)
top_cls, conf, probs = run_cnn(rendered_bgr, 0, 0, rendered_bgr.shape[1], rendered_bgr.shape[0], cnn, class_names)
print(f"   PRIMARY PREDICTION: {top_cls}")
print(f"   Confidence: {conf * 100:.2f}%")
print("   Top 3 predictions:")
sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
for c_name, p in sorted_probs:
    print(f"     - {c_name}: {p*100:.2f}%")
