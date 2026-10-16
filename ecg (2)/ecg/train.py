"""
train.py
--------
15-Class ECG Disorder Classification Training Engine.

Dataset (15 classes total):
    Original 4 (scanned ECG image folders):
        [0]  ECG Images of Myocardial Infarction Patients (240x12=2880)
        [1]  ECG Images of Patient that have History of MI (172x12=2064)
        [2]  ECG Images of Patient that have abnormal heartbeat (233x12=2796)
        [3]  Normal Person ECG Images (284x12=3408)

    New 11 (rendered from PTB-XL signals via dataset_builder.py):
        [4]  Atrial Fibrillation
        [5]  Atrial Flutter
        [6]  Sinus Tachycardia
        [7]  Sinus Bradycardia
        [8]  Complete Left Bundle Branch Block
        [9]  Complete Right Bundle Branch Block
        [10] Incomplete Right Bundle Branch Block
        [11] First Degree AV Block
        [12] Left Anterior Fascicular Block
        [13] Left Ventricular Hypertrophy
        [14] Left Atrial Enlargement

Architecture: MobileNetV2 Transfer Learning
Training:     Two-stage (head warmup -> full fine-tuning)

Run dataset_builder.py first to render the 11 new PTB-XL classes,
then run this script to train on all 15 classes.
"""

import os
import random
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight

from ecg_utils import load_and_crop, IMG_SIZE


# =============================================================================
# SETTINGS
# =============================================================================

DATASET_DIR  = "dataset"
RENDERED_DIR = os.path.join(DATASET_DIR, "rendered_ptbxl")
OUTPUT_DIR   = "outputs"

BATCH_SIZE       = 32
INITIAL_EPOCHS   = 10
FINE_TUNE_EPOCHS = 15
VAL_TEST_SPLIT   = 0.30
RANDOM_STATE     = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# GPU CHECK
# =============================================================================

gpus = tf.config.list_physical_devices("GPU")
print(f"GPU available: {bool(gpus)}")
if gpus:
    print(f"Using: {gpus[0]}")
else:
    print("Using CPU")


# =============================================================================
# CLASS DEFINITIONS  (4 original + 11 PTB-XL rendered = 15 total)
# =============================================================================

ORIGINAL_CLASSES = [
    "ECG Images of Myocardial Infarction Patients (240x12=2880)",
    "ECG Images of Patient that have History of MI (172x12=2064)",
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)",
    "Normal Person ECG Images (284x12=3408)",
]

RENDERED_CLASSES = [
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

class_names = ORIGINAL_CLASSES + RENDERED_CLASSES  # indices 0-14

print(f"\n{'='*60}")
print(f"15-CLASS ECG DISORDER CLASSIFICATION")
print(f"{'='*60}")
print(f"\nTarget classes ({len(class_names)} total):")
for i, name in enumerate(class_names):
    src = "Original Image" if i < len(ORIGINAL_CLASSES) else "PTB-XL Rendered"
    print(f"  [{i:>2}] [{src}] {name}")


# =============================================================================
# COLLECT IMAGE PATHS & LABELS
# =============================================================================

file_paths = []
labels     = []
src_counts = {c: {"orig": 0, "ptbxl": 0} for c in class_names}

for class_index, class_name in enumerate(class_names):

    # -- Original scanned image folders (first 4 classes only) ----------------
    if class_index < len(ORIGINAL_CLASSES):
        orig_dir = os.path.join(DATASET_DIR, class_name)
        if os.path.exists(orig_dir):
            for fname in os.listdir(orig_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    file_paths.append(os.path.join(orig_dir, fname))
                    labels.append(class_index)
                    src_counts[class_name]["orig"] += 1

    # -- PTB-XL rendered folders (all 11 new classes) -------------------------
    rendered_dir = os.path.join(RENDERED_DIR, class_name)
    if os.path.exists(rendered_dir):
        for fname in os.listdir(rendered_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                file_paths.append(os.path.join(rendered_dir, fname))
                labels.append(class_index)
                src_counts[class_name]["ptbxl"] += 1

# Dataset composition table
print(f"\n{'-'*72}")
print(f"  {'Class':<47} {'Orig':>6} {'PTB-XL':>8} {'Total':>7}")
print(f"{'-'*72}")
for c in class_names:
    orig  = src_counts[c]["orig"]
    ptb   = src_counts[c]["ptbxl"]
    total = orig + ptb
    print(f"  {c[:47]:<47} {orig:>6} {ptb:>8} {total:>7}")
print(f"{'-'*72}")
print(f"  {'TOTAL':<47} "
      f"{sum(s['orig']  for s in src_counts.values()):>6} "
      f"{sum(s['ptbxl'] for s in src_counts.values()):>8} "
      f"{len(file_paths):>7}")

if len(file_paths) == 0:
    print("\n[ERROR] No images found. Run dataset_builder.py first to render PTB-XL images.")
    exit(1)

# Save class order for predict.py
with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))


# =============================================================================
# TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    file_paths, labels,
    test_size=VAL_TEST_SPLIT,
    stratify=labels,
    random_state=RANDOM_STATE,
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=RANDOM_STATE,
)

print(f"\nStratified Splits:")
print(f"  Train      : {len(train_paths):>6} images")
print(f"  Validation : {len(val_paths):>6} images")
print(f"  Test       : {len(test_paths):>6} images")


# =============================================================================
# BALANCED CLASS WEIGHTS
# =============================================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_array)}

print("\nComputed Class Weights (balanced):")
for i, w in class_weights.items():
    print(f"  [{i:>2}] {class_names[i][:50]:<50} : {w:.3f}")


# =============================================================================
# DATA PIPELINE
# =============================================================================

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

augment_layer = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.02),
    tf.keras.layers.RandomZoom(0.04),
    tf.keras.layers.RandomBrightness(0.06),
])


def py_load_and_crop(path_tensor):
    path = path_tensor.numpy().decode("utf-8")
    try:
        img = load_and_crop(path, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return img.astype(np.float32)


def tf_load_image(path, label):
    img = tf.py_function(py_load_and_crop, [path], tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img, label


def make_dataset(paths, lbs, augment=False, shuffle=False):
    if shuffle:
        combined = list(zip(paths, lbs))
        random.seed(RANDOM_STATE)
        random.shuffle(combined)
        paths, lbs = zip(*combined)
        paths, lbs = list(paths), list(lbs)

    ds = tf.data.Dataset.from_tensor_slices((paths, lbs))
    if shuffle:
        ds = ds.shuffle(buffer_size=512, seed=RANDOM_STATE)
    ds = ds.map(tf_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(
            lambda x, y: (augment_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths, train_labels, augment=True, shuffle=True)
val_ds   = make_dataset(val_paths,   val_labels)
test_ds  = make_dataset(test_paths,  test_labels)


# =============================================================================
# MODEL ARCHITECTURE - MobileNetV2 + Custom Head (15-class)
# =============================================================================

print(f"\n{'='*55}")
print("Building MobileNetV2 (15-class) Architecture")
print(f"{'='*55}")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
print(f"Output classes: {len(class_names)}")
print(f"Trainable params (head only): {model.count_params():,}")


# =============================================================================
# STAGE 1 - CLASSIFICATION HEAD WARMUP
# =============================================================================

print(f"\n{'='*55}")
print("STAGE 1 - Classification Head Warmup")
print(f"{'='*55}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb_s1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    ),
]

history1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=cb_s1,
)


# =============================================================================
# STAGE 2 - DEEP LAYER FINE-TUNING
# =============================================================================

print(f"\n{'='*55}")
print("STAGE 2 - Deep Layer Fine-Tuning")
print(f"{'='*55}")

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfrozen MobileNetV2 layers: {trainable}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb_s2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
    ),
]

history2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=cb_s2,
)


# =============================================================================
# EVALUATION
# =============================================================================

print(f"\n{'='*55}")
print("FINAL TEST SET EVALUATION")
print(f"{'='*55}")

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array(test_labels[:len(y_pred)])

short_names = [
    "MI", "History-MI", "Abnormal-HB", "Normal",
    "AFIB", "A-Flutter", "S-Tach", "S-Brad",
    "CLBBB", "CRBBB", "IRBBB", "1AVB",
    "LAFB", "LVH", "LAE",
]

report = classification_report(y_true, y_pred, target_names=short_names, zero_division=0)
print("\nClassification Report:")
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write("15-CLASS ECG DISORDER CLASSIFICATION REPORT\n")
    f.write("=" * 55 + "\n\n")
    f.write(report)

# -- Confusion Matrix ----------------------------------------------------------
cm   = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(15, 13))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_names)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("15-Class ECG Disorder Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"\n[+] Confusion matrix saved to: {cm_path}")

# -- Training Curves -----------------------------------------------------------
train_acc = history1.history["accuracy"]     + history2.history["accuracy"]
val_acc   = history1.history["val_accuracy"] + history2.history["val_accuracy"]
train_los = history1.history["loss"]         + history2.history["loss"]
val_los   = history1.history["val_loss"]     + history2.history["val_loss"]
ft_start  = len(history1.history["accuracy"]) - 1

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(train_acc, color="#00E676", linewidth=2, label="Train")
axes[0].plot(val_acc,   color="#00B0FF", linewidth=2, label="Val")
axes[0].axvline(ft_start, linestyle="--", color="#FF5252", label="Fine-Tune Start")
axes[0].set_title("Model Accuracy", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(train_los, color="#00E676", linewidth=2, label="Train")
axes[1].plot(val_los,   color="#00B0FF", linewidth=2, label="Val")
axes[1].axvline(ft_start, linestyle="--", color="#FF5252", label="Fine-Tune Start")
axes[1].set_title("Model Loss", fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=150)
plt.close(fig)
print(f"[+] Training curves saved to:  {curves_path}")

# -- Save Model ----------------------------------------------------------------
model.save("model.keras")
print(f"\n[+] Model saved to: model.keras")
print(f"[+] Class names ({len(class_names)}) saved to: class_names.txt")
print("\nTraining complete!")
