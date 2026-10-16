import os
import io
import re
import urllib.request
import pyarrow.parquet as pq
import yaml
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple


HF_DATASET_REPO = "Ahus-AIM/Open-ECG-Digitizer-Development-Dataset"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/data"

VAL_SHARDS = [
    "val-00000-of-00004.parquet",
    "val-00001-of-00004.parquet",
    "val-00002-of-00004.parquet",
    "val-00003-of-00004.parquet",
]

# Train has 68 shards: train-00000-of-00068.parquet to train-00067-of-00068.parquet
TRAIN_SHARDS = [f"train-{i:05d}-of-00068.parquet" for i in range(68)]


def get_shard_download_url(filename: str) -> str:
    """Returns the direct download URL for a shard."""
    return f"{HF_BASE_URL}/{filename}"


def download_shard_file(
    filename: str,
    output_dir: str = "dataset_cache",
    progress_callback: Optional[Any] = None,
) -> str:
    """Downloads a parquet shard from Hugging Face if not already present."""
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100000:
        return dest_path

    url = get_shard_download_url(filename)
    import subprocess
    cmd = ["curl.exe", "-L", "-o", dest_path, url]
    subprocess.run(cmd, check=True)
    return dest_path


def extract_samples_from_parquet(
    parquet_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    use_t0_transform: bool = False,
    progress_callback: Optional[Any] = None,
) -> int:
    """Extracts scan PNG and mask PNG files from a Hugging Face parquet shard.
    
    Each sample in the dataset contains:
      - 'id': filename ID
      - 'img' / 'img_T0': scan PIL image or dict with 'bytes'
      - 'mask' / 'mask_T0': segmentation mask image or dict with 'bytes'
    """
    os.makedirs(output_dir, exist_ok=True)
    pf = pq.ParquetFile(parquet_path)
    
    img_col = "img_T0" if use_t0_transform else "img"
    mask_col = "mask_T0" if use_t0_transform else "mask"
    
    saved_count = 0
    total_in_file = pf.metadata.num_rows
    limit = min(total_in_file, max_samples) if max_samples else total_in_file

    for rg_idx in range(pf.num_row_groups):
        if max_samples and saved_count >= max_samples:
            break
        table = pf.read_row_group(rg_idx, columns=["id", img_col, mask_col])
        df = table.to_pandas()
        
        for _, row in df.iterrows():
            if max_samples and saved_count >= max_samples:
                break
                
            sample_id = str(row["id"])
            # Remove any unwanted path components in ID
            sample_id = os.path.basename(sample_id).split(".")[0]
            
            img_data = row[img_col]
            mask_data = row[mask_col]
            
            # Save scan image
            img_path = os.path.join(output_dir, f"{sample_id}.png")
            if not os.path.exists(img_path):
                if isinstance(img_data, dict) and "bytes" in img_data:
                    with open(img_path, "wb") as f:
                        f.write(img_data["bytes"])
                elif isinstance(img_data, (bytes, bytearray)):
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                elif hasattr(img_data, "save"):
                    img_data.save(img_path)
                elif isinstance(img_data, dict) and "path" in img_data and img_data.get("bytes") is None:
                    continue

            # Save mask image
            mask_path = os.path.join(output_dir, f"{sample_id}_mask.png")
            if not os.path.exists(mask_path):
                if isinstance(mask_data, dict) and "bytes" in mask_data:
                    with open(mask_path, "wb") as f:
                        f.write(mask_data["bytes"])
                elif isinstance(mask_data, (bytes, bytearray)):
                    with open(mask_path, "wb") as f:
                        f.write(mask_data)
                elif hasattr(mask_data, "save"):
                    mask_data.save(mask_path)
                elif isinstance(mask_data, dict) and "path" in mask_data and mask_data.get("bytes") is None:
                    continue

            saved_count += 1
            if progress_callback:
                progress_callback(saved_count, limit)

    return saved_count


def update_unet_yaml(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    device: str = "cuda",
    compile_model: bool = False,
    output_yaml: str = "src/config/unet.yml",
) -> None:
    """Updates the unet.yml configuration with custom dataset paths and training options."""
    if test_dir is None:
        test_dir = val_dir

    train_dir = os.path.abspath(train_dir).replace("\\", "/")
    val_dir = os.path.abspath(val_dir).replace("\\", "/")
    test_dir = os.path.abspath(test_dir).replace("\\", "/")

    with open(output_yaml, "r") as f:
        content = f.read()

    # Update dataset paths
    content = re.sub(
        r"(TRAIN:\s+KWARGS:\s+data_path:\s*)[^\n]+",
        r"\g<1>'" + train_dir + "'",
        content,
    )
    content = re.sub(
        r"(VAL:\s+KWARGS:\s+data_path:\s*)[^\n]+",
        r"\g<1>'" + val_dir + "'",
        content,
    )
    content = re.sub(
        r"(TEST:\s+KWARGS:\s+data_path:\s*)[^\n]+",
        r"\g<1>'" + test_dir + "'",
        content,
    )

    # Set COMPILE: False for Windows compatibility
    content = re.sub(
        r"(COMPILE:\s*)[A-Za-z]+",
        r"\g<1>" + str(compile_model),
        content,
    )

    # Update epochs
    content = re.sub(
        r"(epochs:\s*)\d+",
        r"\g<1>" + str(epochs),
        content,
    )

    # Update batch_size
    content = re.sub(
        r"(batch_size:\s*)\d+",
        r"\g<1>" + str(batch_size),
        content,
    )

    # Update learning rate
    content = re.sub(
        r"(lr:\s*)[0-9.eE\-]+",
        r"\g<1>" + str(learning_rate),
        content,
    )

    # Update device
    content = re.sub(
        r"(device:\s*'[a-zA-Z0-9:]+')",
        f"device: '{device}'",
        content,
    )

    # num_workers: on Windows 2 or 0 is recommended
    content = re.sub(
        r"(num_workers:\s*)\d+",
        r"\g<1>2",
        content,
    )

    with open(output_yaml, "w") as f:
        f.write(content)
