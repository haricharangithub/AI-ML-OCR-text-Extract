import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load image from disk as BGR numpy array."""
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for image loading but is not installed correctly.")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img


def to_gray(image: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for image conversion but is not installed correctly.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_json(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)


def list_images(directory: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    paths: List[str] = []
    for root, _, files in os.walk(directory):
        for name in files:
            if Path(name).suffix.lower() in exts:
                paths.append(str(Path(root) / name))
    return sorted(paths)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resize_max_dim(image: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

