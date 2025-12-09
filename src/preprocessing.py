from typing import Tuple

try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
import numpy as np

from .utils import resize_max_dim, to_gray


def denoise(image: np.ndarray, diameter: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    """Bilateral filter preserves edges while reducing noise."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def adaptive_threshold(image_gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )


def deskew(image_gray: np.ndarray) -> np.ndarray:
    """Estimate skew angle via image moments and rotate."""
    coords = cv2.findNonZero(255 - image_gray)
    if coords is None:
        return image_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for preprocessing but is not installed correctly.")
    resized = resize_max_dim(image_bgr, max_dim=1800)
    gray = to_gray(resized)
    denoised = denoise(gray)
    thresh = adaptive_threshold(denoised)
    corrected = deskew(thresh)
    return corrected

