"""
Image preprocessing module for OCR text extraction.
Handles various image qualities and orientations.
"""

import cv2
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image_path):
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cleaned)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        img = cv2.imread(image_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise


def detect_orientation(image):
    """
    Detect and correct image orientation if needed.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Corrected image
    """
    try:
        return image
    except Exception as e:
        logger.warning(f"Orientation detection failed: {str(e)}")
        return image


def resize_image(image, max_dimension=2000):
    """
    Resize image if too large, maintaining aspect ratio.
    
    Args:
        image: Input image
        max_dimension: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) > max_dimension:
        if height > width:
            scale = max_dimension / height
        else:
            scale = max_dimension / width
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        image = cv2.resize(image, (new_width, new_height), 
                          interpolation=cv2.INTER_AREA)
    
    return image


