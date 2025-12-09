"""
OCR engine module using open-source OCR tools.
Supports multiple OCR engines for better accuracy.
"""

import pytesseract
import easyocr
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader (lazy loading)
_easyocr_reader = None


def get_easyocr_reader():
    """Lazy load EasyOCR reader to avoid repeated initialization."""
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Initializing EasyOCR reader...")
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader


def extract_text_tesseract(image, config='--psm 6'):
    """
    Extract text using Tesseract OCR.
    
    Args:
        image: Preprocessed image as numpy array
        config: Tesseract configuration
        
    Returns:
        List of tuples (text, confidence, bbox)
    """
    try:
        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        
        texts = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 0:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                texts.append((text, conf, (x, y, w, h)))
        
        return texts
        
    except Exception as e:
        logger.error(f"Tesseract OCR error: {str(e)}")
        return []


def extract_text_easyocr(image):
    """
    Extract text using EasyOCR.
    
    Args:
        image: Preprocessed image as numpy array
        
    Returns:
        List of tuples (text, confidence, bbox)
    """
    try:
        reader = get_easyocr_reader()
        results = reader.readtext(image)
        
        texts = []
        for (bbox, text, confidence) in results:
            # Convert confidence to percentage
            conf = int(confidence * 100)
            # Extract bounding box coordinates
            x = int(min([point[0] for point in bbox]))
            y = int(min([point[1] for point in bbox]))
            w = int(max([point[0] for point in bbox]) - x)
            h = int(max([point[1] for point in bbox]) - y)
            
            texts.append((text.strip(), conf, (x, y, w, h)))
        
        return texts
        
    except Exception as e:
        logger.error(f"EasyOCR error: {str(e)}")
        return []


def extract_text_combined(image, use_easyocr=True, use_tesseract=True):
    """
    Extract text using multiple OCR engines and combine results.
    
    Args:
        image: Preprocessed image as numpy array
        use_easyocr: Whether to use EasyOCR
        use_tesseract: Whether to use Tesseract
        
    Returns:
        Dictionary with OCR results from different engines
    """
    results = {}
    
    if use_tesseract:
        try:
            tesseract_results = extract_text_tesseract(image)
            results['tesseract'] = tesseract_results
        except Exception as e:
            logger.warning(f"Tesseract failed: {str(e)}")
            results['tesseract'] = []
    
    if use_easyocr:
        try:
            easyocr_results = extract_text_easyocr(image)
            results['easyocr'] = easyocr_results
        except Exception as e:
            logger.warning(f"EasyOCR failed: {str(e)}")
            results['easyocr'] = []
    
    return results


def extract_all_text(image, engine='combined'):
    """
    Extract all text from image using specified OCR engine.
    
    Args:
        image: Preprocessed image
        engine: OCR engine to use ('tesseract', 'easyocr', or 'combined')
        
    Returns:
        List of text lines with their positions
    """
    if engine == 'tesseract':
        results = extract_text_tesseract(image)
        return [text for text, conf, bbox in results]
    elif engine == 'easyocr':
        results = extract_text_easyocr(image)
        return [text for text, conf, bbox in results]
    else:  # combined
        ocr_results = extract_text_combined(image)
        
        # Combine results from both engines
        all_texts = []
        if 'tesseract' in ocr_results:
            all_texts.extend([text for text, conf, bbox in ocr_results['tesseract']])
        if 'easyocr' in ocr_results:
            all_texts.extend([text for text, conf, bbox in ocr_results['easyocr']])
        
        # Remove duplicates and return unique text lines
        unique_texts = []
        seen = set()
        for text in all_texts:
            if text.lower() not in seen:
                unique_texts.append(text)
                seen.add(text.lower())
        
        return unique_texts


