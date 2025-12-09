import os
import json
import logging
from typing import Dict, List, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_json_output(data: Dict, output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON output to {output_path}")
    except Exception as e:
        logger.error(f"Error saving JSON: {str(e)}")


def load_json_output(json_path: str) -> Dict:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {str(e)}")
        return {}


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))
        image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in image_files])


def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    if len(predictions) != len(ground_truth):
        logger.warning("Predictions and ground truth have different lengths")
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions) if predictions else 0.0


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return ' '.join(text.lower().split())


def calculate_accuracy_normalized(predictions: List[str], ground_truth: List[str]) -> float:
    if len(predictions) != len(ground_truth):
        logger.warning("Predictions and ground truth have different lengths")
        return 0.0
    
    normalized_pred = [normalize_text(p) for p in predictions]
    normalized_gt = [normalize_text(g) for g in ground_truth]
    
    correct = sum(1 for p, g in zip(normalized_pred, normalized_gt) if p == g)
    return correct / len(normalized_pred) if normalized_pred else 0.0


