import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # Delay hard dependency errors until runtime in CLI/app.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .ocr_engine import OcrEngine
from .preprocessing import preprocess
from .utils import list_images, load_image, save_json


TARGET_REGEX = re.compile(r"[0-9]{5,}[_-]?1[_A-Za-z0-9]*")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def find_target_line(lines: Iterable[Dict[str, str]]) -> Optional[Dict[str, str]]:
    best_match: Optional[Dict[str, str]] = None
    for line in lines:
        cleaned = normalize_text(line["text"])
        if TARGET_REGEX.search(cleaned):
            if best_match is None or line.get("conf", 0) > best_match.get("conf", 0):
                best_match = {**line, "text": cleaned}
    return best_match


def extract_from_image_path(path: str, engine: Optional[OcrEngine] = None) -> Dict[str, object]:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for OCR preprocessing but is not installed correctly.")
    engine = engine or OcrEngine()
    image = load_image(path)
    preprocessed = preprocess(image)
    ocr_lines = engine.run(preprocessed)
    target = find_target_line(ocr_lines)
    return {
        "image": path,
        "target_text": target["text"] if target else None,
        "confidence": target.get("conf") if target else None,
        "lines": [normalize_text(l["text"]) for l in ocr_lines],
    }


def batch_extract(image_dir: str, output_path: str) -> List[Dict[str, object]]:
    engine = OcrEngine()
    records: List[Dict[str, object]] = []
    for path in list_images(image_dir):
        records.append(extract_from_image_path(path, engine))
    save_json(records, output_path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract _1 lines from waybill images.")
    parser.add_argument("--images", required=True, help="Directory containing images.")
    parser.add_argument("--output", required=True, help="Path to JSON output.")
    args = parser.parse_args()
    batch_extract(args.images, args.output)


if __name__ == "__main__":
    main()

