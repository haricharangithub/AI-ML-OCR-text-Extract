from dataclasses import dataclass
from typing import Dict, List, Optional

try:  # Avoid import-time hard failure when cv2 is unavailable.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
import pytesseract


@dataclass
class OcrConfig:
    lang: str = "eng"
    psm: int = 6
    oem: int = 3
    tess_cmd: Optional[str] = None


class OcrEngine:
    def __init__(self, config: Optional[OcrConfig] = None) -> None:
        self.config = config or OcrConfig()
        if self.config.tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tess_cmd

    def run(self, image) -> List[Dict[str, str]]:
        """Run OCR and return list of line dictionaries with text and confidence."""
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for OCR preprocessing but is not installed correctly.")
        custom_config = f"--oem {self.config.oem} --psm {self.config.psm}"
        data = pytesseract.image_to_data(
            image,
            lang=self.config.lang,
            output_type=pytesseract.Output.DICT,
            config=custom_config,
        )
        lines: List[Dict[str, str]] = []
        num_items = len(data["text"])
        for i in range(num_items):
            text = data["text"][i].strip()
            conf = float(data["conf"][i]) if data["conf"][i] not in ("", "-1") else 0.0
            if not text:
                continue
            line_info = {
                "text": text,
                "conf": conf,
                "line_num": data.get("line_num", [None])[i],
                "block_num": data.get("block_num", [None])[i],
            }
            lines.append(line_info)
        return lines

