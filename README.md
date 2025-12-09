# OCR Text Extraction for Shipping Waybills

This repository implements the assessment requirements for building an OCR-based text extraction system that pulls the full text line containing the pattern `_1` from shipping label / waybill images.

## Project Structure
```
project-root/
├── README.md                 # Comprehensive documentation
├── requirements.txt          # Dependencies (Python 3.7+)
├── src/
│   ├── ocr_engine.py         # Core OCR logic
│   ├── preprocessing.py      # Image preprocessing
│   ├── text_extraction.py    # Target text extraction
│   └── utils.py              # Utility functions
├── app.py                    # Streamlit demonstration app
├── tests/                    # Test cases
├── notebooks/                # Jupyter notebooks (placeholder)
└── results/                  # Sample outputs and accuracy metrics
```

## Setup
1. Ensure Python 3.7+ is available.
2. Install system Tesseract OCR (open source, allowed):  
   - Windows: https://github.com/tesseract-ocr/tesseract  
   - Add the installation path to your `PATH` or set `TESSERACT_CMD` env var.
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Streamlit demo
```
streamlit run app.py
```
Features:
- Upload waybill/label images
- Run preprocessing + OCR
- Display extracted `_1` line with confidence and full OCR text

### Batch extraction to JSON
```
python -m src.text_extraction --images \"ReverseWay Bill\" --output results/outputs.json
```
This writes a JSON array of results (one entry per image) to `results/outputs.json`. The same module can be pointed at any directory of test images.

## Technical Approach
- **OCR engine:** Tesseract via `pytesseract` (open-source, offline). Configurable PSM/OEM and language; raises actionable errors if the binary is missing.
- **Preprocessing:** Resize + grayscale, denoise, adaptive thresholding, and deskew using image moments to stabilise text orientation.
- **Extraction logic:** Collect OCR lines with confidence, normalise whitespace, then select the line containing the `_1` pattern using a robust regex (handles `163233702292313922_1_lWV`-style strings).
- **Accuracy focus:** Emphasis on cleaning (noise/contrast) and deskew before OCR; modular design so stronger models or fine-tuned pipelines can be dropped in without API changes.

## Tests
Run unit tests:
```
python -m pytest
```
Tests cover the `_1` line detection and whitespace normalisation.

## Results & Reporting
- Place generated JSON outputs and any accuracy calculations in `results/`.
- `results/README.md` documents how to regenerate outputs.
- Add OCR screenshots per test image to `results/` when produced.

## Challenges & Next Steps
- Accuracy depends on the installed OCR model; consider fine-tuning Tesseract configs, adding language packs, or swapping in lightweight deep-learning OCR (e.g., EasyOCR) while keeping the same interfaces.
- Additional robustness can come from rotation detection, contrast-limited adaptive histogram equalisation (CLAHE), and heuristic cropping around barcodes / tracking numbers.

