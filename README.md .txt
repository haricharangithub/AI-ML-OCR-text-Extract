# OCR Text Extraction – Shipping Label _1_ Line

## Project Overview

This project implements an OCR-based text extraction pipeline for shipping label images.  
The goal is to extract the complete text line that contains the pattern `_1_` from each image with high accuracy.

The solution uses an open-source OCR engine (Tesseract via `pytesseract`) and a modular Python codebase with a simple Streamlit frontend.

## Features

- Image preprocessing using OpenCV
- Line-level OCR using Tesseract
- Robust text-line selection for the pattern `_1_`
- JSON output per image
- Simple Streamlit demo to upload an image and see results
- Basic tests to validate the pipeline

## Directory Structure

project-root/  
├── README.md  
├── requirements.txt  
├── src/  
│   ├── ocr_engine.py  
│   ├── preprocessing.py  
│   ├── text_extraction.py  
│   └── utils.py  
├── app.py  
├── tests/  
│   └── test_pipeline.py  
└── results/  
    └── sample_result.json

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
