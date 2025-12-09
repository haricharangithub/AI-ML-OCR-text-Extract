import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st

from src.ocr_engine import OcrEngine
from src.preprocessing import preprocess
from src.text_extraction import find_target_line, normalize_text
from src.utils import list_images, load_image


st.set_page_config(page_title="Waybill OCR", layout="wide")
st.title("Waybill OCR – _1_ Line Extraction")
st.write("Upload a shipping label/waybill image to extract the full line containing `_1`.")


@st.cache_resource
def get_engine():
    return OcrEngine()


def run_ocr(image_bgr: np.ndarray):
    engine = get_engine()
    preprocessed = preprocess(image_bgr)
    lines = engine.run(preprocessed)
    target = find_target_line(lines)
    return preprocessed, lines, target


def render_results(preprocessed: np.ndarray, lines: List[dict], target: dict):
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB), caption="Preprocessed")

    with col2:
        st.subheader("Extracted Target Line")
        if target:
            st.success(f"{target['text']} (conf: {target.get('conf', 0):.1f})")
        else:
            st.error("No `_1` pattern detected.")

        st.subheader("All OCR Lines")
        for line in lines:
            st.write(f"- {normalize_text(line['text'])} (conf: {line.get('conf', 0):.1f})")


def main():
    sample_dir = Path("ReverseWay Bill")
    sample_files = list_images(str(sample_dir)) if sample_dir.exists() else []

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "bmp"])
    selected_sample = st.selectbox("Or choose a bundled sample", [""] + sample_files)

    image_data = None
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif selected_sample:
        image_data = load_image(selected_sample)

    if image_data is not None:
        preprocessed, lines, target = run_ocr(image_data)
        render_results(preprocessed, lines, target)
    else:
        st.info("Upload an image or pick a sample to start.")


if __name__ == "__main__":
    main()

