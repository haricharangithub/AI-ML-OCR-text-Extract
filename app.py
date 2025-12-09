"""
Streamlit application for OCR text extraction demonstration.
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_image, resize_image
from ocr_engine import extract_text_combined
from text_extraction import extract_target_text, clean_extracted_text
from utils import save_json_output


st.set_page_config(
    page_title="OCR Text Extraction System",
    page_icon="📄",
    layout="wide"
)


st.title("📄 OCR Text Extraction System")
st.markdown("Extract text lines containing the pattern '_1_' from shipping label/waybill images")

# Sidebar
st.sidebar.header("Configuration")
ocr_engine = st.sidebar.selectbox(
    "OCR Engine",
    ["Combined (EasyOCR + Tesseract)", "EasyOCR Only", "Tesseract Only"]
)

# Main content
tab1, tab2 = st.tabs(["Single Image Upload", "Batch Processing"])

with tab1:
    st.header("Upload and Process Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Extract Text", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Preprocess image
                    preprocessed = preprocess_image(temp_path)
                    preprocessed = resize_image(preprocessed)
                    
                    # Display preprocessed image
                    st.subheader("Preprocessed Image")
                    st.image(preprocessed, caption="Preprocessed for OCR", use_container_width=True)
                    
                    # Perform OCR based on selection
                    use_easyocr = ocr_engine in ["Combined (EasyOCR + Tesseract)", "EasyOCR Only"]
                    use_tesseract = ocr_engine in ["Combined (EasyOCR + Tesseract)", "Tesseract Only"]
                    
                    ocr_results = extract_text_combined(preprocessed, use_easyocr=use_easyocr, use_tesseract=use_tesseract)
                    
                    # Extract target text
                    extraction_result = extract_target_text(ocr_results, pattern="_1_")
                    
                    # Display results
                    st.subheader("Extraction Results")
                    
                    if extraction_result['target_text']:
                        st.success(f"✅ Target Text Found!")
                        st.code(extraction_result['target_text'], language=None)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Source Engine", extraction_result['source_engine'])
                        with col2:
                            if extraction_result['confidence']:
                                st.metric("Confidence", f"{extraction_result['confidence']}%")
                    else:
                        st.error("❌ Target pattern '_1_' not found in the image")
                    
                    # Show all OCR text
                    with st.expander("View All Extracted Text"):
                        all_texts = []
                        for engine, texts in ocr_results.items():
                            engine_texts = [text for text, conf, bbox in texts]
                            all_texts.extend(engine_texts)
                        
                        unique_texts = list(set(all_texts))
                        for text in unique_texts:
                            st.text(text)
                    
                    # Show matching lines
                    if extraction_result['all_matching_lines']:
                        with st.expander("View All Matching Lines"):
                            for line in extraction_result['all_matching_lines']:
                                st.text(line)
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.exception(e)

with tab2:
    st.header("Batch Process Images")
    
    input_dir = st.text_input("Input Directory", value="ReverseWay Bill")
    output_dir = st.text_input("Output Directory", value="results")
    
    if st.button("Process All Images", type="primary"):
        if not os.path.exists(input_dir):
            st.error(f"Directory '{input_dir}' does not exist!")
        else:
            with st.spinner("Processing all images..."):
                try:
                    from process_images import process_all_images
                    results = process_all_images(input_dir, output_dir)
                    
                    # Display summary
                    st.success("Processing Complete!")
                    
                    successful = sum(1 for r in results if r.get('target_text'))
                    total = len(results)
                    accuracy = (successful / total * 100) if total > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Images", total)
                    with col2:
                        st.metric("Successful", successful)
                    with col3:
                        st.metric("Accuracy", f"{accuracy:.1f}%")
                    
                    # Display results table
                    st.subheader("Results")
                    import pandas as pd
                    df_data = []
                    for r in results:
                        df_data.append({
                            'Image': r['image_name'],
                            'Target Text': r.get('target_text', 'NOT FOUND'),
                            'Engine': r.get('source_engine', 'N/A'),
                            'Status': '✓' if r.get('target_text') else '✗'
                        })
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in batch processing: {str(e)}")
                    st.exception(e)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**OCR Text Extraction System**")
st.sidebar.markdown("Version 1.0.0")


