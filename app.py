import streamlit as st
from PIL import Image
import pandas as pd
from detector import detect_plate, detect_plate_yolo
from ocr import read_plate
from utils import preprocess_plate, draw_plate_box

st.set_page_config(layout="wide", page_title="License Plate Reader", page_icon="🚗")

# Initialize session state for persistent results
if "last_result" not in st.session_state:
    st.session_state.last_result = None

col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

with col1:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])
    method = st.selectbox("Detection method", ["OpenCV (Contours)", "YOLOv8 (Coming Soon)"])
    conf_threshold = st.slider("OCR confidence threshold", 0.1, 0.9, 0.2, 0.05)
    show_preprocessing = st.checkbox("Show preprocessing steps", value=False)
    process_btn = st.button("Read Plate", type="primary")

with col2:
    st.header("Image Display")
    if uploaded_file is not None:
        # Load uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        if process_btn:
            try:
                # 1. Detection
                if method == "YOLOv8 (Coming Soon)":
                    st.info("YOLOv8 detection is coming soon. Using OpenCV instead.")
                    
                cropped_plate, found = detect_plate(image)
                
                # 2. Preprocessing
                preprocessed_plate = preprocess_plate(cropped_plate)
                
                # 3. OCR
                try:
                    text, confidence, raw_results = read_plate(preprocessed_plate)
                    # Filter OCR results by threshold
                    if confidence < conf_threshold:
                        text = "No text detected"
                except RuntimeError:
                    st.error("OCR engine failed to load. Please refresh the page.")
                    text, confidence, raw_results = "Error", 0.0, []
                
                # 4. Draw Plate Box
                annotated_img = draw_plate_box(image, cropped_plate, found)
                
                # Save to session state
                st.session_state.last_result = {
                    "text": text,
                    "confidence": confidence,
                    "found": found,
                    "annotated_image": annotated_img,
                    "cropped_plate": cropped_plate,
                    "preprocessed_plate": preprocessed_plate,
                    "raw_results": raw_results
                }
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
        
        # Display images based on state
        if st.session_state.last_result and st.session_state.last_result.get("annotated_image"):
            st.image(st.session_state.last_result["annotated_image"], use_container_width=True)
            
            if show_preprocessing:
                st.subheader("Preprocessing Steps")
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.image(st.session_state.last_result["cropped_plate"], caption="Cropped Plate", use_container_width=True)
                with pcol2:
                    st.image(st.session_state.last_result["preprocessed_plate"], caption="Preprocessed Plate (Binarized)", use_container_width=True)
        else:
            # Just show the original if not processed yet
            st.image(image, use_container_width=True)
            
    elif process_btn:
        st.warning("Please upload an image first.")

with col3:
    st.header("Results")
    if st.session_state.last_result:
        res = st.session_state.last_result
        
        # Display large text
        st.markdown(f"<h1 style='text-align:center; letter-spacing:6px; background-color:#1e1e1e; padding:20px; border-radius:10px;'>{res['text']}</h1>",
                    unsafe_allow_html=True)
                    
        st.metric("OCR Confidence", f"{res['confidence']*100:.1f}%")
        
        # Expandable raw results
        with st.expander("Raw OCR Results"):
            if res["raw_results"]:
                data = [{"Text": r[1], "Confidence": f"{r[2]:.4f}"} for r in res["raw_results"]]
                st.table(pd.DataFrame(data))
            else:
                st.write("No raw results available.")
                
        # Status messages
        if res["found"] and res["text"] != "No text detected" and res["text"] != "Error":
            st.success("Plate found and text successfully extracted!")
        elif res["found"]:
            st.warning("Plate region found, but no text detected.")
        else:
            st.error("No plate region found in the image.")
