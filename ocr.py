import easyocr
import numpy as np
from PIL import Image
import streamlit as st
import re

@st.cache_resource
def get_reader():
    # Cache the reader so it doesn't reload each call
    return easyocr.Reader(['en'])

def read_plate(image: Image.Image):
    """
    Reads text from a license plate image.
    Returns (text, confidence, raw_results)
    """
    try:
        reader = get_reader()
    except Exception as e:
        raise RuntimeError("OCR engine failed to load.") from e
    img_arr = np.array(image)
    
    # Re-enable allowlist to prevent total hallucination of symbols,
    # but rely on the new Contrast Normalization in utils to provide a clean image.
    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    results = reader.readtext(img_arr, allowlist=allowlist)
    
    # Filter results where confidence > 0.2
    filtered_results = [res for res in results if res[2] > 0.2]
    
    if not filtered_results:
        return "No text detected", 0.0, []
        
    # Join text segments, uppercase
    text_segments = [res[1] for res in filtered_results]
    raw_text = "".join(text_segments).upper()
    
    # Remove spaces and special characters except hyphen
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text)
    
    # If the OCR picked up the 'IND' text on the side of the plate, remove it
    # so it doesn't break our 9/10 character length check!
    clean_text = clean_text.replace('IND', '')
    
    # Post-process based on common Indian plate formats to fix OCR mistakes
    # Standard format: 2 Letters, 2 Numbers, 1-2 Letters, 4 Numbers (9 or 10 chars total)
    if len(clean_text) in [9, 10]:
        chars = list(clean_text)
        char_to_num = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8', 'S': '5', 'A': '4', 'G': '6'}
        num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S', '4': 'A', '6': 'G'}
        
        # First 2 are always letters
        for i in range(2):
            if chars[i] in num_to_char: chars[i] = num_to_char[chars[i]]
            
        # Snap the first two letters to the closest valid Indian State Code
        valid_states = ["AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"]
        state_code = "".join(chars[:2])
        if state_code not in valid_states:
            # Find the closest match (1 character difference)
            for state in valid_states:
                if (state[0] == state_code[0]) or (state[1] == state_code[1]):
                    chars[0], chars[1] = state[0], state[1]
                    break
                
        # Next 2 are always numbers
        for i in range(2, 4):
            if chars[i] in char_to_num: chars[i] = char_to_num[chars[i]]
                
        # Last 4 are always numbers
        for i in range(len(chars)-4, len(chars)):
            if chars[i] in char_to_num: chars[i] = char_to_num[chars[i]]
                
        # The middle characters are letters
        for i in range(4, len(chars)-4):
            if chars[i] in num_to_char: chars[i] = num_to_char[chars[i]]
            # Special case: Y is often confused with V
            if chars[i] == 'Y': chars[i] = 'V'
            
        clean_text = "".join(chars)
    
    # Calculate average confidence
    confidence = sum(res[2] for res in filtered_results) / len(filtered_results)
    
    return clean_text, confidence, filtered_results
