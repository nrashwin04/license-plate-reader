import cv2
import numpy as np
from PIL import Image, ImageDraw
import base64
from io import BytesIO

def preprocess_plate(image: Image.Image):
    """
    Preprocess cropped plate image for OCR.
    Aggressive binarization often destroys text, so we just grayscale and upscale.
    """
    img_arr = np.array(image)
    
    # Convert to grayscale
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_arr
        
    # Upscale by 2x to give the AI enough pixels to read comfortably
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    # Contrast Normalization: Stretches the image's colors so the darkest 
    # pixel is 0 (pure black) and brightest is 255 (pure white).
    # This fixes shadows without destroying the letters like Binarization does.
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # A very slight Gaussian blur to smooth out the edges and reduce noise
    final_img = cv2.GaussianBlur(normalized, (3, 3), 0)
        
    return Image.fromarray(final_img)

def draw_plate_box(original_image: Image.Image, cropped_region: Image.Image, found: bool):
    """
    Draws a green rectangle around the plate region if found,
    otherwise draws a red border around the full image.
    """
    img_copy = original_image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img_copy.size
    
    if found and cropped_region is not None:
        # Use template matching to locate the cropped region within the original image
        orig_arr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        crop_arr = cv2.cvtColor(np.array(cropped_region), cv2.COLOR_RGB2BGR)
        
        # Ensure dimensions are valid for template matching
        if crop_arr.shape[0] <= orig_arr.shape[0] and crop_arr.shape[1] <= orig_arr.shape[1]:
            result = cv2.matchTemplate(orig_arr, crop_arr, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            top_left = max_loc
            h, w = crop_arr.shape[:2]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            draw.rectangle([top_left, bottom_right], outline="green", width=5)
        else:
            # Fallback if dimensions are weird
            draw.rectangle([0, 0, width-1, height-1], outline="green", width=10)
    else:
        # If not found, red border around the whole image
        draw.rectangle([0, 0, width-1, height-1], outline="red", width=10)
        
    return img_copy

def pil_to_base64(image: Image.Image):
    """
    Convert PIL image to base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
