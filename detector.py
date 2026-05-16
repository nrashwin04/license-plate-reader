import cv2
import numpy as np
from PIL import Image

def detect_plate(image: Image.Image):
    """
    Detects a license plate in an image using edge detection and contours.
    Returns (cropped_plate_pil, found)
    """
    # Convert PIL to numpy array
    img_arr = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to reduce noise while keeping edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Canny edge detection
    edged = cv2.Canny(bfilter, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        # The higher the epsilon (e.g. 15 instead of 10), the more aggressive the approximation
        # We use a dynamic epsilon based on the contour perimeter
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            # Get bounding box to check aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / float(h)
            
            # License plates are rectangular and wider than they are tall.
            # A typical Indian plate aspect ratio is between 2.0 and 5.0.
            # We'll use 1.5 to 6.0 to be safe and account for perspective distortion.
            if 1.5 <= aspect_ratio <= 6.0:
                location = approx
                break
            
    if location is None:
        # Return original image and False if no 4-corner polygon found
        return image, False
        
    # Create mask and extract plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img_arr, img_arr, mask=mask)
    
    # Crop the bounding box
    (x, y) = np.where(mask == 255)
    if len(x) == 0 or len(y) == 0:
        return image, False
        
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = img_arr[topx:bottomx+1, topy:bottomy+1]
    
    # Convert cropped BGR to RGB PIL Image
    if len(cropped.shape) == 3 and cropped.shape[2] == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    else:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
        
    cropped_pil = Image.fromarray(cropped)
    
    return cropped_pil, True

def detect_plate_yolo(image: Image.Image):
    # TODO: implement YOLOv8 detection
    raise NotImplementedError("YOLOv8 detection coming soon")
