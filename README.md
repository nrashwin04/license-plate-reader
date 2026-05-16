# 🚗 License Plate Reader

> 🚧 Work in progress — coming soon.

Automatic license plate detection and text extraction from vehicle images.
Upload any photo of a vehicle and the app detects the plate, crops it, and reads the text using OCR.

> Built as part of my MSc CS portfolio to explore computer vision pipelines and real-world OCR applications.

---

## What it does

1. **Upload** a vehicle image
2. **Detect** the license plate region using OpenCV
3. **Crop** and preprocess the plate for better OCR accuracy
4. **Read** the plate text using EasyOCR
5. **Display** the original image, cropped plate, and extracted text

---

## Planned features

- 📸 Upload any vehicle image (jpg, png)
- 🔍 Automatic plate region detection
- ✂️ Plate cropping and preprocessing
- 🔤 OCR text extraction with confidence score
- 🇮🇳 Support for Indian number plates
- 📦 Multiple plates in a single image
- 🌐 Streamlit web interface

---

## Planned tech stack

| Layer | Tech |
|---|---|
| UI | Streamlit |
| Plate detection | OpenCV / YOLOv8 |
| OCR | EasyOCR |
| Image processing | OpenCV, PIL |
| Deep learning | PyTorch |

---

## How it will work

```
Vehicle image → OpenCV contour detection → plate region crop
    → greyscale + thresholding → EasyOCR → extracted text + confidence
```

---

## Planned project structure

```
license-plate-reader/
├── app.py                  # Streamlit UI
├── detector.py             # plate detection logic
├── ocr.py                  # EasyOCR wrapper
├── utils.py                # image preprocessing helpers
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Roadmap

- [ ] OpenCV contour-based plate detection
- [ ] EasyOCR integration
- [ ] Streamlit UI with image upload
- [ ] YOLOv8 upgrade for better detection
- [ ] Indian number plate support
- [ ] Multi-plate detection
- [ ] Deploy to Streamlit Community Cloud

---

*Planned with OpenCV + EasyOCR · MSc CS portfolio project*
