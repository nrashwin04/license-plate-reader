# 🚗 License Plate Reader (ANPR Pipeline)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Ready-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)

An end-to-end Automated Number Plate Recognition (ANPR) pipeline built as part of an MSc Computer Science portfolio. This project demonstrates a modular approach to computer vision — separating detection, preprocessing, character recognition, and domain-specific post-processing into distinct, scalable layers.

---

## ✨ Features

- **Modular Architecture** — built with separation of concerns in mind, allowing the detection or OCR engine to be swapped without touching the rest of the pipeline
- **Robust Image Preprocessing** — OpenCV contrast normalisation, median blurring, and cubic upscaling to recover details from compressed or noisy camera feeds
- **Domain-Specific Post-Processing** — custom parser tailored for Indian High-Security Registration Plates (HSRP), enforcing the strict `[2 Letters][2 Numbers][1-2 Letters][4 Numbers]` format
- **Intelligent Error Correction** — automatically corrects common OCR hallucinations (e.g. `0` → `O`, `5` → `S`) and snaps misread state codes back to valid RTO formats
- **Interactive UI** — persistent Streamlit interface that visualises every step of the CV pipeline for easy debugging and demonstration

---

## 🧠 Architecture

The application is broken into four core modules:

| Module | Responsibility |
|---|---|
| `app.py` | Orchestration, session state, Streamlit UI |
| `detector.py` | Plate localisation using OpenCV contour detection and aspect-ratio validation |
| `utils.py` | Image preprocessing — upscaling, blurring, contrast normalisation |
| `ocr.py` | EasyOCR inference, post-processing, and Indian plate format correction |

### Pipeline flow

```
Vehicle image → contour detection → plate crop
    → 2x cubic upscale → median blur → contrast normalisation
    → EasyOCR inference → format correction → result
```

---

## 🔍 Engineering Decisions & Honest Constraints

### What works well

The preprocessing pipeline performs reliably. Cropped plate images are upscaled, denoised, and contrast-normalised before OCR — producing clean, legible input even from compressed camera feeds.

Domain-specific post-processing squeezes additional accuracy out of the lightweight model by enforcing the known Indian plate format and correcting character-level mistakes based on positional rules.

### Known limitation — EasyOCR and the HSRP font

Despite clean preprocessed input, EasyOCR occasionally misreads characters on Indian High-Security Registration Plates. For example:

- `MH04KW7706` read as `MH01KL2206`
- `MH20DV2366` read as `MHHZODV2366` — a metal rivet on the plate was interpreted as the letter `H`

This is a fundamental constraint of EasyOCR — a lightweight, generalised model optimised for CPU speed that has not been specifically trained on the HSRP font, which features open-top `4`s, squarish `0`s, and metallic rivets that confuse the neural network even on perfectly clean input.

### Why the modular architecture matters

Because each stage is isolated, upgrading to an enterprise-grade OCR engine requires fewer than 10 lines of change in `ocr.py` — the detection, preprocessing, and post-processing logic remain fully intact:

```python
# Current — lightweight, runs on CPU
reader = easyocr.Reader(['en'])

# Drop-in upgrades:
# → PaddleOCR       (open source, higher HSRP accuracy)
# → Google Cloud Vision API  (99%+ accuracy)
# → AWS Textract
```

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/license-plate-reader.git
cd license-plate-reader
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use it
- Upload a vehicle image (jpg, png)
- Adjust the confidence threshold slider as needed
- Check "Show preprocessing steps" to see the full CV pipeline breakdown
- View the extracted plate text and raw OCR results

---

## 📁 Project structure

```
license-plate-reader/
├── app.py              # Streamlit UI and orchestration
├── detector.py         # OpenCV plate localisation
├── ocr.py              # EasyOCR inference and post-processing
├── utils.py            # Image preprocessing helpers
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

- [x] OpenCV contour-based plate detection
- [x] EasyOCR integration with allowlist
- [x] Indian HSRP post-processing and error correction
- [x] Streamlit UI with preprocessing visualisation
- [ ] PaddleOCR upgrade for improved HSRP font accuracy
- [ ] YOLOv8-based detection for angled or distant plates
- [ ] Multi-plate detection in a single image
- [ ] Deploy to Streamlit Community Cloud

---

*Built with OpenCV + EasyOCR · MSc CS portfolio project*
