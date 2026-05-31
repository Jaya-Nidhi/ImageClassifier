# NeuralEye — Image Classification System

## Features

- **1000-class ImageNet classification** using MobileNetV2 CNN
- **Top-10 predictions** with confidence scores and visual bars
- **Confidence distribution chart** (Chart.js bar chart)
- **Webcam capture** — take a photo directly from your camera
- **Drag & drop** or click-to-browse image upload
- **CSV export** of all results
- **100% client-side** — no server, no data leaves your browser
- **Model pre-loading** for instant results after first visit

---

## How to Run

### Option 1 — Open directly (simplest)
Just double-click `index.html` to open in your browser.


### Option 2 — Local server (recommended)

**Using Python (built-in):**
```bash
cd image-classifier
python3 -m http.server 8080
# Then open http://localhost:8080
```

**Using Node.js (npx):**
```bash
cd image-classifier
npx serve .
# Then open the URL shown
```

**Using VS Code:**
Install the "Live Server" extension, right-click `index.html` → Open with Live Server.

---

## Project Structure

```
image-classifier/
├── index.html          # Main HTML page
├── css/
│   └── style.css       # All styles (dark industrial theme)
├── js/
│   └── app.js          # Core app logic (TF.js + UI)
└── README.md
```

## Tech Stack

 Layer  Technology 
 ML Framework | TensorFlow.js 4.11 |
 Model | MobileNetV2 (ImageNet, 1000 classes) |
 Charting | Chart.js 4.4 |
 Fonts | Syne + Space Mono (Google Fonts) |
 Runtime | 100% browser, no backend |


## How It Works

1. **Image Preprocessing** — Input resized to 224×224, normalized to [0,1] tensor
2. **CNN Feature Extraction** — MobileNetV2 depthwise separable convolutions extract hierarchical features
3. **Softmax Classification** — Final dense layer outputs 1000-class probabilities
4. **Ranked Results** — Top-10 predictions sorted by confidence


## Browser Support

Works in all modern browsers: Chrome, Firefox, Safari, Edge.
Webcam requires HTTPS or localhost.

---
Notes

- First classification loads the MobileNetV2 model (~14 MB) from jsDelivr CDN
- Subsequent classifications reuse the cached model (instant)
- The app loads the model in the background automatically after page load
