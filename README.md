# 🌿 Plant Disease Detection Using DenseNet

A deep learning system that identifies plant diseases from leaf photographs and provides treatment recommendations. Built with DenseNet121 (transfer learning) and served via a Flask web application.

## Demo

![Demo](PlantDiseaseDetectionDemo .gif)

---

## What It Does

Upload a photograph of a plant leaf. The model identifies:
- Which crop the leaf belongs to
- Whether the plant is healthy or diseased
- If diseased, what disease it has
- What treatment to apply

It covers **44 conditions** across **14 crops** — including tomato, potato, apple, grape, corn, tea, and more.

---

## Supported Crops and Diseases

| Crop | Diseases Detected |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tea | Algal Spot, Brown Blight, Gray Blight, Helopeltis, Red Spot, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## How It Works

```
User uploads leaf image
         │
         ▼
Resize to 224×224 px → Normalise pixel values (÷255)
         │
         ▼
DenseNet121 (pre-trained on ImageNet)
   └── GlobalAveragePooling2D
   └── Dense(44, softmax)
         │
         ▼
Predicted disease class + confidence
         │
         ▼
Look up treatment recommendation from cure dictionary
         │
         ▼
Return: disease name + treatment advice
```

**Why DenseNet121?**
DenseNet (Densely Connected Convolutional Network) connects each layer to every other layer, which helps with gradient flow and feature reuse. This makes it effective for image classification even with limited training data through transfer learning from ImageNet weights.

---

## Model Architecture

| Component | Detail |
|---|---|
| Base model | DenseNet121 — ImageNet pretrained weights |
| Input size | 224 × 224 × 3 (RGB) |
| Added layers | GlobalAveragePooling2D → Dense(44, softmax) |
| Training strategy | Two-phase: freeze base → train head, then unfreeze → fine-tune |
| Data augmentation | Horizontal flip, zoom, rotation, width/height shift |
| Callbacks | EarlyStopping (patience=5), ReduceLROnPlateau |
| Train/Val split | 80% / 20% (stratified) |
| Output classes | 44 |

---

## Dataset

Two Kaggle datasets combined:

| Dataset | Contents | Link |
|---|---|---|
| Leaf Disease Detection Dataset | 14 crops, common diseases, 87,000+ images | [Kaggle](https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset/data) |
| Tea Leaf Disease Dataset | 6 tea-specific conditions | [Kaggle](https://www.kaggle.com/datasets/saikatdatta1994/tea-leaf-disease) |

The model was trained on Google Colab with the dataset stored in Google Drive.

---

## Project Structure

```
PlantDiseaseDetection/
├── PlantDiseaseDetectionUsingDenseNet_Final.ipynb  # Training notebook (run on Google Colab)
├── Flask.ipynb                                     # Flask web app for inference
├── leaf_detection_densenet_model.h5                # Trained model weights
├── PlantDiseaseDetectionDemo.gif                   # Demo
├── Documentation_Aishwarya.pdf                     # Full project documentation
└── README.md
```

---

## Running Locally

### Prerequisites

```bash
pip install tensorflow flask numpy pillow
```

> Tested on Python 3.11.7. Run `Flask.ipynb` in Jupyter or convert to `app.py`.

### Step 1 — Make sure the model file is in the same folder

```
leaf_detection_densenet_model.h5
Flask.ipynb  (or app.py if converted)
templates/
  welcome.html
  main.html
uploads/     (created automatically)
```

### Step 2 — Run the Flask app

Open `Flask.ipynb` in Jupyter Notebook and run all cells. The server starts at:

```
http://127.0.0.1:5000
```

Or convert to a standalone script first:

```bash
jupyter nbconvert --to script Flask.ipynb
python Flask.py
```

### Step 3 — Use the app

1. Go to `http://127.0.0.1:5000`
2. Click through to the main upload page
3. Upload a leaf photograph (JPG or PNG)
4. See the predicted disease and treatment recommendation

---

## Training Your Own Model

Open `PlantDiseaseDetectionUsingDenseNet_Final.ipynb` in **Google Colab**.

The notebook expects the dataset to be in Google Drive:

```python
zip_path = '/content/drive/My Drive/dataset.zip'
```

Steps the notebook runs:
1. Mount Google Drive and unzip dataset
2. Build DataFrames of image paths and labels
3. 80/20 stratified train/validation split
4. ImageDataGenerator with augmentation
5. Load DenseNet121 with ImageNet weights, freeze base layers
6. Train the classification head (Dense + GlobalAveragePooling2D)
7. Unfreeze base layers and fine-tune at lower learning rate
8. Evaluate with classification report and confusion matrix
9. Save model as `.h5`

---

## Treatment Recommendations

The app returns a specific treatment recommendation for each disease. Examples:

| Disease | Recommendation |
|---|---|
| Tomato Early Blight | Apply fungicides and practice crop rotation |
| Tomato Late Blight | Apply fungicides and remove infected plants |
| Potato Late Blight | Apply fungicides and destroy infected plants |
| Tea Helopeltis | Spray neem oil mixed with water and a small amount of soap directly on affected plants |
| Orange Haunglongbing | Remove infected trees and control psyllid populations |
| Apple Apple Scab | Use fungicides and remove infected leaves |
| Grape Esca (Black Measles) | Prune and remove infected wood |
| Any healthy plant | No action needed — the plant is healthy |

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning framework | TensorFlow / Keras |
| Model architecture | DenseNet121 (transfer learning) |
| Web framework | Flask |
| Training environment | Google Colab |
| Image preprocessing | Keras ImageDataGenerator, OpenCV |
| Evaluation | scikit-learn (classification report, confusion matrix) |
| Visualisation | Matplotlib, Seaborn |

---

## Limitations

- Works best with clear, well-lit photographs of individual leaves
- Not suitable for diagnosing crops not in the 44-class list
- Model was trained on controlled dataset images — performance on photos taken in varied field lighting may differ
- The `.h5` model file (~100MB+) is required to run inference; it is not included in the GitHub repo due to size — download the pre-trained model separately or retrain using the notebook

---

## Dataset Credits

- [Leaf Disease Detection Dataset](https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset/data) — Kaggle
- [Tea Leaf Disease Dataset](https://www.kaggle.com/datasets/saikatdatta1994/tea-leaf-disease) — Kaggle

Based on the [PlantVillage Dataset](https://plantvillage.psu.edu/) (Penn State University).

---

## Author

**Aishwarya** — Built as part of Infosys Springboard Internship, 2024.
