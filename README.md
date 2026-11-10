# 🚗 Car Damage Detection — MLOps

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

MLOps capstone: Detect car damage severity using CNN with supervised & unsupervised learning. Image blur preprocessing + neural network classification pipeline.

## 🎯 Problem Statement

Build a production-ready ML pipeline to automatically detect whether a car is damaged or whole, and assess damage severity using deep learning image classification.

## 🏗️ Pipeline

### Supervised Learning
- **Image Blur Detection** — Preprocessing for image quality assessment
- **Traditional ML** — Baseline classifier (whole vs. damaged)
- **Neural Network** — CNN-based damage classification

### Unsupervised Learning
- **Mask R-CNN** — Instance segmentation for damage localization
- **Custom annotations** — VIA region data for damage areas

## 📁 Project Structure

```
car-damage-detection-mlops/
├── supervised_learning/
│   ├── Image_Blur_code.ipynb          # Image blur preprocessing
│   ├── NN_damaged_or_whole.ipynb      # Neural network classifier
│   ├── traditional_whole_or_damged.ipynb  # Traditional ML baseline
│   └── supervised_writeup/sup_summary.md
├── unsupervised_learning/
│   ├── Mask_rcnn.ipynb                # Mask R-CNN segmentation
│   ├── custom.py                      # Custom layer implementations
│   ├── data/                          # Annotation JSONs (VIA format)
│   └── writeup/unsup_summary.md
└── README.md
```

> **Note:** Training image datasets are not included in this repo due to size constraints.
> Download dataset from [Kaggle — Car Damage Dataset](https://www.kaggle.com/datasets/anujms/car-damage-detection).

## ✨ Features

- 🚗 Damaged vs. whole car binary classification
- 🔵 Image blur quality preprocessing pipeline
- 🧠 CNN neural network with transfer learning
- 🎭 Mask R-CNN damage region segmentation
- 📊 Supervised + unsupervised approaches compared
- 🏭 MLOps-ready reproducible pipeline

## 🛠️ Setup

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib
jupyter notebook
```

## 👨‍💻 Author

**mrchandrayee** · [GitHub](https://github.com/mrchandrayee) · [Profile](https://github.com/mrchandrayee/mrchandrayee)

## 📄 License

[MIT License](./LICENSE)
