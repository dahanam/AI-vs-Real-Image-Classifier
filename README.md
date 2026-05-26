# AI-Generated vs Real Image Classifier

A deep learning image classification project that compares ResNet-50 
and VGG-16 for detecting AI-generated images versus real photographs, 
including Monte Carlo uncertainty estimation and user-submitted image 
testing.

**Course:** CPS 4801 — Artificial Intelligence  
**Author:** Dahana Moz Ruiz 
**Institution:** Kean University

---

## Project Overview

The rapid rise of AI-generated imagery presents new challenges for 
digital media verification. This project evaluates the effectiveness 
of two pretrained CNN architectures — ResNet-50 and VGG-16 — in 
classifying images as either AI-generated or real, using transfer 
learning on a Kaggle dataset of 975 images.

---

## Dataset

**Source:** [AI Generated Images vs Real Images — Kaggle](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)

- 975 total images: 539 AI-generated, 436 real
- No preprocessing or augmentation applied
- Split into training and testing subsets via `train_test_split`
- Custom `CustomDataset` class handles image loading, resizing, 
  and tensor conversion

---

## Models

### ResNet-50
- 50-layer CNN with residual (skip) connections
- Avoids vanishing gradient problem through direct gradient flows
- Pretrained on ImageNet, fine-tuned for binary classification
- Loss: Cross-Entropy with Softmax

### VGG-16
- 16-layer CNN with consecutive 3×3 convolutional blocks
- Max pooling between blocks, fully connected layers at the end
- Pretrained on ImageNet, fine-tuned for binary classification
- Loss: Cross-Entropy with Softmax

---

## Results

### Training Performance (10 Epochs)

| Model | Best Training Accuracy | Final Epoch Accuracy |
|-------|----------------------|---------------------|
| ResNet-50 | 87.24% | 84.15% |
| VGG-16 | 98.20% | 98.32% |

### Test Set Performance (Confusion Matrix)

**ResNet-50:** Strong at detecting AI-generated images (91 correct), 
weaker on real images (47 correct, 40 misclassified as AI)

**VGG-16:** Strong at detecting real images (75 correct), 
weaker on AI-generated (61 correct, 46 misclassified as real)

### Key Finding
Both models show significant accuracy drop from training to testing, 
suggesting overfitting to specific image characteristics rather than 
generalizable AI vs real patterns. An ensemble approach combining both 
models' complementary strengths could improve overall performance.

---

## Notebook

| File | Description |
|------|-------------|
| `ai_image_classifier.ipynb` | Full pipeline — dataset loading, data visualization, ResNet-50 training, VGG-16 training, confusion matrices, Monte Carlo uncertainty estimation, and user-submitted image testing |
| `AI_Final_Project_Report.pdf` | Full project report including literature review, architecture details, results, and conclusions |

---

## Additional Features

### User-Submitted Image Testing
Both models support real-time classification of images submitted 
via URL from outside the dataset, demonstrating generalization 
capability beyond the training set.

### Monte Carlo Uncertainty Estimation
10-iteration Monte Carlo dropout simulations per model output:
- **Mean Prediction** — model confidence per class
- **Variance / Standard Deviation** — prediction spread (uncertainty)
- **95% Confidence Interval** — probability distribution range

---

## Tech Stack

- **Python** — Core language
- **PyTorch** — Model training and inference
- **TorchVision** — Pretrained ResNet-50 and VGG-16, transforms
- **Scikit-learn** — Train/test split, confusion matrix
- **Seaborn / Matplotlib** — Visualization
- **PIL** — Image loading
- **Google Colab** — Development environment (GPU runtime)

---

## Setup

```bash
pip install torch torchvision scikit-learn seaborn matplotlib pillow imagehash opencv-python requests
```

Download the dataset from Kaggle, upload it to Google Drive, then update the `DATA_DIR` variable at the top of the notebook to point to your folder before running.

---

## Report

The full project report (`AI_Final_Project_Report.pdf`) includes:
- Literature review
- Model architecture details
- Training and testing results
- Confusion matrix analysis
- Monte Carlo uncertainty discussion
- Conclusions and future work

---

## Author

Dahana Moz Ruiz
Kean University — CPS 4801 Artificial Intelligence
