# Deepfake Detection Using Deep Learning

An AI-powered deepfake detection system that analyzes images and videos to identify potential digital manipulation. The project combines an **EfficientNet-B0 image classification model built with PyTorch** and a **pretrained DeepGuard video detection model** to provide separate detection pipelines for images and videos through a Streamlit web application.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Models Used](#models-used)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Disclaimer](#disclaimer)

## Overview

Deepfakes are synthetic or manipulated images and videos created using artificial intelligence techniques. Although these technologies have legitimate applications, manipulated media can also be used to spread misinformation, impersonate individuals, and create misleading digital content.

This project aims to detect potential deepfake content using deep learning. It supports two types of media:

- **Image Detection:** Uses an EfficientNet-B0-based image classification model to classify uploaded images as real or fake.
- **Video Detection:** Uses the pretrained DeepGuard `ms_eff_gcvit_b0` model and its video inference pipeline to analyze uploaded videos.

The two models operate through separate inference pipelines suited to their respective input types. A Streamlit interface allows users to upload media and view the corresponding prediction.

## Features

- **Image Deepfake Detection:** Classifies images using the EfficientNet-B0 model.
- **Video Deepfake Detection:** Analyzes videos using the pretrained DeepGuard model.
- **Separate Inference Pipelines:** Uses image-specific and video-specific processing workflows.
- **Automated Video Frame Analysis:** Samples video frames and aggregates predictions through DeepGuard's inference pipeline.
- **Prediction Scores:** Displays model outputs to help users interpret predictions.
- **GPU Acceleration:** Supports CUDA where compatible hardware and dependencies are available.
- **CPU Compatibility:** Can run without a GPU, although inference may be slower.
- **Interactive Web Interface:** Provides a user-friendly application built with Streamlit.
- **Cloud Deployment:** Designed for deployment through Streamlit Community Cloud.

## System Architecture

The system consists of two detection branches that share a common web interface.

```text
                  User
                   |
                   v
          Streamlit Web Application
                   |
           Select Media Type
             /           \
            /             \
       Image Upload     Video Upload
            |               |
            v               v
      Image Processing  Video Processing
            |               |
            v               v
      EfficientNet-B0   DeepGuard Model
       Image Model      ms_eff_gcvit_b0
            |               |
            v               v
       Image Score      Video Score
            |               |
            v               v
       Real / Fake      Real / Fake
            \               /
             \             /
              v           v
              Display Result
```

The image and video models are independent components. The application selects the appropriate pipeline according to the type of media uploaded by the user.

## Models Used

### 1. EfficientNet-B0 — Image Detection

EfficientNet-B0 is a convolutional neural network architecture designed to achieve a balance between computational efficiency and classification performance.

In this project, EfficientNet-B0 is used for image-based deepfake detection.

**Key characteristics:**

- Implemented using PyTorch and Torchvision.
- Uses transfer learning with pretrained weights as the starting point.
- Processes images resized to 224 × 224 pixels.
- Applies image normalization using the ImageNet mean and standard deviation.
- Uses a binary classification head to distinguish real and fake images.

**Image preprocessing**

The image pipeline uses the following transformations:

- Resize images to 224 × 224.
- Apply random horizontal flipping during training.
- Convert images to tensors.
- Normalize using the ImageNet channel statistics.

The normalization parameters are:

| Channel | Mean | Standard Deviation |
|---|---:|---:|
| Red | 0.485 | 0.229 |
| Green | 0.456 | 0.224 |
| Blue | 0.406 | 0.225 |

The trained model produces an output used to determine the predicted image class.

### 2. DeepGuard — Video Detection

The video detection pipeline uses the pretrained DeepGuard model with the architecture name `ms_eff_gcvit_b0`.

DeepGuard provides the model and associated inference pipeline used to analyze video content.

**Key characteristics:**

- Uses a pretrained deep learning model.
- Supports video-level inference through the DeepGuard `VideoPredictor`.
- Uses the configured frame extraction and face detection pipeline.
- Samples video frames for analysis.
- Aggregates frame-level information into a video-level score.

The current inference configuration uses:

| Parameter | Value |
|---|---|
| Model name | `ms_eff_gcvit_b0` |
| Input image size | 224 × 224 |
| Number of sampled frames | 20 |
| Aggregation mode | `conf` |
| Horizontal-flip test-time augmentation | Disabled |

The exact behavior of the frame extractor, face detector, and score aggregation follows the DeepGuard implementation.

**Important:** The video model is pretrained, while EfficientNet-B0 is the separate image detection model. The image model's evaluation results should not be treated as the video model's evaluation results.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning and model inference |
| Torchvision | Image transformations and EfficientNet model support |
| EfficientNet-B0 | Image classification |
| DeepGuard | Pretrained video deepfake detection |
| Streamlit | Web application interface |
| OpenCV | Video and image processing |
| NumPy | Numerical operations |
| Pillow | Image loading and processing |
| scikit-learn | Classification evaluation metrics |

## Project Structure

The following is a suggested structure. Adjust it to match the actual files in your repository.

```text
Deepfake-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   └── deepfake_images.ipynb
│
├── saved_model/
│   ├── deepguard_weights.pth
│   └── config.json
│
├── inference/
│   └── DeepGuard inference modules
│
├── dataset/
│   ├── images/
│   └── videos/
│
└── scripts/
    └── download-FaceForensics.py
```

**Notes:**

- The notebook contains the image model development workflow if you choose to include it in the repository.
- `saved_model/` contains the saved DeepGuard weights and inference configuration if your application loads them directly.
- The `inference/` directory represents the DeepGuard modules required by the application; preserve the actual source files and package structure.
- Keep large datasets and unnecessary intermediate files out of GitHub.
- If the EfficientNet-B0 model has a separately saved checkpoint, include its actual path and loading method in the project structure.

## Dataset

### Image Dataset

The image detection model uses the Deepfake and Real Images dataset available on Kaggle:

[Deepfake and Real Images — Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

The dataset contains real and manipulated images intended for deepfake classification.

The image pipeline resizes and normalizes the input images before passing them to EfficientNet-B0.

### Video Dataset

The video development and testing workflow uses FaceForensics++:

[FaceForensics++ — Official Repository](https://github.com/ondyari/FaceForensics)

The project uses real and manipulated video samples for video detection experiments and evaluation.

The DeepGuard model is pretrained, so the videos used for testing should not automatically be considered a training dataset for that model.

**Dataset handling:** Follow the applicable dataset licenses and access conditions. Do not upload restricted or large dataset files to a public GitHub repository.

## Installation

### Prerequisites

- Python 3.11, subject to dependency compatibility.
- Git.
- A compatible PyTorch installation.
- Internet access to install dependencies and retrieve any required pretrained checkpoints.

A CUDA-compatible NVIDIA GPU is recommended for faster local inference but is not mandatory if all required components support CPU execution.

### 1. Clone the Repository

Replace the placeholder URL with your actual repository URL.

```bash
git clone https://github.com/Madhubalakumar07/DeepFake-Detection.git
cd DeepFake-Detection
```

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If `python` is not available in your terminal, use the Python launcher:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Ensure that the requirements file includes all dependencies required by both the image and video pipelines, including DeepGuard-specific packages and compatible versions.

For CUDA-enabled PyTorch, follow the official installation instructions:

[PyTorch — Get Started](https://pytorch.org/get-started/locally/)

### 4. Configure Model Weights

Make sure the application can access:

- The EfficientNet-B0 checkpoint, if your implementation saves and loads one separately.
- The DeepGuard pretrained weights or its supported checkpoint-loading mechanism.
- The correct preprocessing configuration for each model.

A saved state dictionary alone does not contain the complete model architecture or video preprocessing pipeline.

## Running the Application

Start the Streamlit application from the project root:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in a browser to use the application.

## How It Works

### Image Detection Workflow

1. The user selects image detection.
2. The user uploads a supported image.
3. The application loads the image and applies the required preprocessing.
4. The preprocessed tensor is passed to EfficientNet-B0.
5. The model output is converted into a predicted class according to the implemented classification logic.
6. The application displays the prediction and any supported score.

### Video Detection Workflow

1. The user selects video detection.
2. The user uploads a supported video.
3. The application passes the video to the DeepGuard inference pipeline.
4. The pipeline extracts and processes sampled frames.
5. The pretrained model analyzes the video content.
6. DeepGuard aggregates the relevant outputs into a video-level score.
7. The application displays the resulting prediction.

The video score and the image model's output are produced by different models. Their scores should not be assumed to have the same calibration or meaning.

## Model Evaluation

A reliable evaluation should measure image and video performance separately.

### Recommended Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of correct predictions |
| Precision | Proportion of predicted fake samples that are actually fake |
| Recall | Proportion of fake samples correctly detected |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows true and false predictions for both classes |

### Evaluation Considerations

- Evaluate the EfficientNet-B0 image model on a held-out image test set.
- Evaluate DeepGuard on a separate video test set.
- Avoid overlap between training and test images, source videos, and identities where applicable.
- Test videos from manipulation methods not represented in development data to assess generalization.
- Report the test-set size, class distribution, threshold, and evaluation conditions alongside the metrics.
- Do not assume that results on a small sample represent real-world performance.

Initial experiments can help verify that the models work, but robust conclusions require a carefully designed evaluation.

## Deployment on Streamlit Community Cloud

The application is intended to be deployed on [Streamlit Community Cloud](https://share.streamlit.io/).

### Deployment Steps

1. Push the application code to GitHub.
2. Ensure `app.py` and `requirements.txt` are in the repository root.
3. Keep the required model code and configuration files accessible.
4. Sign in to Streamlit Community Cloud.
5. Create a new app and select the GitHub repository and branch.
6. Set the main file path to `app.py`.
7. Select a compatible Python version in the deployment settings.
8. Deploy and review the application logs if an error occurs.

Official deployment documentation:

[Streamlit Community Cloud — Deploy an App](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)

### Cloud Deployment Considerations

- Community Cloud hardware is limited compared with a local GPU workstation.
- Loading large model checkpoints can increase startup time and memory use.
- Video processing may be significantly slower on CPU.
- Dependency versions must be compatible with the selected Python environment.
- Large checkpoints may require a suitable external model-storage approach.
- Avoid committing the complete image or video datasets to GitHub.
- Test model loading and inference locally before deploying.

## Limitations

- Deepfake detection models may fail on unfamiliar manipulation techniques.
- Image compression, low resolution, lighting, and face visibility can affect predictions.
- Video predictions depend on the sampled frames and the model's preprocessing pipeline.
- A high or low score does not guarantee that media is authentic or manipulated.
- The image and video models may have different performance characteristics.
- Cloud resource limits can restrict the length and size of videos that can be processed efficiently.

## Future Improvements

- Fine-tune and evaluate the image model using carefully separated training data.
- Improve detection of previously unseen deepfake generation methods.
- Evaluate video predictions on source-disjoint test data.
- Add frame-level visualizations to explain video predictions.
- Improve error handling for invalid, corrupted, or unsupported media.
- Optimize model loading and video inference for cloud deployment.
- Add reproducible evaluation reports and model version tracking.
- Explore temporal models that learn relationships between video frames.

## References

- **DeepGuard:** [GitHub Repository](https://github.com/HanMoonSub/DeepGuard)
- **FaceForensics++:** [GitHub Repository](https://github.com/ondyari/FaceForensics)
- **Deepfake and Real Images Dataset:** [Kaggle Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
- **PyTorch:** [Official Website](https://pytorch.org/)
- **Torchvision:** [Official Documentation](https://pytorch.org/vision/stable/index.html)
- **Streamlit:** [Official Website](https://streamlit.io/)
- **Streamlit Community Cloud:** [Deployment Documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)

## Disclaimer

This project is intended for educational and research purposes. Predictions are estimates generated by machine learning models and should not be treated as definitive proof of authenticity or manipulation. The system should not be used as the sole basis for legal, financial, identity-related, or other high-impact decisions.

---

**Developed as a deep learning project exploring image and video deepfake detection.**
