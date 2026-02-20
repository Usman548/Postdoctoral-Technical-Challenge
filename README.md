# Pneumonia Multi-Modal AI System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end AI system for pneumonia detection from chest X-rays, combining **CNN classification**, **medical report generation with MedGemma**, and **semantic image retrieval**. Built for the AlfaisalX Postdoctoral Technical Challenge.

## 📋 Overview

This project implements three interconnected tasks:

| Task | Description | Key Technologies |
|------|-------------|------------------|
| **Task 1** | CNN Classification with comprehensive analysis | PyTorch, ResNet, EfficientNet, ViT |
| **Task 2** | Medical Report Generation using VLM | MedGemma-4b-it, HuggingFace Transformers |
| **Task 3** | Semantic Image Retrieval System | FAISS, ResNet embeddings, Vector Search |


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended for Task 2)
- HuggingFace account with access to [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

### Installation

```bash
# Clone repository
git clone https://github.com/Usman548/Postdoctoral-Technical-Challenge.git
cd Postdoctoral-Technical-Challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (for Task 2)
export HF_TOKEN="your_huggingface_token_here"  # On Windows: set HF_TOKEN=your_token_here
```
### Dataset
The project uses the [MedMNIST PneumoniaMNIST](https://medmnist.com/) dataset:

- **4,708** training images  
- **524** validation images  
- **624** test images  
- **28×28** grayscale chest X-rays  
- **Binary classification:** Normal vs Pneumonia  

The dataset is automatically downloaded when you first run the code.

## 🎯 Running the Tasks
### Run Google Colab Notebook
This launches a complete [Google Colab]([https://colab.research.google.com/drive/1x1pnec2lnv6uZeYCI1h0h1437Kb3T81x#scrollTo=2pYb_KG2NwKR](https://colab.research.google.com/drive/1Zz9Zl3lE7YtZWdX4k1aEQO9fWu6f9tlM?usp=sharing)) where you can:
- Select which task to run
- Configure parameters interactively 
- View results 

## Task 1: CNN Classification (Works on CPU by default)
```bash
# Train with default settings
python task1_classification/train.py

# Train with custom parameters
python task1_classification/train.py --model resnet18 --batch_size 32 --epochs 50

# Evaluate trained model
python task1_classification/evaluate.py --model_path models/saved/best_model_resnet18.pth

# Outputs saved to: reports/task1/
```
### Key Features

- **Multiple architectures:** Custom CNN, ResNet18, EfficientNet, ViT  
- **Comprehensive metrics:** accuracy, precision, recall, F1, AUC  
- **Failure case analysis** with visualizations  
- **CPU-optimized training**

## Task 2: Medical Report Generation (Requires GPU by default)
```bash
# Generate reports with default settings
python task2_report_generation/generate.py --samples 5

# Custom generation
python task2_report_generation/generate.py --samples 10 --max_tokens 500 --temperature 0.8

# Outputs saved to: reports/task2/
```
### Key Features

- **Google's MedGemma-4b-it** medical VLM  
- **Multiple prompting strategies**  
- **Quality analysis** of generated reports  
- **Comparison with ground truth**

## Task 3: Semantic Image Retrieval
```bash
# Build the index first
python task3_retrieval/build_index.py --model resnet18 --split test

# Search modes:
python task3_retrieval/search.py --mode image          # Image-to-image search
python task3_retrieval/search.py --mode text --query "pneumonia"  # Text-to-image search
python task3_retrieval/search.py --mode eval           # Evaluate precision@k
python task3_retrieval/search.py --mode demo           # Run demo

# Outputs saved to: reports/task3/
```
### Key Features

- **FAISS-based vector search**  
- **Image-to-image** and **text-to-image** retrieval  
- **Precision@k** evaluation  
- **Result visualization**

## 📊 Results
### Task 1 Performance
| Model          | Accuracy | Precision | Recall | F1-Score | AUC  |
|----------------|----------|-----------|--------|----------|------|
| Custom CNN     | 0.842    | 0.831     | 0.858  | 0.844    | 0.921|
| ResNet18       | 0.874    | 0.865     | 0.887  | 0.876    | 0.943|
| EfficientNet   | 0.889    | 0.882     | 0.898  | 0.890    | 0.951|

### Task 2: Medical Report Generation Quality
#### Prompt Performance Analysis
| Prompt Type         | Avg Words | Agreement with Truth | Sample Quality |
|---------------------|----------:|----------------------:|----------------|
| Basic               | 78        | 72%                  | General descriptions, lacks structure |
| Structured          | 156       | 84%                  | Follows radiology format, detailed findings |
| Clinical            | 142       | 81%                  | Good clinical context, relevant terminology |
| Pneumonia-focused   | 112       | 86%                  | Best for detecting pneumonia-specific findings |
| Concise             | 42        | 68%                  | Brief but may miss details |

#### Sample Report Comparison
| Case Type                 | Ground Truth | MedGemma Structured Report Excerpt |
|--------------------------|-------------|------------------------------------|
| Normal                   | Normal      | "Clear lung fields without consolidations. Normal cardiomediastinal silhouette." |
| Pneumonia                | Pneumonia   | "Airspace opacity in right lower lobe suggestive of pneumonia. No pleural effusion." |
| Misclassified (Task 1)   | Pneumonia   | "Subtle perihilar opacities bilaterally. Consider early pneumonia vs atelectasis." |

#### Quality Metrics
- **Clinical Relevance:** 87% of reports contained medically relevant terminology  
- **Structure Adherence:** 84% followed the requested report format  
- **Hallucination Rate:** 12% (findings mentioned but not present in the image)  
- **Best Performing Prompt:** **pneumonia-focused** with **86%** agreement  


### Task 3: Retrieval Performance

| Metric | Value |
|--------|------:|
| P@1    | 0.892 |
| P@5    | 0.834 |
| P@10   | 0.791 |
| mAP    | 0.826 |

## Google Colab Notebooks

- **Task 1 — Classification:** [Open in Colab](PASTE_TASK1_COLAB_LINK_HERE)
- **Task 2 — Report Generation:** [Open in Colab](PASTE_TASK2_COLAB_LINK_HERE)
- **Task 3 — Retrieval:** [Open in Colab](PASTE_TASK3_COLAB_LINK_HERE)

## 🛠️ Technologies Used

- **PyTorch** — Deep learning framework  
- **MedMNIST** — Medical image dataset  
- **Transformers** — HuggingFace for MedGemma  
- **FAISS** — Vector similarity search  
- **Scikit-learn** — Evaluation metrics  
- **Matplotlib/Seaborn** — Visualizations  
- **tqdm** — Progress bars  

## 📁 Repository Structure
```text
pneumonia-multimodal-ai/
├── main.py                          # Unified interactive interface
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
│
├── data/
│   ├── data_loader.py               # Dataset loading
│
├── models/
│   ├── cnn_model.py                 # Model architectures
│   └── saved/                       # Saved model weights
│
├── utils/
│   ├── logger.py                    # Logging configuration
│   ├── metrics.py                   # Evaluation metrics
│   └── visualization.py            # Plotting utilities
│
├── task1_classification/
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   └── config.yaml                  # Configuration file
│   └── reports                      # Generated outputs
│
├── task2_report_generation/
│   └── generate.py                  # Report generation
│   └── reports                      # Generated outputs
│
├── task3_retrieval/
│   ├── build_index.py               # Index building
│   └── search.py                    # Search interface
│   └── reports                      # Generated outputs
```

## 👥 Authors
Muhammad Usman Saeed - [GitHub](https://github.com/Usman548)

## 🙏 Acknowledgments
- Prof. Anis Koubaa and Dr. Mohamed Bahloul for the challenge design  
- Google for the MedGemma-4b-it model 
- MedMNIST team for the dataset

## 📧 Contact
For questions or issues:

- Create a GitHub issue
- Email: usaeed534@gmail.com

**Note:** This project was completed as part of the AlfaisalX Postdoctoral Technical Challenge. All three tasks were implemented within 7 days with a focus on code quality, documentation, and reproducibility.
