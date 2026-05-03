Project Overview: This repository implements machine learning pipelines to predict breast cancer receptor status (ER, PR, HER2) from multi-modal data. We combine magnetic resonance imaging (MRI) and clinical/pathological features from the Duke Breast Cancer dataset to develop and compare models. The goal is to evaluate single-modality (image-only, clinical-only) and hybrid (late fusion) models, and provide interpretable insights into feature importance.

<img width="936" height="530" alt="image" src="https://github.com/user-attachments/assets/abc7bf47-17dc-42ee-80d5-6321532357df" />




Dataset: We use the Duke Breast Cancer MRI dataset (a subset from The Cancer Imaging Archive). This is a single-institution, retrospective collection of 922 biopsy-confirmed invasive breast cancer patients with the following data:

Pre-operative dynamic contrast-enhanced (DCE) MRI volumes.
Clinical/pathology annotations (tumor size, grade, histologic type, molecular markers including ER/PR/HER2).
Computer-extracted imaging features (529 radiomic features).
For each patient, the receptor status (ER/PR/HER2) and molecular subtype are provided.
The repository includes scripts to preprocess these data for training and evaluation.

Repository Files:

.gitignore – Specifies files and directories to be ignored by git (e.g., large data files, environment settings).
dataset.py – Data loading and preprocessing module. Defines PyTorch datasets and data loaders for MRI images and clinical tables. Handles train/validation/test splits.
model.py – Model definitions. Contains architectures for 3D ResNet-18 and DenseNet-121 (adapted for MRI input), as well as interfaces for classical ML models (Logistic Regression, Random Forest, SVM) on clinical features.
main.py – Main training and evaluation script. Parse command-line arguments (config files or flags) to train specified models, perform validation/testing, and save results. Supports image-only, clinical-only, and hybrid training modes.
Explainability.py – Generates model explanations. Computes SHAP values for clinical models and Grad-CAM heatmaps for CNNs. Produces visualizations (plots) for understanding feature contributions per-prediction.
gpu_job.sh – Example SLURM (or other HPC) job script for running experiments on GPU nodes. Shows how to request resources, activate the environment, and launch training.
requirements.txt – (if present) Pin lists of Python dependencies.
Environment & Dependencies:

Python 3.8+.
PyTorch (v1.7 or newer) with CUDA support (for GPU training; GPUs are strongly recommended for CNN models).
torchvision (for model architectures).
scikit-learn (for logistic regression, random forest, SVM, metrics).
numpy, pandas (data handling).
shap, matplotlib, seaborn (for explainability plots).
opencv-python or Pillow (if needed for image pre-processing).
For reproducibility, create a conda environment (or use venv) and install dependencies via pip install -r requirements.txt or conda install.
Installation:

Clone the repository:
bash
Copy
git clone https://github.com/yourusername/Multi-modal-Breast-Cancer-Prediction-using-AI.git
cd Multi-modal-Breast-Cancer-Prediction-using-AI
(Optional) Create a conda environment:
bash
Copy
conda create -n breast-ai python=3.8
conda activate breast-ai
Install dependencies:
bash
Copy
pip install -r requirements.txt
Data Preparation:

Download the Duke Breast Cancer MRI dataset from TCIA [TCIA link here] (requires TCIA account). Place image data and clinical CSVs into designated folders as configured in config.yaml or as documented in dataset.py.
Ensure the CSV files have columns matching our code’s expectations (ER/PR/HER2 labels, molecular subtype, tumor grade, etc.). You may need to preprocess the Duke data into a standard format using the provided dataset.py utilities.
Usage / Quickstart:

Training an Image-Only Model:

bash
Copy
python main.py --mode train --model ResNet18 --data images/ --labels clinical.csv --out_dir results/resnet18
This will train a 3D ResNet-18 on the MRI volumes to predict ER/PR/HER2.

Training a Clinical-Only Model:

bash
Copy
python main.py --mode train --model LogisticRegression --data clinical.csv --out_dir results/lr_clinical
Uses clinical features only (no images).

Training the Hybrid Model:

bash
Copy
python main.py --mode train --model Hybrid --image_dir images/ --clinical_data clinical.csv --out_dir results/hybrid
Late fusion of image and clinical data.

Evaluation/Test:

bash
Copy
python main.py --mode eval --checkpoint results/hybrid/checkpoint.pth --test_data test_data/
Runs the trained model on test set and prints metrics (accuracy, confusion matrix, AUC).

Explainability:
After training, generate explanations:

bash
Copy
python Explainability.py --model results/hybrid/checkpoint.pth --data test_data/ --out_dir results/hybrid/explain
This script will produce SHAP value plots (for clinical inputs) and Grad-CAM heatmaps (for MRI inputs) for sample test cases.

GPU Job **(example)**: Edit and submit gpu_job.sh via sbatch gpu_job.sh (for SLURM) to run a training session on an HPC cluster.

# Expected Outputs:

Trained Model Files: Checkpoints saved (e.g., *.pth) for each trained model.
Metrics: Printed classification reports (accuracy, precision, recall, F1, AUC) in console or logs.
ROC/Confusion Matrices: Plotted to results/ directory.
Explainability Plots: SHAP summary plots and Grad-CAM visualizations (PNG files) in results/<model>/explain/.
# Reproducibility:

Ensure fixed random seeds for train/val splits (configurable in main.py).
Use --split seed 42 or similar flags to replicate data splits and results.
We recommend performing k-fold cross-validation for robust estimates.
# License & Citation:
This code is released under the MIT License. If you use it in your work, please cite this repository and the Duke Breast Cancer MRI dataset as follows:

Duke Breast Cancer MRI dataset: [“Duke-Breast-Cancer-MRI,” The Cancer Imaging Archive (TCIA), https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903].
This repository: Multi-modal Breast Cancer Prediction using AI (GitHub, Year).
# Troubleshooting:

Out of Memory (OOM) Errors: Reduce batch size or use a smaller network. Ensure CUDA is visible (torch.cuda.is_available() should return True).
Package Import Errors: Double-check requirements.txt and Python version (3.8+). The code may use torchvision.transforms and sklearn utilities.
Data Mismatch: Verify that clinical CSV columns match those expected in dataset.py (look for column names in the code and example). Check for NaNs in clinical data.
License/Citation Issues: Contact the dataset providers (TCIA) or original paper authors if needed for permissions beyond research use.
