from dataset import load_clinical_data,BreastMRI3DDataset
from model import train_clinical_models, get_resnet18_3d, train_image_model, evaluate_model
from Explainability import explain_model_shap,GradCAM3D
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from pytorch_grad_cam import GradCAM
import numpy as np
import pandas as pd
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import os
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "/home/bao67/MastersProject/Medical_imaging/Tumor_Data"
LABEL_FILE = "/home/bao67/MastersProject/Medical_imaging/labels-2.xlsx"
EXPLAIN_DIR = "/home/bao67/MastersProject"
TARGET_COLUMNS = ["Tumor Characteristics_ER", "Tumor Characteristics_PR", "Tumor Characteristics_HER2"]

# -------------------------------
# Tabular Classifier Results
# -------------------------------
X_train, X_test, y_train, y_test, preprocessor, X_test_raw = load_clinical_data()
results = train_clinical_models(X_train, X_test, y_train, y_test, target_columns=TARGET_COLUMNS)

for name, (model, acc) in results.items():
    print(f"{name.upper()} Accuracy: {acc:.2f}")
    model_dir = os.path.join(EXPLAIN_DIR, f"{name}_all_markers")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Generating SHAP explanation for {name}_all_markers...")
    explain_model_shap(
        model=model,
        X_test_raw=X_test_raw,
        preprocessor=preprocessor,
        save_dir=model_dir,
        top_n=10
    )

# -------------------------------
# Image Classifier Setup
# -------------------------------
df = pd.read_excel(LABEL_FILE, index_col=0)
image_dirs = [os.path.join(IMAGE_DIR, d) for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
image_ids = [os.path.basename(p).replace("Breast_MRI_", "", 1) for p in image_dirs]
df = df.loc[df.index.intersection(image_ids)]

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


num_train_samples = 500
train_df = train_df.iloc[:num_train_samples]

train_image_dirs = [os.path.join(IMAGE_DIR, f"Breast_MRI_{idx}") for idx in train_df.index]
val_image_dirs = [os.path.join(IMAGE_DIR, f"Breast_MRI_{idx}") for idx in val_df.index]


print("Validation label distribution per class:")
for col in TARGET_COLUMNS:
    positive_count = val_df[col].sum()
    total_count = len(val_df)
    print(f"{col}: {int(positive_count)} positive / {total_count - int(positive_count)} negative")

train_dataset = BreastMRI3DDataset(train_image_dirs, train_df)
val_dataset = BreastMRI3DDataset(val_image_dirs, val_df)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# -------------------------------
# Model Training
# -------------------------------
num_classes = 3
model = get_resnet18_3d(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()
best_val_loss = float('inf')

with open('training_metrics.txt', 'a') as f:
    for epoch in range(5):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        f.write(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}\n")
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += loss_fn(outputs, labels).item()
                all_preds.append(torch.sigmoid(outputs).cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_dataloader)
        f.write(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}\n")
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        binary_preds = (all_preds > 0.5).float()
        acc = (binary_preds == all_labels).float().mean().item()
        f.write(f"Epoch {epoch+1}, Validation Accuracy: {acc:.4f}\n")
        print(f"Epoch {epoch+1}, Validation Accuracy: {acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            f.write(f"✅ Saved best model at Epoch {epoch+1}\n")
            print("✅ Saved best model")

# -------------------------------
# Final Evaluation: Smooth ROC
# -------------------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in tqdm(val_dataloader, desc="Validation"):
        x, y = x.to(device), y.to(device)
        all_preds.append(torch.sigmoid(model(x)).cpu())
        all_labels.append(y.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
true_classes = all_labels.argmax(1)
binarized_labels = np.eye(3)[true_classes]

class_names = ["ER", "PR", "HER2"]
plt.figure(figsize=(8, 6))

for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(binarized_labels[:, i], all_preds[:, i])
    auc = roc_auc_score(binarized_labels[:, i], all_preds[:, i])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    print(f"{name} AUC: {auc:.4f}")

plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.title("ROC Curve per Class")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("multi_class_roc_auc.png")
plt.close()