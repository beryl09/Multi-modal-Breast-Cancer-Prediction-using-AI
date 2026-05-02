import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, accuracy_score
from monai.networks.nets import resnet18



EXPLAIN_DIR = "/home/bao67/MastersProject"
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# ---------------------------------------------
# Clinical Models
# ---------------------------------------------
def train_clinical_models(X_train, X_test, y_train, y_test, target_columns):
    models = {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "RF": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }
    results, report = {}, ""
    for name, clf in models.items():
        model = MultiOutputClassifier(clf).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report += f"\n{name} Classification Report:\n"
        precs, recs, f1s = [], [], []
        for i, label in enumerate(target_columns):
            yt, yp = y_test.iloc[:, i], y_pred[:, i]
            report += f"\n--- {label} ---\n{classification_report(yt, yp)}"
            for (m, disp_class, suffix) in [
                ((confusion_matrix(yt, yp),), ConfusionMatrixDisplay, "confusion_matrix"),
                (precision_recall_curve(yt, yp)[:2], PrecisionRecallDisplay, "precision_recall_curve"),
                (roc_curve(yt, yp)[:2], RocCurveDisplay, "roc_curve")
            ]:
                disp = disp_class(*m) if suffix != "roc_curve" else disp_class(fpr=m[0], tpr=m[1], roc_auc=auc(*m))
                disp.plot()
                plt.savefig(os.path.join(EXPLAIN_DIR, f"{name}_{label}_{suffix}.png"))
                plt.close()
            precs.append(precision_score(yt, yp))
            recs.append(recall_score(yt, yp))
            f1s.append(f1_score(yt, yp))
        with open(os.path.join(EXPLAIN_DIR, f"{name}_precision_recall_f1.txt"), "w") as f:
            for i, label in enumerate(target_columns):
                f.write(f"--- {label} ---\nPrecision: {precs[i]:.4f}\nRecall: {recs[i]:.4f}\nF1 Score: {f1s[i]:.4f}\n\n")
        acc = accuracy_score(y_test, y_pred)
        report += f"\n{name} Accuracy: {acc:.2f}\n"
        results[name] = (model, acc)
    with open(os.path.join(EXPLAIN_DIR, "classification_report.txt"), "w") as f: f.write(report)
    return results



def get_resnet18_3d(num_classes: int, in_channels: int = 1):
    model = resnet.resnet18_3d(
        spatial_dims=3,
        n_input_channels=in_channels,
        num_classes=num_classes,
    )
    return model



# ---------------------------------------------
# Train Image Model
# ---------------------------------------------
def train_image_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Acc: {100*correct/total:.2f}%")

# Evaluate Model
def evaluate_model(model, test_loader, device, class_names=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")


    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(EXPLAIN_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot()
    plt.savefig(os.path.join(EXPLAIN_DIR, "confusion_matrix.png"))
    plt.close()

    
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    with open(os.path.join(EXPLAIN_DIR, "precision_recall_f1.txt"), "w") as f:
        f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\n")

    
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, [p[1] for p in all_probs])
    PrecisionRecallDisplay(precision=precision_vals, recall=recall_vals).plot()
    plt.savefig(os.path.join(EXPLAIN_DIR, "precision_recall_curve.png"))
    plt.close()

    
    fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.savefig(os.path.join(EXPLAIN_DIR, "roc_curve.png"))
    plt.close()

    with open(os.path.join(EXPLAIN_DIR, "auc_score.txt"), "w") as f:
        f.write(f"AUC: {roc_auc:.4f}\n")

    return acc