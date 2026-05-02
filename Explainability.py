import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shap
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from torchvision.transforms import ToPILImage
from captum.attr import IntegratedGradients
from monai.visualize import GradCAM

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------------------------------------
#1.SHAP
# ---------------------------------------------

def explain_model_shap(model, X_test_raw, preprocessor, save_dir, top_n=10):
    X_test_trans = preprocessor.transform(X_test_raw)
    X_test_trans = X_test_trans.toarray() if hasattr(X_test_trans, "toarray") else X_test_trans
    feature_names = preprocessor.get_feature_names_out()

    for i, est in enumerate(model.estimators_):
        print(f"Generating SHAP for output {i}...")
        label = ["ER", "PR", "HER2"][i]
        out_dir = os.path.join(save_dir, label)
        os.makedirs(out_dir, exist_ok=True)

        explainer = {
            RandomForestClassifier: shap.TreeExplainer,
            LogisticRegression: lambda e: shap.LinearExplainer(e, X_test_trans),
            SVC: lambda e: shap.KernelExplainer(e.predict_proba, X_test_trans)
        }.get(type(est), None)
        
        shap_values = explainer(est).shap_values(X_test_trans) if callable(explainer) else explainer(est).shap_values(X_test_trans)
        shap_sum = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap.summary_plot(shap_sum, X_test_trans, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary - {label}")
        plt.savefig(os.path.join(out_dir, f"shap_summary_{label}.png"), bbox_inches='tight', dpi=300)
        plt.close()

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model, self.target_layer = model, target_layer
        self.gradients = self.activations = None
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        self.model(input_tensor)[:, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)
        gradcam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        gradcam = F.interpolate(gradcam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        
        return gradcam.squeeze().cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
