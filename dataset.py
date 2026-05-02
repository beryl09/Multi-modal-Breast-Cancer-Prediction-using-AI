import os
import pandas as pd
import numpy as np
import torch
import glob
import cv2
import pydicom
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised, ScaleIntensityd, ToTensord
)

CLINICAL_PATH = "/home/bao67/MastersProject/Medical_imaging/cleaned_clinical_data3.xlsx"
TARGET_COLUMNS = ["Tumor Characteristics_ER", "Tumor Characteristics_PR", "Tumor Characteristics_HER2"]

def load_clinical_data(target_columns=TARGET_COLUMNS, num_samples=500):
    df = pd.read_excel(CLINICAL_PATH).rename(columns=str.strip).head(num_samples)
    X = df.drop(columns=["Patient Information_Patient ID"] + target_columns, errors='ignore')
    y = df[target_columns]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_raw, y_train = X_train_raw.head(num_samples), y_train.head(num_samples)

    num = ['Tumor Grade(T) (Tubule)', 'Tumor Grade(N) (Nuclear)', 
           'Tumor Grade(M) (Mitotic)', 'Staging(Tumor size [T]', 
           'Staging(Nodes)[N]', 'Staging Metastasis(M)']
    cat = ['Histologic type', 'Tumor Characteristics_Mol Subtype']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
    ])
    X_train_trans = preprocessor.fit_transform(X_train_raw[num + cat])
    X_test_trans = preprocessor.transform(X_test_raw[num + cat])

    to_df = lambda x: pd.DataFrame(x.toarray() if hasattr(x, "toarray") else x, columns=preprocessor.get_feature_names_out())
    return to_df(X_train_trans), to_df(X_test_trans), y_train.reset_index(drop=True), y_test.reset_index(drop=True), preprocessor, X_test_raw

class BreastMRI3DDataset(Dataset):
    def __init__(self, image_dirs, targets_df, transform=None, output_shape=(32, 128, 128), num_samples=500):
        self.image_dirs = image_dirs[:num_samples]
        self.targets_df = targets_df.head(num_samples)
        self.output_shape = output_shape
        self.base_transform = transform or Compose([
            ScaleIntensityd(keys=["volume"]),
            RandFlipd(keys=["volume"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["volume"], prob=0.5, max_k=3),
            RandAffined(keys=["volume"], prob=0.3, translate_range=10),
            RandGaussianNoised(keys=["volume"], prob=0.3, mean=0.0, std=0.1),
            ToTensord(keys=["volume"])
        ])

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        path = self.image_dirs[idx]
        slices = sorted(glob.glob(os.path.join(path, "**", "*.dcm"), recursive=True))
        if not slices:
            raise FileNotFoundError(f"No DICOM files found in {path}")

        try:
            vol = torch.from_numpy(np.stack([pydicom.dcmread(s).pixel_array for s in slices]).astype('float32')).unsqueeze(0)
            vol = F.interpolate(vol.unsqueeze(0), size=self.output_shape, mode='trilinear', align_corners=False).squeeze(0)
        except Exception as e:
            raise RuntimeError(f"Failed to process DICOMs from {path}: {e}")

        #image_id = os.path.basename(path).replace("Breast_MRI_", "")
        image_id = "_".join(os.path.basename(path).strip().split("_")[-3:])
        if image_id not in self.targets_df.index:
            raise KeyError(f"Image ID '{image_id}' not found in targets DataFrame.")

        data = self.base_transform({"volume": vol})
        label = torch.tensor(self.targets_df.loc[image_id].values, dtype=torch.float32)
        return data["volume"], label