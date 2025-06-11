import torch
import torch.nn as nn
from torchvision import models

def load_feature_extractor(device, pretrained_path):
    swin = models.swin_v2_s(weights=None)  
    num_features = swin.head.in_features

    swin.head = nn.Identity()
    swin = swin.to(device)

    checkpoint = torch.load(pretrained_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        swin.load_state_dict(checkpoint['model_state_dict'])
    else:
        swin.load_state_dict(checkpoint)
    
    swin.eval()
    return swin

import numpy as np
from tqdm import tqdm

def extract_features(model, dataloader, device):
    features = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            feats = model(images)
            feats = feats.cpu().numpy()
            features.append(feats)
    return np.concatenate(features, axis=0)


import numpy as np
from tqdm import tqdm

def extract_features(model, dataloader, device):
    features = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            feats = model(images)
            feats = feats.cpu().numpy()
            features.append(feats)
    return np.concatenate(features, axis=0)

from sklearn.svm import OneClassSVM

def train_ocsvm(X_train_feats):
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # nu é o % esperado de outliers
    oc_svm.fit(X_train_feats)
    return oc_svm


def test_ocsvm(model, oc_svm, dataloader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feats = model(images)
            feats = feats.cpu().numpy()

            preds = oc_svm.predict(feats)  # -1 = anomalia, 1 = normal

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)
