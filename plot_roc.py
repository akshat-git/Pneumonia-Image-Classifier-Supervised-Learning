import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from torchvision import transforms as T, datasets, models
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F

def plot_roc_curve(model, dataloader,device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Generating ROC Data"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.show()
    print(f"this is the threshold:{thresholds}")