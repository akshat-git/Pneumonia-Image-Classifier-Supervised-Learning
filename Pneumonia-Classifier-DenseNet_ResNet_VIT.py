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

# -------------------
# Classification Configuration
# -------------------
class CFG:
    epochs = 10
    lr = 0.001
    batch_size = 16
    img_size = 224
    DATA_DIR = "chest_xray"
    TEST = 'test'
    TRAIN = 'train'
    VAL = 'val'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------
# Transforms & Data Loading
# -------------------
train_transform = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(os.path.join(CFG.DATA_DIR, CFG.TRAIN), transform=train_transform)
val_data   = datasets.ImageFolder(os.path.join(CFG.DATA_DIR, CFG.VAL), transform=val_test_transform)
test_data  = datasets.ImageFolder(os.path.join(CFG.DATA_DIR, CFG.TEST), transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=CFG.batch_size, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=CFG.batch_size, shuffle=False)

print(f"Train Size: {len(train_data)}, Val Size: {len(val_data)}, Test Size: {len(test_data)}")

# -------------------
# Model Building
# -------------------
def create_model(model_name: str, num_classes=2, freeze=True):
    """ Creates and returns a specified model (ResNet, DenseNet, ViT) with relevant classification head """
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        in_feats = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    elif model_name == "DenseNet121": #for xray analysis specifically 
        model = models.densenet121(pretrained=True)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        in_feats = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    elif model_name == "ViT_B_16":
        # part of torchvision
        model = models.vit_b_16(pretrained=True)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        in_feats = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)

# -------------------
# Training & Validation Sets
# -------------------
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# -------------------
# Testing (Accuracy & Loss + Probabilities for ROC stored for final plot) 
# -------------------
def test_model(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Collect loss & accuracy metrics
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            
            # Collect ROC
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss = total_loss / len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    return test_loss, test_acc, np.array(all_labels), np.array(all_probs)

# -------------------
# ROC & Youden’s J Computation
# -------------------
def compute_roc_info(all_labels, all_probs):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc

def compute_youdens_j(fpr, tpr, thresholds):
    youdens_j = tpr - fpr
    best_idx = np.argmax(youdens_j)
    best_thresh = thresholds[best_idx] # best threshold (used in plot)
    best_j = youdens_j[best_idx] #best j (used in plot)
    return thresholds, youdens_j, best_thresh, best_j

# -------------------
# Plotting ROC overlay
# -------------------
def plot_roc_curves(roc_data_dict):
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, roc_auc) in roc_data_dict.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.show()

# -------------------
# Plotting j curve overlay
# -------------------
def plot_all_threshold_vs_j(j_data_dict):
    plt.figure(figsize=(8, 6))
    for model_name, (thresholds, j_vals, best_thresh, best_j) in j_data_dict.items():
        plt.plot(thresholds, j_vals, label=f'{model_name}')
        # Mark best threshold
        plt.scatter(best_thresh, best_j, color='red')
        plt.text(best_thresh, best_j,
                 f'\nTh={best_thresh:.3f}\nJ={best_j:.3f}',
                 color='red', fontsize=9)

    plt.xlabel('Threshold')
    plt.ylabel('Youden\'s J (TPR - FPR)')
    plt.title('Threshold vs. Youden\'s J (All Models)')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

# -------------------
# Main func (run all models basically)
# -------------------
models_to_run = ["ResNet18", "DenseNet121", "ViT_B_16"] # //simple array of models
criterion = nn.CrossEntropyLoss()

test_losses = {}
test_accuracies = {}
roc_data = {}   # store fpr, tpr, roc_auc
j_data = {}     # store thresholds and J data

for model_name in models_to_run:
    print(f"\n========== {model_name} ==========")
    # Create model
    model = create_model(model_name, num_classes=2, freeze=True)

    # Define optimization
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr
    )

    # init store for training and val metrics per epoch
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    # run train and val sets
    best_model_wts = None
    best_acc = 0.0
    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch+1}/{CFG.epochs} for {model_name}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        # save metrics for final plot
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f},   Val Acc:   {val_acc:.4f}")
        
        # Track best model for val acc
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    # plot loss acc graphs
    epochs_range = range(1, CFG.epochs + 1)
    plt.figure(figsize=(12, 4))

    # loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_hist, label='Train Loss')
    plt.plot(epochs_range, val_loss_hist,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc_hist, label='Train Acc')
    plt.plot(epochs_range, val_acc_hist,   label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # load best weights and test
    model.load_state_dict(best_model_wts)
    test_loss, test_acc, all_labels, all_probs = test_model(model, test_loader, criterion)
    print(f"{model_name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # bar chart store for all 3 models
    test_losses[model_name] = test_loss
    test_accuracies[model_name] = test_acc

    # store roc data for final overlay plot
    fpr, tpr, thresholds, roc_auc = compute_roc_info(all_labels, all_probs)
    roc_data[model_name] = (fpr, tpr, roc_auc)

    # j calc
    thr, j_vals, best_thr, best_j = compute_youdens_j(fpr, tpr, thresholds)
    j_data[model_name] = (thr, j_vals, best_thr, best_j)

# -------------------
# Bar Chart for Acc & Loss (test set)
# -------------------
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
model_names = list(test_accuracies.keys())
acc_values  = [test_accuracies[m] for m in model_names]
plt.bar(model_names, acc_values, color=['blue','green','orange'])
plt.ylabel('Test Accuracy')
plt.title('Comparison of Test Accuracy')

# Loss
plt.subplot(1, 2, 2)
loss_values = [test_losses[m] for m in model_names]
plt.bar(model_names, loss_values, color=['blue','green','orange'])
plt.ylabel('Test Loss')
plt.title('Comparison of Test Loss')

plt.tight_layout()
plt.show()

# -------------------
# roc plot for all Models
# -------------------
plot_roc_curves({ 
    m: (roc_data[m][0], roc_data[m][1], roc_data[m][2])
    for m in roc_data
})

# -------------------
# j plot for all models
# -------------------
plot_all_threshold_vs_j(j_data)
