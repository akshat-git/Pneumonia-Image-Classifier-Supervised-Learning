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
from plot_roc import plot_roc_curve

class CFG:
    epochs = 2
    lr = 0.001
    batch_size = 16
    img_size = 224
    DATA_DIR = "chest_xray"
    TEST = 'test'
    TRAIN = 'train'
    VAL = 'val'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
val_data = datasets.ImageFolder(os.path.join(CFG.DATA_DIR, CFG.VAL), transform=val_test_transform)
test_data = datasets.ImageFolder(os.path.join(CFG.DATA_DIR, CFG.TEST), transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=CFG.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=CFG.batch_size, shuffle=False)

print(f"Train Size: {len(train_data)}, Val Size: {len(val_data)}, Test Size: {len(test_data)}")

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=CFG.lr)

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
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
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

best_model_wts = None
best_acc = 0.0
for epoch in range(CFG.epochs):
    print(f"Epoch {epoch + 1}/{CFG.epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = model.state_dict()

model.load_state_dict(best_model_wts)

def test_model(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(dataloader.dataset)

test_acc = test_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")

plot_roc_curve(model,test_loader,device)

