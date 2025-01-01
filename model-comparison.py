import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
import timm
from helper import accuracy

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

# Define transformations
train_transform = T.Compose([ 
    T.Resize((CFG.img_size, CFG.img_size)),
    T.RandomRotation(degrees=(-20, +20)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = T.Compose([ 
    T.Resize((CFG.img_size, CFG.img_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = T.Compose([ 
    T.Resize((CFG.img_size, CFG.img_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_path = os.path.join(CFG.DATA_DIR, CFG.TRAIN)
valid_path = os.path.join(CFG.DATA_DIR, CFG.VAL)
test_path = os.path.join(CFG.DATA_DIR, CFG.TEST)

# Load datasets
trainset = datasets.ImageFolder(train_path, transform=train_transform)
validset = datasets.ImageFolder(valid_path, transform=valid_transform)
testset = datasets.ImageFolder(test_path, transform=test_transform)

trainloader = DataLoader(trainset, batch_size=CFG.batch_size, shuffle=True)
validloader = DataLoader(validset, batch_size=CFG.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=CFG.batch_size, shuffle=False)

class PneumoniaTrainer:
    def __init__(self, criterion, optimizer, scheduler):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_batch_loop(self, model, trainloader):
        train_acc = 0.0
        train_loss = 0.0
        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += accuracy(logits, labels)
        return train_acc / len(trainloader), train_loss / len(trainloader)

    def valid_batch_loop(self, model, validloader):
        valid_acc = 0.0
        valid_loss = 0.0
        for images, labels in tqdm(validloader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = self.criterion(logits, labels)
            valid_loss += loss.item()
            valid_acc += accuracy(logits, labels)
        return valid_loss / len(validloader), valid_acc / len(validloader)

    def fit(self, model, trainloader, validloader, epochs):
        model = model.to(device)
        valid_min_loss = np.inf
        for i in range(epochs):
            model.train()
            avg_train_acc, avg_train_loss = self.train_batch_loop(model, trainloader)
            model.eval()
            avg_valid_acc, avg_valid_loss = self.valid_batch_loop(model, validloader)

            if avg_valid_loss <= valid_min_loss:
                print(f"Valid loss decreased {valid_min_loss} --> {avg_valid_loss}")
                torch.save(model.state_dict(), 'best_model.pt')
                valid_min_loss = avg_valid_loss

            print(f"Epoch {i + 1} | Train Loss: {avg_train_loss} | Train Acc: {avg_train_acc}")
            print(f"Epoch {i + 1} | Valid Loss: {avg_valid_loss} | Valid Acc: {avg_valid_acc}")

    def evaluate(self, model, testloader):
        model.load_state_dict(torch.load('best_model.pt'))
        model.eval()
        avg_test_loss, avg_test_acc = self.valid_batch_loop(model, testloader)
        print(f"Test Loss: {avg_test_loss}")
        print(f"Test Acc: {avg_test_acc}")
        return avg_test_loss, avg_test_acc

# Define the models
models = {
    "ResNet": timm.create_model('resnet18', pretrained=True),
    "ViT": timm.create_model('vit_base_patch16_224', pretrained=True),
    "DenseNet": timm.create_model('densenet121', pretrained=True)
}

# Modify the classifier for each model
for model_name, model in models.items():
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 2)
    )

# Set optimizer and criterion
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(models["ResNet"].parameters(), lr=CFG.lr)
trainer = PneumoniaTrainer(criterion, optimizer, None)

# Train and evaluate models
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    trainer.fit(model, trainloader, validloader, epochs=CFG.epochs)
    test_loss, test_acc = trainer.evaluate(model, testloader)
    results[model_name] = {"loss": test_loss, "accuracy": test_acc}

# Plot bar chart for test accuracy and loss
names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in names]
losses = [results[name]["loss"] for name in names]

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(names, accuracies)
ax[0].set_title("Test Accuracy for Each Model")
ax[0].set_ylabel("Accuracy")

ax[1].bar(names, losses)
ax[1].set_title("Test Loss for Each Model")
ax[1].set_ylabel("Loss")

plt.show()

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()

for model_name, model in models.items():
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    true_labels = []
    predicted_probs = []
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
        true_labels.extend(labels.cpu().numpy())
        predicted_probs.extend(probs)

    fpr[model_name], tpr[model_name], _ = roc_curve(true_labels, predicted_probs)
    roc_auc[model_name] = auc(fpr[model_name], tpr[model_name])

# Plot ROC curves
plt.figure(figsize=(10, 7))
for model_name in models.keys():
    plt.plot(fpr[model_name], tpr[model_name], label=f'{model_name} (AUC = {roc_auc[model_name]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Determine which model to use
best_model = max(results, key=lambda x: results[x]["accuracy"])
print(f"Best model based on accuracy: {best_model}")