import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torchvision import transforms as T, datasets
from torch.utils.data import DataLoader
from torch import nn, optim

# -------------------
# Configuration
# -------------------
class CFG:
    epochs = 10
    lr = 0.001
    batch_size = 16
    img_size = 224
    DATA_DIR = "chest_xray"
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------
# Transforms & Data Loading
# -------------------
transform = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(os.path.join(CFG.DATA_DIR, CFG.TRAIN), transform=transform)
train_loader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True)

print(f"Train Size: {len(train_data)}")

# -------------------
# Autoencoder Model
# -------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 112, 112]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 56, 56]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, 28, 28]
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 128, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 3, 224, 224]
            nn.Sigmoid()  # Output pixel values in [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# -------------------
# Training Function
# -------------------
def train_autoencoder(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, _ in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Compare reconstructed input with original input
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

# -------------------
# Visualization Function
# -------------------
def visualize_reconstruction(model, dataloader):
    model.eval()
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
    outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)

    # Display original and reconstructed images
    plt.figure(figsize=(12, 6))
    for i in range(5):
        # Original
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.clip(inputs[i], 0, 1))
        plt.axis("off")
        plt.title("Original")
        # Reconstructed
        plt.subplot(2, 5, i + 6)
        plt.imshow(np.clip(outputs[i], 0, 1))
        plt.axis("off")
        plt.title("Reconstructed")
    plt.show()

# -------------------
# Main
# -------------------
model = Autoencoder().to(device)
criterion = nn.MSELoss()  # Reconstruction loss
optimizer = optim.Adam(model.parameters(), lr=CFG.lr)

for epoch in range(CFG.epochs):
    print(f"\nEpoch {epoch + 1}/{CFG.epochs}")
    train_loss = train_autoencoder(model, train_loader, criterion, optimizer)
    print(f"Train Loss: {train_loss:.4f}")

# Visualize reconstructions
visualize_reconstruction(model, train_loader)
