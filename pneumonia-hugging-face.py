import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from plotter import plot_acc_loss
from plot_roc import plot_roc_curve

class CFG:
    
    epochs = 5                                        # No. of epochs of training the model 
    lr = 0.001                                         # Learning rate 
    batch_size = 16                                    # Batch Size For Dataset 
    model_name = 'google/vit-base-patch16-224-in21k'   # Hugging Face ViT model
    img_size = 224                                     # Image size for resizing images
    
    # Going to be use for loading dataset 
    DATA_DIR = "chest_xray"                            # Data Directory 
    TEST = 'test'                                      # Test folder name in data directory 
    TRAIN = 'train'                                    # Train folder name in data directory 
    VAL ='val'                                         # Valid folder name in data directory 
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("On which device we are on : {}".format(device))

from torchvision import transforms as T, datasets
from helper import show_image

train_transform = T.Compose([

    T.Resize(size = (CFG.img_size, CFG.img_size)),
    T.RandomRotation(degrees = (-20,+20)),
    T.ToTensor(), #(h,w,c) -> (c,h,w)
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

valid_transform = T.Compose([

    T.Resize(size = (CFG.img_size, CFG.img_size)),
    T.ToTensor(), #(h,w,c) -> (c,h,w)
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

test_transform = T.Compose([

    T.Resize(size = (CFG.img_size, CFG.img_size) ),
    T.ToTensor(), #(h,w,c) -> (c,h,w)
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

train_path = os.path.join(CFG.DATA_DIR, CFG.TRAIN)
valid_path = os.path.join(CFG.DATA_DIR, CFG.VAL)
test_path = os.path.join(CFG.DATA_DIR, CFG.TEST)

trainset = datasets.ImageFolder(train_path, transform=train_transform)
validset = datasets.ImageFolder(valid_path, transform=valid_transform)
testset = datasets.ImageFolder(test_path, transform=test_transform)

print("Trainset Size : {}".format(len(trainset)))
print("Validset Size : {}".format(len(validset)))
print("Testset Size : {}".format(len(testset)))

#DISPLAY SAMPLE IMAGE
# image, label = trainset[2]
# class_name = ['NORMAL', 'PNEUMONIA']
# show_image(image, class_name[label])

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from helper import show_grid

# Assuming trainset, valset, and testset are defined earlier
trainloader = DataLoader(trainset, batch_size=CFG.batch_size, shuffle=True)
validloader = DataLoader(validset, batch_size=CFG.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=CFG.batch_size, shuffle=True)

# Print the number of batches and examples in the trainloader
print("No. of batches in trainloader: {}".format(len(trainloader)))
print("No. of Total examples: {}".format(len(trainloader.dataset)))

# Iterate over the trainloader and display some images
dataiter = iter(trainloader)
images, labels = next(dataiter)

from transformers import ViTForImageClassification
from torch import nn
import torch.nn.functional as F

# Load Hugging Face ViT model
model = ViTForImageClassification.from_pretrained(
    CFG.model_name,
    num_labels = 2
).to(device)

model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 2)
).to(device)

# Freeze all layers except the classifier head
for param in model.vit.parameters():
    param.requires_grad = False

from helper import accuracy

class PneumoniaTrainer():
    def __init__(self, criterion, optimizer, schedular, train_acc_list, train_loss_list):
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
        self.train_acc_list = train_acc_list
        self.train_loss_list = train_loss_list
    
    def train_batch_loop(self, model, trainloader):
        train_acc = 0.0
        train_loss = 0.0

        for images, labels in tqdm(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).logits
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits, labels)

        return train_loss / len(trainloader), train_acc / len(trainloader)
    
    def valid_batch_loop(self, model, validloader):
        valid_acc = 0.0
        valid_loss = 0.0

        for images, labels in tqdm(validloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).logits
            loss = self.criterion(logits, labels)

            valid_loss += loss.item()
            valid_acc += accuracy(logits, labels)
        
        return valid_loss / len(validloader), valid_acc / len(validloader)

    def fit(self, model, trainloader, validloader, epochs):
        model = model.to(device)
        valid_min_loss = np.inf

        for i in range(epochs):
            model.train()  # turn on dropout
            avg_train_loss, avg_train_acc = self.train_batch_loop(model, trainloader)
            self.train_loss_list += [avg_train_loss]
            self.train_acc_list += [avg_train_acc]
            model.eval()  # turn off dropout and batch norm
            avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model, validloader)

            if avg_valid_loss <= valid_min_loss:
                print("Valid loss decreased {} --> {}".format(valid_min_loss, avg_valid_loss))
                torch.save(model.state_dict(), 'ColabPneumoniaModel.pt')
                valid_min_loss = avg_valid_loss
            
            print(f"Epoch {i+1}: Train Loss: {avg_train_loss}, Train Acc: {avg_train_acc}")
            print(f"Epoch {i+1}: Valid Loss: {avg_valid_loss}, Valid Acc: {avg_valid_acc}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
schedular = None

trainer = PneumoniaTrainer(criterion, optimizer, schedular,[],[])
trainer.fit(model, trainloader, validloader, epochs=CFG.epochs)
plot_acc_loss(trainer.train_acc_list,trainer.train_loss_list)

model.load_state_dict(torch.load('ColabPneumoniaModel.pt'))
model.eval()


avg_test_loss, avg_test_acc = trainer.valid_batch_loop(model, testloader)
print("Test Loss : {}".format(avg_test_loss))
print("Test Acc : {}".format(avg_test_acc))


from helper import view_classify

image, label = testset[324]
logits = model(image.to(device).unsqueeze(0)).logits
ps = F.softmax(logits, dim=1)

view_classify(image, ps, label)