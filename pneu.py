import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from tqdm.notebook import tqdm 

class CFG:
    
    epochs = 2                                        # No. of epochs of training the model 
    lr = 0.001                                         # Learning rate 
    batch_size = 16                                    # Batch Size For Dataset 
    
    model_name = 'tf_efficientnet_b4_ns'               # Model name (We are going to import model from timm)
    img_size = 224
    
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

valid_transform = T. Compose( [

    T.Resize(size = (CFG.img_size, CFG.img_size)),
    T.ToTensor(), #(h,w,c) -> (c,h,w)
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

test_transform = T.Compose([

    T.Resize(size = (CFG.img_size, CFG.img_size) ),
    T.ToTensor(), #(h,w,c) -> (c,h,w)
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

train_path = os.path.join(CFG.DATA_DIR,CFG.TRAIN)
valid_path = os.path.join(CFG.DATA_DIR,CFG.VAL)
test_path = os.path.join(CFG.DATA_DIR,CFG.TEST)

trainset = datasets.ImageFolder(train_path,transform = train_transform)
validset = datasets.ImageFolder(valid_path,transform = valid_transform)
testset = datasets.ImageFolder(test_path,transform = test_transform)

print("Trainset Size : {}".format(len(trainset)))
print("Validset Size : {}".format(len(validset)))
print("Testset Size : {}".format(len(testset)))

image,label = trainset[2]

class_name = ['NORMAL','PNEUMONIA']

show_image(image, class_name[label])

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

# Create a grid of images and display them
out = make_grid(images, nrow=4)
show_grid(out, title=[class_name[x] for x in labels])


from torch import nn
import torch.nn.functional as F
import timm

model = timm.create_model(CFG.model_name, pretrained = True)

for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Linear(in_features = 1792, out_features = 625),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(in_features = 625, out_features = 256),
    nn.ReLU(),
    nn.Linear(in_features = 256, out_features = 2)
)
#print(model)

#from torchsummary import summary

#summary(model,input_size = (3,224,224))

from helper import accuracy
from tqdm import tqdm
class PneumoniaTrainer():
    def __init__(self,criterion,optimizer,schedular):
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
    def train_batch_loop(self,model,trainloader):
        train_acc = 0.0
        train_loss = 0.0

        for images,labels in tqdm(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = self.criterion(logits,labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss+= loss.item()
            train_acc += accuracy(logits,labels)

        return train_acc / len(trainloader), train_loss / len(trainloader)
    def valid_batch_loop(self,model,validloader):
        valid_acc = 0.0
        valid_loss = 0.0

        for images,labels in tqdm(validloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = self.criterion(logits,labels)

            valid_loss += loss.item()
            valid_acc += accuracy(logits,labels)
        return valid_loss / len(validloader), valid_acc / len(validloader)

    def fit(self,model,trainloader,validloader,epochs):
        model = model.to(device)
        valid_min_loss = np.inf

        for i in range(epochs):
            model.train() # turn on dropout
            avg_train_acc, avg_train_loss = self.train_batch_loop(model,trainloader)

            model.eval() #turn off dropout batch norm
            avg_valid_acc, avg_valid_loss = self.valid_batch_loop(model,validloader)

            if avg_valid_loss <= valid_min_loss:
                print("Valid loss decreased {} --> {}".format(valid_min_loss, avg_valid_loss))
                torch.save(model.state_dict(),'ColabPneumoniaModel.pt')
                valid_min_loss = avg_valid_loss
            
            print("Epoch : {} Train Loss : {} Train Acc : {}".format(i+1,avg_train_loss,avg_train_acc))
            print("Epoch : {} Valid Loss : {} Valid Acc : {}".format(i+1,avg_valid_loss,avg_valid_acc))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = CFG.lr)
schedular = None

trainer = PneumoniaTrainer(criterion,optimizer,schedular)
trainer.fit(model,trainloader,validloader,epochs = CFG.epochs)

model.load_state_dict(torch.load('ColabPneumoniaModel.pt'))
model.eval()

avg_test_loss, avg_test_acc = trainer.valid_batch_loop(model,testloader)


print("Test Loss : {}".format(avg_test_loss))
print("Test Acc : {}".format(avg_test_acc))


from helper import view_classify
import torch.nn.functional as F

image,label = testset[324]

logits = model(image.to(device).unsqueeze(0))
ps = F.softmax(logits,dim = 1)

view_classify(image,ps,label)