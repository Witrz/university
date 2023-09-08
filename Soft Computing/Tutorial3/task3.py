import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# load the image
img_path = '/home/ghazal/cxray/test/normal/person1633_virus_2829.jpeg'
img = Image.open(img_path)
# convert PIL image to numpy array
img_np = np.array(img)
# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
# Python code for converting PIL Image to
# PyTorch Tensor image and plot pixel values
# import necessary libraries
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# define custom transform function
transform = transforms.Compose([
transforms.ToTensor()
])
# transform the pIL image to tensor
# image
img_tr = transform(img)
# Convert tensor image to numpy array
img_np = np.array(img_tr)
# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
img_tr = transform(img)
# calculate mean and std
mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
# print mean and std
print("mean and std before normalize:")
print("Mean of the image:", mean)
print("Std of the image:", std)
# python code to normalize the image
from torchvision import transforms
# define custom transform
# here we are using our calculated
# mean & std
transform_norm = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean, std)
])
# get normalized image
img_normalized = transform_norm(img)
# convert normalized image to numpy
# array
img_np = np.array(img_normalized)
# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
# Python Code to visualize normalized image
# get normalized image
img_normalized = transform_norm(img)
# convert tis image to numpy array
img_normalized = np.array(img_normalized)
# transpose from shape of (3,,) to shape of (,,3)
img_normalized = img_normalized.transpose(1, 2, 0)
# display the normalized image
plt.imshow(img_normalized)
plt.xticks([])
plt.yticks([])
# Python code to calculate mean and std
# of normalized image
# get normalized image
img_nor = transform_norm(img)
# cailculate mean and std
mean, std = img_nor.mean([1,2]), img_nor.std([1,2])
# print mean and std
print("Mean and Std of normalized image:")
print("Mean of the image:", mean)
print("Std of the image:", std)
variable = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.5 for _ in range(3)], [0.2 for _ in range(3)]),])
Train_data = datasets.ImageFolder(root = '/home/ghazal/cxray/train',
transform =variable )
Train_loader = DataLoader(Train_data, batch_size = 12, shuffle= True,
drop_last=True)
val_data = datasets.ImageFolder(root = '/home/ghazal/cxray/val', transform =variable )
val_loader = DataLoader(val_data, batch_size = 4, shuffle= True, drop_last=True)
test_data = datasets.ImageFolder(root = '/home/ghazal/cxray/test',transform =variable )
test_loader = DataLoader(test_data, batch_size = 12, shuffle= True,drop_last=True)
# Checking the images from the test dataset
examples = enumerate(Train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print('Shape of the Image: ',example_data.shape)
print('Shape of the label: ', example_targets.shape)
print(example_targets[0:6])
#checking the labels
class_name_train = Train_data.classes
print(class_name_train)
print(Train_data.class_to_idx)
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    #print(example_targets[i])
    plt.xticks([])
    plt.yticks([])
fig
# Set Device
device = 'cpu'
# CNN Architechture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 16,
        kernel_size= 5, stride=1,padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, out_channels= 32, kernel_size= 5,
        stride=1,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*27*27, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,2)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        #print(out.shape)
        out = self.pool1(out)
        #print(out.shape)
        out = F.relu(self.conv2(out))
        #print(out.shape)
        out = self.pool2(out)
        #print(out.shape)
        # flattening the layer
        out = out.reshape(out.size(0),-1)
        #print(out.shape)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
        stride=1,padding = 1), # (64*224*224)
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128,
        kernel_size=4,stride=2, padding=1), #(128*112*112)
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),
        #(128*112*112)
        # layer 1 complete
        nn.Conv2d(in_channels=128, out_channels=256,
        kernel_size=4,stride=2, padding=1), #(256*56*56)
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),
        #(256*56*56)
        #layer 2 complete
        nn.Conv2d(in_channels=256, out_channels=512,
        kernel_size=4,stride=2, padding=1), #(512*28*28)
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),
        #(512*28*28)
        # Layer 3 complete
        nn.Conv2d(in_channels=512, out_channels=512,
        kernel_size=4,stride=2, padding=1), #(512*14*14)
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),
        #(512*14*14)
        # layer 4 complete
        nn.Conv2d(in_channels=512, out_channels=512,
        kernel_size=4,stride=2, padding=1), #(512*7*7)
        nn.BatchNorm2d(512),
        nn.ReLU(),
        )
        self.fc = nn.Sequential(
        nn.Linear(512*7*7,256),
        nn.Linear(256, 128),
        nn.Linear(128,64),
        nn.Linear(64,2)
        )
    def forward(self,x):
        out = self.conv2(x)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out
#hecking the desired output with random input
model = ConvNet2()
x = torch.rand(10,3, 224,224)
print(model(x).shape)
output = model(x)
#print(output)
#hecking the desired output with random input
model = ConvNet()
x = torch.rand(128,3, 64,64)
print(model(x).shape)
output = model(x)
#print(output)
# Setting the hyperparameters
learning_rate = 0.01
num_classes = 2
num_epochs = 5
channel_img = 3
feature_d = 64
# Initializing the ConvNet
model = ConvNet2().to(device)
# loss criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
writer = SummaryWriter(f'runs/CNN/Plotting_on_tensorBoard')
# Training loop
num_total_steps = len(Train_loader.dataset)
step = 0
losses = []
accuracies = []
steps = []
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    for i , (images, labels) in enumerate(Train_loader):
        images = images.to(device)
        #print(len(images))
        labels = labels.to(device)
        #print(labels)
        #forward
        outputs = model(images)
        #print(outputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        # backwards and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calculating running training accuracies
        _, predictions = outputs.max(1)
        #print((predictions))
        num_correct = (predictions == labels).sum()
        running_train_acc = float(num_correct)/float(images.shape[0])
        accuracies.append(running_train_acc)
        train_acc += running_train_acc
        train_loss += loss.item()
        avg_train_acc = train_acc / len(Train_loader)
        avg_train_loss = train_loss / len(Train_loader)
        writer.add_scalar('Training Loss', loss, global_step= step)
        writer.add_scalar('Training Accuracy', running_train_acc,
        global_step=step)
        step += 1
        steps.append(step)
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i*len(images),num_total_steps, loss.item()))
print('Training Ended')

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
#torch.save(critic.state_dict(), "model_trained.h5")

torch.save(model.state_dict(), "ConvNet_1.pth")
plt.title('Training Loss')
plt.xlabel('Steps')
plt.ylabel('Losses')
plt.plot(steps, losses)
plt.show()
plt.title('Training accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(steps, accuracies)
plt.show()
# Training loop
num_total_steps = len(Train_loader.dataset)
valid_loss_min = np.Inf
step = 0
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []
accuracies = []
steps = []
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0
    model.train()
    for _ , (images, labels) in enumerate(Train_loader):
        images = images.to(device)
        #print(len(images))
        labels = labels.to(device)
        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        training_loss.append(loss.item())
        # backwards and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calculating running training accuracies
        _, predictions = outputs.max(1)
        num_correct = (predictions == labels).sum()
        running_train_acc = float(num_correct)/float(images.shape[0])
        training_accuracy.append(running_train_acc)
        train_acc += running_train_acc
        train_loss += loss.item()
        avg_train_acc = train_acc / len(Train_loader)
        avg_train_loss = train_loss / len(Train_loader)
        model.eval()
        with torch.no_grad():
            for _ , (images, labels) in enumerate(val_loader):
                images = images.to(device)
                #print(len(images))
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss.append(loss)
                # Calculating running training accuracies
                _, predictions = outputs.max(1)
                num_correct = (predictions == labels).sum()
                running_val_acc = float(num_correct)/float(images.shape[0])
                validation_accuracy.append(running_val_acc)
                val_acc += running_val_acc
                val_loss += loss.item()
            avg_valid_acc = val_acc / len(val_loader)
            avg_valid_loss = val_loss / len(val_loader)
        step += 1
        steps.append(step)
    # if avg_valid_loss <= valid_loss_min:
    # print('Validation loss decreased ({:.6f} --> {:.6f}).Saving model ...'.format(valid_loss_min,avg_valid_loss))
    # torch.save({
    # 'epoch' : i,
    # 'model_state_dict' : model.state_dict(),
    # 'optimizer_state_dict' : optimizer.state_dict(),
    # 'valid_loss_min' : avg_valid_loss
    # },'Pneumonia_model.pt')
    # valid_loss_min = avg_valid_loss
    print("Epoch : {} Train Loss : {:.6f} Train Acc : {:.6f}".format(epoch+1,avg_train_loss,avg_train_acc))
    print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {:.6f}".format(epoch+1,avg_valid_loss,avg_valid_acc))