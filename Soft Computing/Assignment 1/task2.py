
# 1. Import libraries and packages.
import numpy as np
import pandas as pd
from pandas import read_csv as rc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage import transform
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random;
import math;


# Define a custom dataset class
class MusicGenresDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Retrieve features and labels from the CSV file
        features = self.dataframe.iloc[idx, :-1].values.astype(float)
        label = int(self.dataframe.iloc[idx, -1])

        # Apply transforms if specified
        if self.transform:
            features = self.transform(features)

        return torch.tensor(features), torch.tensor(label)

# Load the CSV file
csv_file = r'./data/task2/music_genre.csv'
data = rc(csv_file)

# Split the data into a 70-30 train/test split
train_size = int(0.7 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
train_data = pd.DataFrame(train_data.dataset, columns=data.columns)
test_data = pd.DataFrame(test_data.dataset, columns=data.columns)

# Define transformations (you can customize these)
transform = transforms.Compose([transforms.ToTensor()])

# Create custom datasets and data loaders for train and test
train_dataset = MusicGenresDataset(train_data, transform=transform)
test_dataset = MusicGenresDataset(test_data, transform=transform)

# Define batch size
num_epochs = 5
num_classes = 10
batch_size = 32
learning_rate = 0.001

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. Set hyperparameter (epoch, learning rate, batch size, number of class)
# 4. There are 10 classes for this data.


# 7. Develop a CNN model including conv2d, batchnormalize, ReLu activation,
# MaxPool2d, and fully connected layer.
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=5, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    


# 8. Call and run training for the model in the GPU/CPU.
device ='cpu'
model = CNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 9. Define loss function and optimizer.
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step,loss.item()))


# 10. Train and validate the model.
model.eval() # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.float())
        labels = Variable(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {}%'.format(100 * correct / total))
        torch.save(model.state_dict(), 'model.ckpt')

