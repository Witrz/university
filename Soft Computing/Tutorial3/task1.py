
# 1. Import libraries and packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage import transform
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
#from vis_utils import *
import random;
import math;


class FashionMNISTDataset(Dataset):
    '''Fashion MNIST Dataset'''
    def __init__(self, csv_file, transform=None):
        """
        Args:
        csv_file (string): Path to the csv file
        transform (callable): Optional transform to apply to sample
        """
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28)#.astype(float);
        self.Y = np.array(data.iloc[:, 0])
        del data
        self.transform = transform
    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]
        if self.transform:
            item = self.transform(item)
        return (item, label)


# 2. Read Data train and test.
test_data = pd.read_csv('Soft Computing/Tutorial3/fashion-mnist_test.csv')
train_data = pd.read_csv('Soft Computing/Tutorial3/fashion-mnist_train.csv')

test_dataset = FashionMNISTDataset(csv_file='Soft Computing/Tutorial3/fashion-mnist_test.csv')
train_dataset = FashionMNISTDataset(csv_file='Soft Computing/Tutorial3/fashion-mnist_train.csv')

# 3. Set hyperparameter (epoch, learning rate, batch size, number of class)
# 4. There are 10 classes for this data.
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# 5. Load data for train and test.

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 6. Visualise some samples.
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};
fig = plt.figure(figsize=(8,8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img_xy = np.random.randint(len(train_dataset))
    img = train_dataset[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

# 7. Develop a CNN model including conv2d, batchnormalize, ReLu activation,
# MaxPool2d, and fully connected layer.
class CNN(nn.Module):
    def __init__(self,num_classes=10):
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
        self.fc = nn.Linear(7*7*32, 10)

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
