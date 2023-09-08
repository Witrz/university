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
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
#from vis_utils import *
import random;
import math;

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F




# 2. Apply transforms function to normalise data.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
shuffle=True, num_workers=0)
testset = dsets.MNIST(
root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
shuffle=False, num_workers=0)


# 3. Load train and test online by torchvision.datasets.MNIST.
# 4. Change tensor to numpy image
# 5. Visualise some samples by getting some random training images.
# 6. Select device and parameters
# 7. Built CNN model including 2 Conv2d layers with kernel sizes, pooling layer, and fully
# connection layers.
# 8. Call model, create loss function, and optimizer.
# 9. Train model and measure the loss.


import matplotlib.pyplot as plt
import numpy as np
# functions to change tensor to numpy image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
# show images
imshow(torchvision.utils.make_grid(images[:6], nrow=3))
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class NumClassifyNet(nn.Module):
    def __init__(self):
        super(NumClassifyNet, self).__init__()
        # 1 input image channel, 16 output channels, 5X5 square convolutional kernels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = NumClassifyNet()
net = net.to(device)
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001)
test_data_iter = iter(testloader)
test_images, test_labels = next(test_data_iter)
for epoch in range(10):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        input_imgs, labels = data
        optimizer.zero_grad()
        input_imgs = input_imgs.to(device)
        labels = labels.to(device)
        outputs = net(input_imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # printing stats to check that out model is being trained correctly
    # and test on one image as we train
    running_loss += loss.item()
    #if i % 1000 == 0:
    print('epoch', epoch+1, 'loss', running_loss/1000)
    imshow(torchvision.utils.make_grid(test_images[0].detach()))
    test_out = net(test_images.to(device))
    _, predicted_out = torch.max(test_out, 1)
    print('Predicted : ', ' '.join('%5s' % predicted_out[0]))
print('Training finished')