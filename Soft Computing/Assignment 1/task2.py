
# 1. Import libraries and packages.
import numpy as np
import pandas as pd
from pandas import read_csv as rc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage import transform
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random;
import math;


# Define a custom dataset class
class MusicGenresDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        features = row[:-1].values.astype(float)
        label = int(row[-1])
        return torch.tensor(features), torch.tensor(label)

# Load the CSV file
csv_file = r'./data/task2/music_genre.csv'
data = pd.read_csv(csv_file)

# Split the data into a 70-30 train/test split
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Define batch size
batch_size = 32

# Create data loaders
train_loader = DataLoader(MusicGenresDataset(train_data), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(MusicGenresDataset(test_data), batch_size=batch_size, shuffle=False)

# Define the CNN model for 1D input features
class MusicGenresCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenresCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the CNN model
input_size = len(data.columns) - 1  # Number of input features (excluding the label)
num_classes = 10  # Number of classes
model = MusicGenresCNN(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop (you may need to adjust the number of epochs)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model on the test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test data: {100 * correct / total:.2f}%")