
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load dataset
df = pd.read_csv(r'./data/task2/diabetes_binary.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#scale X array
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split and Create Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train = torch.FloatTensor(X_train).unsqueeze(1)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_test = torch.LongTensor(y_test)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
for batch in [128]:
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)

# Define network architecture
class DiabetesNet(nn.Module):
    def __init__(self):
        super(DiabetesNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))
        # No layer 2
        self.layer2 = nn.Sequential(
            nn.Linear(32 * ((X_train.shape[2] // 2)), 8),  # 
            nn.Dropout(0.5))
        self.fc = nn.Linear(8, 2)  # Binary classification

    def forward(self, x):
        out = self.layer1(x)
        # No layer 2
        out = out.view(out.size(0), -1)
        out = self.layer2(out)
        out = self.fc(out)
        return out

# Initialize the model and optimizer
net = DiabetesNet()
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Tracking Loss and Accuracy
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

num_epochs = 50  # Set the number of epochs

# Training loop
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    
    # Validation
    with torch.no_grad():
        output = net(X_test)
        test_loss = criterion(output, y_test)
        test_losses.append(test_loss.item())
        
        # Calculate the training accuracy
        output_train = net(X_train)
        _, pred_train = torch.max(output_train, 1)
        train_accuracy = (pred_train == y_train).float().mean()
        train_accuracies.append(train_accuracy)

        # Calculate the test accuracy
        _, pred_test = torch.max(output, 1)
        test_accuracy = (pred_test == y_test).float().mean()
        test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}')

# Plotting the training and test accuracy
plt.figure()
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epochs')
plt.show()

# Plotting the training and test loss
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')
plt.show()

# Final evaluation
with torch.no_grad():
    outputs = net(X_test)
    _, predicted = torch.max(outputs, 1)
    cm = confusion_matrix(y_test, predicted)
    cr = classification_report(y_test, predicted)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    # Heatmap for Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()