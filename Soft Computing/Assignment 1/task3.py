import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

#download directory
os.environ['TORCH_HOME'] = r'./data'

# Create Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])

# Load Data, Set Dataloaders
train_dataset = datasets.ImageFolder(root=r'./data/task3/ChestXray/train', transform=transform)
test_dataset = datasets.ImageFolder(root=r'./data/task3/ChestXray/test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Load the pretrained AlexNet model and freeze its weights
model = models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier to include a Dropout layer before the final Linear layer
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1], nn.Dropout(0.5), nn.Linear(4096, 2))

# Set Cuda if available, set CPU if not.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Loss function and optimizer for the final layer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)

# Metrics Storage
train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

# Training Loop
num_epochs = 11
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += predicted.eq(targets).sum().item()
        total_train += targets.size(0)

    # Metrics after the epoch
    train_losses.append(train_loss/len(train_loader))
    train_accuracies.append(100. * correct_train / total_train)

    # Validation
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_labels_test, all_preds_test = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_test += predicted.eq(targets).sum().item()
            total_test += targets.size(0)

            all_labels_test.extend(targets.cpu().numpy())
            all_preds_test.extend(predicted.cpu().numpy())

    test_losses.append(test_loss/len(test_loader))
    test_accuracies.append(100. * correct_test / total_test)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, Test Accuracy: {test_accuracies[-1]:.2f}%")

# Plots
plt.figure()
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epochs')
plt.show()

plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')
plt.show()

# Final evaluation
test_f1 = f1_score(all_labels_test, all_preds_test, average='macro')
cm = confusion_matrix(all_labels_test, all_preds_test)
cr = classification_report(all_labels_test, all_preds_test)
print("Test F1 Score:", test_f1)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(cr)

# Heatmap for Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()