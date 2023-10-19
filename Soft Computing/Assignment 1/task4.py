import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

# Load your CSV data into a pandas DataFrame
data = pd.read_csv(r'./data/task4/Alcohol_Sales.csv')
timeseries = data[["SALES"]].values.astype('float32')

# Convert 'Month' column to datetime and set it as the index
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)

# Normalize the 'Sales' column using Min-Max scaling
scaler = MinMaxScaler()
data['SALES'] = scaler.fit_transform(data['SALES'].values.reshape(-1, 1))

# Split the data into training and testing sets (70-30 split)
train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)

# Define a custom PyTorch dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data.values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Create data loaders for training and testing
lookback = 10  # You can adjust this based on your data
batch_size = 32
train_dataset = TimeSeriesDataset(train_data, lookback)
test_dataset = TimeSeriesDataset(test_data, lookback)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Initialize the model and set hyperparameters
input_size = 1  # Number of features (Sales column)
hidden_size = 100
num_layers = 2
output_size = 1  # Predicting a single value (Sales)
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 400
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
test_losses = []
train_losses = []
with torch.no_grad():
    for x, y in test_loader:
        outputs = model(x)
        test_loss = criterion(outputs, y)
        test_losses.append(test_loss.item())

with torch.no_grad():
    for x, y in train_loader:
        outputs = model(x)
        train_loss = criterion(outputs, y)
        train_losses.append(train_loss.item())



# Calculate and print the mean squared error on the test set
test_mse = np.mean(test_losses)
test_rmse = math.sqrt(test_mse)

train_mse = np.mean(train_losses)
train_rmse = math.sqrt(train_mse)
print(f'Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}')
print(f'Train MSE: {train_mse:.4f}, Train RMSE: {train_rmse:.4f}')

# You can also convert the predictions back to the original scale using scaler.inverse_transform if needed.

# Set the model back to evaluation mode
model.eval()

# Lists to store actual values, training predictions, and test predictions
actual_values = []
train_predictions = []
test_predictions = []

# Get actual values and predictions for the training dataset
with torch.no_grad():
    for x, y in train_loader:
        outputs = model(x)
        actual_values.extend(y.cpu().numpy())
        train_predictions.extend(outputs.cpu().numpy())

# Get actual values and predictions for the test dataset
with torch.no_grad():
    for x, y in test_loader:
        outputs = model(x)
        actual_values.extend(y.cpu().numpy())
        test_predictions.extend(outputs.cpu().numpy())

# Inverse transform the values if they were scaled
actual_values = scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))
train_predictions = scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1))
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))

# Create a time index for the actual values
time_index = range(len(actual_values))

# Plot the actual values, training predictions, and test predictions
plt.figure(figsize=(12, 6))
plt.plot(time_index, actual_values, label='Actual', color='blue')
plt.plot(time_index[:len(train_predictions)], train_predictions, label='Training Predictions', color='green')
plt.plot(time_index[len(train_predictions):], test_predictions, label='Test Predictions', color='red')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.grid(True)
plt.show()



