import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from seaborn import heatmap, set
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Step 1 : Load Data - load the dataset, split into input (X) and output (y) variables
df1 = pd.read_csv(r'./data/task2/diabetes_binary.csv')
X = df1.iloc[:, :-1].values
y = df1.iloc[:, -1].values

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# Step 2: Define the PyTorch model
class DermatologyMLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, 80)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(80, 32)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(32, output_dim)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x
    

input_dim = 7
output_dim = 2

model = DermatologyMLPClassifier(input_dim, output_dim)

# creating our optimizer and loss function object
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
def train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs,train_losses,test_losses):
    for epoch in range(num_epochs):
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        #forward feed
        output_train = model(X_train)
        #calculate the loss
        loss_train = criterion(output_train, y_train)
        #backward propagation: calculate gradients
        loss_train.backward()
        #update the weights
        optimizer.step()
        output_test = model(X_test)
        loss_test = criterion(output_test,y_test)
        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")
num_epochs = 220
train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)
train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs,train_losses,test_losses)
plt.figure(figsize=(10,10))
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()
predictions_train = []
predictions_test = []
with torch.no_grad():
    predictions_train = model(X_train)
    predictions_test = model(X_test)
# Check how the predicted outputs look like and after taking argmax compare with y_train or y_test
#predictions_train
#y_train,y_test
def get_accuracy_multiclass(pred_arr,original_arr):
    if len(pred_arr)!=len(original_arr):
        return False
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred= []
    # we will get something like this in the pred_arr [32.1680,12.9350,-58.4877]
# so will be taking the index of that argument which has the highest value here 32.1680 which corresponds to 0th index
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0
#here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)

train_acc = get_accuracy_multiclass(predictions_train,y_train)
test_acc = get_accuracy_multiclass(predictions_test,y_test)

print(f"Training Accuracy: {round(train_acc*100,3)}")
print(f"Test Accuracy: {round(test_acc*100,3)}")
