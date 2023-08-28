import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('./WineQT.csv')
df.shape
df.head()
df['quality'].value_counts()
df_train, df_val = train_test_split(df, test_size=0.2)
X_train = df_train.drop(['quality', 'Id'], axis=1)
X_val = df_val.drop(['quality', 'Id'], axis=1)

y_train = df_train['quality']
y_val = df_val['quality']

# Scaling
scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# To Tensor
X_train_ts = torch.FloatTensor(X_train)
X_val_ts = torch.FloatTensor(X_val)
y_train_ts = torch.LongTensor(y_train.values)
y_val_ts = torch.LongTensor(y_val.values)

# Hyperparameter
LR = 1e-3
N_EPOCH = 500
DROP_PROB = 0.3

# Model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(X_train_ts.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 11)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=DROP_PROB)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc4(x)
        return output
    
model = DNN()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(1, N_EPOCH+1):
    model.train()
    out = model(X_train_ts)
    loss = loss_fn(out, y_train_ts)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = (torch.argmax(out, dim=1) == y_train_ts).float().mean().item()
    model.eval()
    with torch.no_grad():
        out_val = model(X_val_ts)
        loss_val = loss_fn(out_val, y_val_ts)
        acc_val = (torch.argmax(out_val, dim=1) == y_val_ts).float().mean().item()
    if epoch % 20 == 0:
        print('Epoch : {:3d} / {}, Loss : {:.4f}, Accuracy : {:.2f} %, Val Loss : {:.4f}, Val Accuracy : {:.2f} %'.format(epoch, 
                                                                                                                          N_EPOCH, 
                                                                                                                          loss.item(), 
                                                                                                                          acc*100, 
                                                                                                                          loss_val.item(),
                                                                                                                          acc_val*100))
