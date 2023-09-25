# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
from torchvision import models

# loading dataset
train = pd.read_csv(r'.\data\week6_train.csv')
train.head()

# loading training images
train_img = []
for img_name in tqdm(train['image_names']):
    # defining the image path
    image_path = 'data/week6_images/' + img_name
    # reading the image
    img = imread(image_path)
    # normalizing the pixel values
    img = img/255
    # resizing the image to (224,224,3)
    img = resize(img, output_shape=(224,224,3), mode='constant', anti_aliasing=True)
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    train_img.append(img)

# converting the list to numpy array
train_x = np.array(train_img)
train_x.shape

# Exploring the data
index = 10
plt.imshow(train_x[index])
if (train['emergency_or_not'][index] == 1):
    print('It is an Emergency vehicle')
else:
    print('It is a Non-Emergency vehicle')

# defining the target
train_y = train['emergency_or_not'].values

# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13, stratify=train_y)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

# converting training images into torch format
train_x = train_x.reshape(1481, 3, 224, 224)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# shape of training data
train_x.shape, train_y.shape

# converting validation images into torch format
val_x = val_x.reshape(165, 3, 224, 224)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)

# shape of validation data
val_x.shape, val_y.shape

# class Net(Module):   
#     def __init__(self):
#         super(Net, self).__init__()

#         self.cnn_layers = Sequential(
#             # Defining a 2D convolution layer
#             Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#             # Defining another 2D convolution layer
#             Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(8),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.linear_layers = Sequential(
#             Linear(8 * 56 * 56, 2)
#         )

#     # Defining the forward pass    
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x
# # defining the model
# model = Net()
# # defining the optimizer
# optimizer = Adam(model.parameters(), lr=0.0001)
# # defining the loss function
# criterion = CrossEntropyLoss()
# # checking if GPU is available
# #if torch.cuda.is_available():
# device='cpu'
# model = model.to(device)
# criterion = criterion.to(device)

# print(model)

# # batch size of the model
# batch_size = 128

# # number of epochs to train the model
# n_epochs = 15

# for epoch in range(1, n_epochs+1):

#     # keep track of training and validation loss
#     train_loss = 0.0
        
#     permutation = torch.randperm(train_x.size()[0])

#     training_loss = []
#     for i in tqdm(range(0,train_x.size()[0], batch_size)):

#         indices = permutation[i:i+batch_size]
#         batch_x, batch_y = train_x[indices], train_y[indices]
        
#         #if torch.cuda.is_available():
#         batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
        
#         optimizer.zero_grad()
#         # in case you wanted a semi-full example
#         outputs = model(batch_x)
#         loss = criterion(outputs,batch_y.long())

#         training_loss.append(loss.item())
#         loss.backward()
#         optimizer.step()
        
#     training_loss = np.average(training_loss)
#     print('epoch: \t', epoch, '\t training loss: \t', training_loss)
    
# # prediction for training set
# prediction = []
# target = []
# permutation = torch.randperm(train_x.size()[0])
# for i in tqdm(range(0,train_x.size()[0], batch_size)):
#     indices = permutation[i:i+batch_size]
#     batch_x, batch_y = train_x[indices], train_y[indices]

#     #if torch.cuda.is_available():
#     batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()

#     with torch.no_grad():
#         output = model(batch_x)

#     softmax = torch.exp(output).cpu()
#     prob = list(softmax.numpy())
#     predictions = np.argmax(prob, axis=1)
#     prediction.append(predictions)
#     target.append(batch_y.long())
    
# # training accuracy
# accuracy = []
# for i in range(len(prediction)):
#     accuracy.append(accuracy_score(target[i],prediction[i]))
    
# print('training accuracy: \t', np.average(accuracy))

# # prediction for validation set
# prediction_val = []
# target_val = []
# permutation = torch.randperm(val_x.size()[0])
# for i in tqdm(range(0,val_x.size()[0], batch_size)):
#     indices = permutation[i:i+batch_size]
#     batch_x, batch_y = val_x[indices], val_y[indices]

#     #if torch.cuda.is_available():
#     batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()

#     with torch.no_grad():
#         output = model(batch_x)

#     softmax = torch.exp(output).cpu()
#     prob = list(softmax.numpy())
#     predictions = np.argmax(prob, axis=1)
#     prediction_val.append(predictions)
#     target_val.append(batch_y.long())
    
# # validation accuracy
# accuracy_val = []
# for i in range(len(prediction_val)):
#     accuracy_val.append(accuracy_score(target_val[i],prediction_val[i]))
    
# print('validation accuracy: \t', np.average(accuracy_val))

import os
# loading the pretrained model
os.environ['TORCH_HOME'] = r"C:\2. University of Canberra\9. Semester 2, 2023\Soft Computing\Tutorial4\data"
model = models.vgg16_bn(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# checking if GPU is available
#if torch.cuda.is_available():
#model = model.cuda()

# Add on classifier
model.classifier[6] = Sequential(
                      Linear(4096, 2))
for param in model.classifier[6].parameters():
    param.requires_grad = True

# batch_size
batch_size = 128

# extracting features for train data
data_x = []
label_x = []

inputs,labels = train_x, train_y

for i in tqdm(range(int(train_x.shape[0]/batch_size)+1)):
    input_data = inputs[i*batch_size:(i+1)*batch_size]
    label_data = labels[i*batch_size:(i+1)*batch_size]
    input_data , label_data = Variable(input_data),Variable(label_data)
    x = model.features(input_data)
    data_x.extend(x.data.cpu().numpy())
    label_x.extend(label_data.data.cpu().numpy())
    
# extracting features for validation data
data_y = []
label_y = []

inputs,labels = val_x, val_y

for i in tqdm(range(int(val_x.shape[0]/batch_size)+1)):
    input_data = inputs[i*batch_size:(i+1)*batch_size]
    label_data = labels[i*batch_size:(i+1)*batch_size]
    input_data , label_data = Variable(input_data),Variable(label_data)
    x = model.features(input_data)
    data_y.extend(x.data.cpu().numpy())
    label_y.extend(label_data.data.cpu().numpy())

# converting the features into torch format
x_train  = torch.from_numpy(np.array(data_x))
x_train = x_train.view(x_train.size(0), -1)
y_train  = torch.from_numpy(np.array(label_x))
x_val  = torch.from_numpy(np.array(data_y))
x_val = x_val.view(x_val.size(0), -1)
y_val  = torch.from_numpy(np.array(label_y))

import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.0005)

# batch size
batch_size = 128

# number of epochs to train the model
n_epochs = 30

for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
        
    permutation = torch.randperm(x_train.size()[0])

    training_loss = []
    for i in range(0,x_train.size()[0], batch_size):

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]
        
        #if torch.cuda.is_available():
        batch_x, batch_y = batch_x, batch_y.long()
        
        optimizer.zero_grad()
        # in case you wanted a semi-full example
        outputs = model.classifier(batch_x)
        loss = criterion(outputs,batch_y.long())

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)

# prediction for training set
prediction = []
target = []
permutation = torch.randperm(x_train.size()[0])
for i in tqdm(range(0,x_train.size()[0], batch_size)):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = x_train[indices], y_train[indices]

    #if torch.cuda.is_available():
    batch_x, batch_y = batch_x, batch_y.long()

    with torch.no_grad():
        output = model.classifier(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y.long())
    
# training accuracy
accuracy = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i],prediction[i]))
    
print('training accuracy: \t', np.average(accuracy))

# prediction for validation set
prediction_val = []
target_val = []
permutation = torch.randperm(x_val.size()[0])
for i in tqdm(range(0,x_val.size()[0], batch_size)):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = x_val[indices], y_val[indices]

    #if torch.cuda.is_available():
    batch_x, batch_y = batch_x, batch_y.long()

    with torch.no_grad():
        output = model.classifier(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_val.append(predictions)
    target_val.append(batch_y.long())
    
# validation accuracy
accuracy_val = []
for i in range(len(prediction_val)):
    accuracy_val.append(accuracy_score(target_val[i],prediction_val[i]))
    
print('validation accuracy: \t', np.average(accuracy_val))
