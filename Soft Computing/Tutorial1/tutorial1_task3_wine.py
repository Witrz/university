import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#matplotlib inline

# Read in white and red wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

# Print info on white wine and red wine
print(white.info())
print(red.info())
white.describe()
red.describe()


#checking for null value
pd.isnull(red).count()
pd.isnull(white).count()


#visualisation
fig, ax = plt.subplots(1, 2)
ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5,
alpha=0.5, label="White wine")
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05,
wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()
sns.pairplot(data=red, diag_kind = 'kde')
sns.pairplot(data=white, diag_kind = 'kde')


# Add `type` column to `red` with value 1
red['type'] = 1
# Add `type` column to `white` with value 0
white['type'] = 0
# Append `white` to `red`
wines = red.append(white, ignore_index=True)
wines.tail()


# Specify the data
X=wines.iloc[:,0:11]
# Specify the target labels and flatten the array
y= np.ravel(wines.type)


#y= wines.type
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
random_state=42)
y_test[0:10]
# Define the scaler
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)




# Using Tensorflow Keras instead of the original Keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# define the model architecture

# Initialize the constructor
model = Sequential()
# Add an input layer
model.add(Dense(12, activation='sigmoid', input_shape=(11,)))
# Add one hidden layer
model.add(Dense(8, activation='sigmoid'))
# Add an output layer
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])
history = model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)