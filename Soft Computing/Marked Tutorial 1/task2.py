from sklearn.neural_network import MLPClassifier
from pandas import read_csv as rc
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix, classification_report
from seaborn import set, heatmap

# Loading the data
dataset_init = rc(r'./data/card.csv')
dataset = dataset_init.values
X = dataset[:, 0:30]  # Attributes
Y = dataset[:, 30]  # True outputs

hidden_layers = 5
learning_rate = 0.1
epoch = 14
random_state = 42


# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting data into train, validation, and test sets (33-33-33)
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Training set
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layers,), activation='relu', solver='adam', learning_rate_init=learning_rate, alpha=0.0001, max_iter=epoch, random_state=random_state)
mlp.fit(X_train, Y_train)


accuracy_train = mlp.score(X_train, Y_train)
print("Training set accuracy : ", accuracy_train)

# Validation set
predictions_val = mlp.predict(X_val)
accuracy_val = mlp.score(X_val, Y_val)
print("Validation set accuracy : ", accuracy_val)

# Test set
predictions_test = mlp.predict(X_test)
accuracy_test = mlp.score(X_test, Y_test)
print("Test set accuracy : ", accuracy_test)

# Calculate the confusion matrix
cm = confusion_matrix(Y_test, predictions_test)

# Calculate the classification report
cr = classification_report(Y_test, predictions_test)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Print the classification report
print("Classification Report:")
print(cr)

# Plot the confusion matrix as a heatmap
set(font_scale=1.4)  # for label size
heatmap(cm, annot=True, annot_kws={"size": 20})
pyplot.xlabel("Predicted Output")
pyplot.ylabel("True Output")
pyplot.show()

# #Confusion matrix for test set
# cm = confusion_matrix(Y_test, predictions_test, labels=[0, 1])
# df_cm = df(cm, range(2), range(2))
# set(font_scale=1.4)  # for label size
# heatmap(df_cm, annot=True, annot_kws={"size": 20})
# pyplot.xlabel("Predicted Output")
# pyplot.ylabel("True Output")
# pyplot.show()
