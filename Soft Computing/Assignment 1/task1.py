from sklearn.neural_network import MLPClassifier
from pandas import read_csv as rc
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix, classification_report
from seaborn import set, heatmap

# """
#     Task 1: Training a basic MLP

#     This MLP is trained on the Dermatology Dataset

# """

# # Loading the data
# dataset_init = rc(r'./data/task1/dermatology.csv')
# dataset = dataset_init.values
# X = dataset[:, 0:33]  # Attributes
# Y = dataset[:, 33]  # True outputs

# hidden_layers = 5
# learning_rate = 0.1
# epoch = 9
# random_state = 42


# # Standardizing the data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Splitting data into train, validation, and test sets (33-33-33)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# # Training set
# mlp = MLPClassifier(hidden_layer_sizes=(hidden_layers,), activation='relu', solver='adam', learning_rate_init=learning_rate, alpha=0.0001, max_iter=epoch, random_state=random_state)
# mlp.fit(X_train, Y_train)


# accuracy_train = mlp.score(X_train, Y_train)
# print("Training set accuracy : ", accuracy_train)

# # Test set
# predictions_test = mlp.predict(X_test)
# accuracy_test = mlp.score(X_test, Y_test)
# print("Test set accuracy : ", accuracy_test)

# # Calculate the confusion matrix
# cm = confusion_matrix(Y_test, predictions_test)

# # Calculate the classification report
# cr = classification_report(Y_test, predictions_test)

# # Print the confusion matrix
# print("Confusion Matrix:")
# print(cm)

# # Print the classification report
# print("Classification Report:")
# print(cr)

# # Plot the confusion matrix as a heatmap
# set(font_scale=1.4)  # for label size
# heatmap(cm, annot=True, annot_kws={"size": 20})
# pyplot.xlabel("Predicted Output")
# pyplot.ylabel("True Output")
# pyplot.show()


from sklearn.neural_network import MLPClassifier
from pandas import read_csv as rc
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
from seaborn import set, heatmap
import numpy as np

## this file prints the accuracy vs epoch graph. Will need to be improved to allow for testing of multiple variables.

# Loading the data
dataset_init = rc(r'./data/task1/dermatology_adjusted.csv')
dataset = dataset_init.values
X = dataset[:, 0:34]  # Attributes
Y = dataset[:, 34]  # True outputs

# Standardizing the data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Splitting data into train, validation, and test sets (33-33-33)


hidden_layer_size = 50
learning_rate = 0.001
random_state = 42

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=random_state)
# Load your data and preprocess it (as in your provided code)

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation='relu', solver='adam', learning_rate_init=learning_rate, alpha=0.0001, max_iter=1, warm_start=True, random_state=random_state)

# Lists to store accuracy values for plotting
train_accuracies = []
val_accuracies = []
test_accuracies = []

# Training parameters
num_epochs = 1000  # You can adjust this number as needed

for epoch in range(num_epochs):
    # Train the model for one epoch
    mlp.partial_fit(X_train, Y_train, classes=[1, 2, 3, 4, 5, 6])

    # Calculate and store training accuracy
    train_accuracy = mlp.score(X_train, Y_train)
    train_accuracies.append(train_accuracy)

    test_accuracy = mlp.score(X_test, Y_test)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot accuracy vs. epochs
epochs = np.arange(1, num_epochs + 1)
pyplot.plot(epochs, train_accuracies, label='Training Accuracy')
pyplot.plot(epochs, test_accuracies, label='Test Accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.title('Accuracy vs. Epochs')
pyplot.show()
