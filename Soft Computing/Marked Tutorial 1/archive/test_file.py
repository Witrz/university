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
dataset_init = rc(r'./data/Vote.csv')
dataset = dataset_init.values
X = dataset[:, 0:16]  # Attributes
Y = dataset[:, 16]  # True outputs

# Standardizing the data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Splitting data into train, validation, and test sets (33-33-33)


hidden_layer_size = 20
learning_rate = 0.1
random_state = 42

X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.33, random_state=random_state)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=random_state)
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
    mlp.partial_fit(X_train, Y_train, classes=[1, 0])

    # Calculate and store training accuracy
    train_accuracy = mlp.score(X_train, Y_train)
    train_accuracies.append(train_accuracy)

    # Calculate and store validation accuracy
    val_accuracy = mlp.score(X_val, Y_val)
    val_accuracies.append(val_accuracy)

    test_accuracy = mlp.score(X_test, Y_test)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot accuracy vs. epochs
epochs = np.arange(1, num_epochs + 1)
pyplot.plot(epochs, train_accuracies, label='Training Accuracy')
pyplot.plot(epochs, val_accuracies, label='Validation Accuracy')
pyplot.plot(epochs, test_accuracies, label='Test Accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.title('Accuracy vs. Epochs')
pyplot.show()



    



