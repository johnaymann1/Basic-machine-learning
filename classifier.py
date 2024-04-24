# This code demonstrates a basic machine learning workflow using the Iris dataset. 
# The goal is to train a machine learning model to classify different types of iris flowers based on their sepal and petal measurements.

# We'll follow these steps:
# 1. Load the Iris dataset, which contains measurements of iris flowers and their corresponding species (from sklearn.datasets).
# 2. Split the dataset into training, validation, and test sets (the train_validate_test_split function).
# 3. Train a Gaussian Naive Bayes (GNB) model on the training set (from sklearn.naive_bayes)
# 4. Evaluate the model's performance on the validation set.
# 5. Finally, assess the model's accuracy on the test set to see how well it can classify iris flowers it has never seen before( the calculate_accuracy function.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Function for custom train-validate-test split
def train_validate_test_split(data, labels, test_ratio=0.3, val_ratio=0.3):
    num_samples = len(data)
    num_test_samples = int(num_samples * test_ratio)
    num_val_samples = int(num_samples * val_ratio)
    num_train_samples = num_samples - num_test_samples - num_val_samples #the rest of the data

    # Shuffle the data and labels
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Split the data into training, validation, and test sets
    X_train = data[:num_train_samples]
    y_train = labels[:num_train_samples]
    
    X_val = data[num_train_samples:num_train_samples + num_val_samples]
    y_val = labels[num_train_samples:num_train_samples + num_val_samples]
    
    X_test = data[num_train_samples + num_val_samples:]
    y_test = labels[num_train_samples + num_val_samples:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# Calculate accuracy using the custom accuracy function for validation set
def calculate_accuracy(predicted_y, y):
    # Initialize a variable to count correct predictions
    correct_predictions = 0

    # Iterate through predicted_y and y to compare each prediction
    for i in range(len(predicted_y)):
        if predicted_y[i] == y[i]:
            correct_predictions += 1

    # Calculate the accuracy as the ratio of correct predictions to total predictions
    accuracy = correct_predictions / len(y)

    return accuracy



# Split the dataset into training, validation, and test sets
X_train, y_train, X_val, y_val, X_test, y_test = train_validate_test_split(X, y, test_ratio=0.3, val_ratio=0.3)

# Train the Naive Bayes model on the training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = gnb.predict(X_val)
# Calculate accuracy using the custom accuracy function for the validation set
validation_accuracy = calculate_accuracy(y_pred, y_val)
print("Validation Accuracy:", validation_accuracy * 100)


# Make predictions on the test set
y_pred = gnb.predict(X_test)
# Calculate accuracy using the custom accuracy function for the test set
test_accuracy = calculate_accuracy(y_pred, y_test)
print("Test Accuracy:", test_accuracy * 100)









