"""
- Split into training and testing data
- Load the diabetes dataset
- Standardize features
- Train a linear regression model using gradient descent
- Evaluate using MSE and RMSE
- Plot training loss
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Load dataset

diabetes_dataset = load_diabetes()

X = diabetes_dataset.data      # input features
y = diabetes_dataset.target    # target values

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print("Feature names:", diabetes_dataset.feature_names)


# 2. Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# 3. Standardize features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. Define helper functions

def predict(X, weights, bias):
    """
    Makes predictions using the linear model.

    Formula:
    y_pred = X @ weights + bias
    """
    return X @ weights + bias


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error.

    Measures the average squared difference between
    true values and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse_loss(y_true, y_pred))


# 5. Initialize model parameters

n_features = X_train.shape[1]

weights = np.zeros(n_features)
bias = 0.0

print("Initial weights:", weights)
print("Initial bias:", bias)


# 6. Set hyperparameters

learning_rate = 0.01
epochs = 1000

train_losses = []


# 7. Train model with gradient descent

for epoch in range(epochs):
    # Make predictions on training data
    y_pred = predict(X_train, weights, bias)

    # Compute loss
    loss = mse_loss(y_train, y_pred)
    train_losses.append(loss)

    # Number of training examples
    n = len(y_train)

    # Compute gradients
    dw = (-2 / n) * (X_train.T @ (y_train - y_pred))
    db = (-2 / n) * np.sum(y_train - y_pred)

    # Update weights and bias
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Training MSE: {loss:.4f}")


# 8. Evaluate model on test data

test_predictions = predict(X_test, weights, bias)

test_mse = mse_loss(y_test, test_predictions)
test_rmse = rmse(y_test, test_predictions)

print("\nFinal weights:", weights)
print("Final bias:", bias)

print("\nTest MSE:", test_mse)
print("Test RMSE:", test_rmse)


# 9. Plot training loss

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.show()


# 10. Plot predicted vs actual values

plt.scatter(y_test, test_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values")
plt.show()


# Plot regression line for one feature: BMI

feature_index = diabetes_dataset.feature_names.index("bmi")

X_bmi_test = X_test[:, feature_index]

# Sort values so the line looks clean
sorted_indices = np.argsort(X_bmi_test)

X_bmi_sorted = X_bmi_test[sorted_indices]
predictions_sorted = test_predictions[sorted_indices]

plt.scatter(X_bmi_test, y_test, label="Actual Data")
plt.plot(X_bmi_sorted, predictions_sorted, label="Model Prediction")
plt.xlabel("BMI feature (standardized)")
plt.ylabel("Disease Progression")
plt.title("Linear Regression Prediction Using BMI Visualization")
plt.legend()
plt.show()