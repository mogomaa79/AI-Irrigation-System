from dataset import path, prepare_data
from knn import knn
from random_forest import random_forest
from sklearn.impute import SimpleImputer

# Create an imputer object for handling missing values
imputer = SimpleImputer()

# Load data without imputation
X_train, X_test, y_train, y_test = prepare_data(path)

# Load data with imputation
im_X_train, im_X_test, im_y_train, im_y_test = prepare_data(path, imputer)

# K-Nearest Neighbors (KNN) without imputation
n_neighbors = 84
knn_model, accuracy = knn(X_train, X_test, y_train, y_test, n_neighbors)

print("Unimputed Results: ")

# Print accuracy for KNN without imputation
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

# Random Forest without imputation
n_estimators = 290
random_state = 42
random_forest_model, accuracy = random_forest(X_train, X_test, y_train, y_test, n_estimators, random_state)

# Print accuracy for Random Forest without imputation
print(f'Random Forest Model Accuracy: {accuracy * 100:.2f}%')

# K-Nearest Neighbors (KNN) with imputation
n_neighbors = 20
knn_model, accuracy = knn(im_X_train, im_X_test, im_y_train, im_y_test, n_neighbors)

print("\nImputed Results: ")

# Print accuracy for KNN with imputation
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

# Random Forest with imputation
n_estimators = 180
random_forest_model, accuracy = random_forest(im_X_train, im_X_test, im_y_train, im_y_test, n_estimators, random_state)

# Print accuracy for Random Forest with imputation
print(f'Random Forest Model Accuracy: {accuracy * 100:.2f}%')
