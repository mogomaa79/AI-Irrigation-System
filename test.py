from dataset import path, prepare_data
from knn import knn
from random_forest import random_forest
from sklearn.impute import SimpleImputer

"""
After Testing
With Un-imputed data (2200 Rows): 
    Best K: 84 --> 75.68% Accuracy
    Best N_Estimators: 290 --> 90.23% Accuracy

With Imputed Data (100000 Rows)
    Best K: 20 --> 90.98% Accuracy
    Best N_Estimators: 180 --> 99.61% Accuracy
"""

def test_knn(X_train, X_test, y_train, y_test, imputer, k_range):
    """Tests k values through a range to find the most accuracy"""
    high = 0
    k = 0

    for i in k_range:
        knn_model, accuracy = knn(X_train, X_test, y_train, y_test, i, imputer)
        if accuracy > high:
            k = i
            high = accuracy

    return k, high

def test_random_forest(X_train, X_test, y_train, y_test, n_estimators_range):
    """tests n_estimators to find most accurate"""
    high = 0
    n_estimators = 0

    for i in n_estimators_range:
        random_forest_model, accuracy = random_forest(X_train, X_test, y_train, y_test, i, 42)
        if accuracy > high:
            n_estimators = i
            high = accuracy

    return n_estimators, high

if __name__ == "__main__":
    # Load data with imputation
    imputer = SimpleImputer()
    X_train, X_test, y_train, y_test = prepare_data(path, imputer)

    # Specify the ranges to search through
    knn_range = range(20, 22)
    rf_n_estimators_range = range(180, 182, 10)

    # Find the best K for KNN
    best_k, best_k_accuracy = test_knn(X_train, X_test, y_train, y_test, imputer, knn_range)
    print(f"Best K for KNN: {best_k} with accuracy {best_k_accuracy * 100:.2f}%")

    # Find the best number of estimators for Random Forest
    n_estimators, best_rf_accuracy = test_random_forest(X_train, X_test, y_train, y_test, rf_n_estimators_range)
    print(f"Best Number of Estimators for Random Forest: {n_estimators} with accuracy {best_rf_accuracy * 100:.2f}%")