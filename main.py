from models.models import RandomForestModel, KNNModel
from sklearn.impute import SimpleImputer

path = "data/raw/TARP.csv"

# Create an imputer object for handling missing values
imputer = SimpleImputer()

# K-Nearest Neighbors (KNN) without imputation
knn = KNNModel(path, n_neighbors=84)
acc_knn = knn.train_model()

print("Unimputed Results: ")

# Display KNN model parameters
knn.show_params()

# Print accuracy for KNN without imputation
print(f"KNN Model Accuracy: {acc_knn * 100:.2f}%")

# Random Forest without imputation
rf = RandomForestModel(path, n_estimators=290, random_state=42)
acc_rf = rf.train_model()

# Display Random Forest model parameters
rf.show_params()

# Print accuracy for Random Forest without imputation
print(f'Random Forest Model Accuracy: {acc_rf * 100:.2f}%')

# K-Nearest Neighbors (KNN) with imputation
knn_imputed = KNNModel(path, imputer, 20)
acc_knn_imputed = knn_imputed.train_model()

print("\nImputed Results: ")

# Display KNN model parameters
knn_imputed.show_params()

# Print accuracy for KNN with imputation
print(f"KNN Model Accuracy: {acc_knn_imputed * 100:.2f}%")

# Random Forest with imputation
rf_imputed = RandomForestModel(path, imputer, 180, 42)
acc_rf_imputed = rf_imputed.train_model()

# Display Random Forest model parameters
rf_imputed.show_params()

# Print accuracy for Random Forest with imputation
print(f'Random Forest Model Accuracy: {acc_rf_imputed * 100:.2f}%')
