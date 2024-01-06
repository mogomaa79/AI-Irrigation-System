import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset import prepare_data, path

def knn(X_train, X_test, y_train, y_test, n_neighbors=84, *args, **kwargs):
        """
        Train a K-Nearest Neighbors classifier and evaluate its accuracy.

        Params:
        - X_train: Training features
        - X_test: Testing features
        - y_train: Training labels
        - y_test: Testing labels
        - n_neighbors: Number of neighbors for KNN (Measured Best: 84)

        Ret:
        - knn_model: Trained KNN model
        - accuracy: Accuracy of the model on the test set
        """
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return knn_model, accuracy

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(path)

    # K (Adjustable Variable)
    n_neighbors = 84
    knn_model, accuracy = knn(X_train, X_test, y_train, y_test)

    print("KNN Model Parameters:")
    for param, value in knn_model.get_params().items():
        print(f"{param}: {value}")

    print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model as "model.pkl"
    joblib.dump(knn_model, "models/knn_model.pkl")
