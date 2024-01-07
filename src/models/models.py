import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataset.dataset import DataLoader

class KNNModel:
    def __init__(self, path, imputer=None, n_neighbors=84):
        self.data_loader = DataLoader(path, imputer)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train_model(self):
        """Train a K-Nearest Neighbors classifier and evaluate its accuracy."""
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data()

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def show_params(self):
        """Display KNN model parameters."""
        print("KNN Model Parameters:")
        for param, value in self.model.get_params().items():
            print(f"{param}: {value}")

    def save_model(self, model, filename):
        """Save the trained model."""
        joblib.dump(model, filename)

class RandomForestModel:
    def __init__(self, path, imputer=None, n_estimators=290, random_state=42):
        self.data_loader = DataLoader(path, imputer)
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train_model(self):
        """Train a Random Forest classifier and evaluate its accuracy."""
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data()

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def show_params(self):
        """Display Random Forest model parameters."""
        print("Random Forest Model Parameters:")
        for param, value in self.model.get_params().items():
            print(f"{param}: {value}")

    def save_model(self, filename):
        """Save the trained model."""
        joblib.dump(self.model, filename)

