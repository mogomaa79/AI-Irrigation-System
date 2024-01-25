"""Models module is used for all models in the system"""

import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dataset.dataset import DataLoader


class Model:
    """
    Generic model class for training and evaluating a classifier model.
    """
    def __init__(self, path, imputer=None):
        self.data_loader = DataLoader(path, imputer)
        self.model = None
        self.accuracy = 0

    def train_model(self):
        """Train the classifier model and evaluate its accuracy."""
        if self.model is None:
            raise NotImplementedError("Model not implemented.")
        x_train, x_test, y_train, y_test = self.data_loader.prepare_data()

        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        return self.accuracy

    def show_params(self):
        """Display model parameters."""
        if self.model is None:
            raise NotImplementedError("Model not implemented.")
        print(f"{self.__class__.__name__} Model Parameters:")
        for param, value in self.model.get_params().items():
            print(f"{param}: {value}")
        print(f'{self.__class__.__name__} Model Accuracy: {self.accuracy * 100:.2f}%')

    def save_model(self, filename):
        """Save the trained model."""
        joblib.dump(self.model, filename)


class KNNModel(Model):
    """
    KNNModel class for training and evaluating a K-Nearest Neighbors classifier.
    """
    def __init__(self, path, imputer=None, n_neighbors=84):
        super().__init__(path, imputer)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)


class RandomForestModel(Model):
    """
    RandomForestModel class for training and evaluating a Random Forest classifier.
    """
    def __init__(self, path, imputer=None, n_estimators=290, random_state=42):
        super().__init__(path, imputer)
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


class XGBoostModel(Model):
    """
    XGBoostModel class for training and evaluating a XGBoost classifier.
    """
    def __init__(self, path, imputer=None):
        super().__init__(path, imputer)
        self.model = XGBClassifier()
