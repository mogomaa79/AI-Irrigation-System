"""Models module is used for all models in the system"""

import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from siri.dataset import DataLoader


class Model():
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

    def predict(self, data):
        """Predict the class of the given data."""
        if self.model is None:
            raise NotImplementedError("Model not implemented.")
        return self.model.predict(data)


class KNNModel(Model):
    """
    KNNModel class for training and evaluating a K-Nearest Neighbors classifier.
    """
    def __init__(self, path, imputer=None, **kwargs):
        super().__init__(path, imputer)
        self.model = KNeighborsClassifier(**kwargs)


class RandomForestModel(Model):
    """
    RandomForestModel class for training and evaluating a Random Forest classifier.
    """
    def __init__(self, path, imputer=None, **kwargs):
        super().__init__(path, imputer)
        self.model = RandomForestClassifier(**kwargs)


class XGBoostModel(Model):
    """
    XGBoostModel class for training and evaluating a XGBoost classifier.
    """
    def __init__(self, path, imputer=None, **kwargs):
        super().__init__(path, imputer)
        self.model = XGBClassifier(**kwargs)


class Factory:
    """
    Factory class for creating a model.
    """
    @staticmethod
    def create_model(model_name, path, imputer=None, **kwargs):
        """Create a model."""
        if model_name == 'knn':
            return KNNModel(path, imputer, **kwargs)
        if model_name == 'random_forest':
            return RandomForestModel(path, imputer, **kwargs)
        if model_name == 'xgboost':
            return XGBoostModel(path, imputer, **kwargs)
        raise NotImplementedError(f"{model_name} model not implemented.")

    @staticmethod
    def create_model_from_file(filename):
        """Create a model from a file."""
        return joblib.load(filename)
