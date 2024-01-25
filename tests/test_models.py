"""Test the models module."""

from models.models import KNNModel, RandomForestModel, XGBoostModel

PATH = "data/raw/TARP.csv"


def test_knn_model():
    """
    Test the KNNModel class by training the model and asserting the accuracy.
    """
    model = KNNModel(path=PATH)
    accuracy = model.train_model()
    assert accuracy >= 0.75


def test_random_forest_model():
    """
    Test the RandomForestModel class by training the model and asserting the accuracy.
    """
    model = RandomForestModel(path=PATH)
    accuracy = model.train_model()
    assert accuracy >= 0.85


def test_xgboost_model():
    """
    Test the XGBoostModel class by training the model and asserting the accuracy.
    """
    model = XGBoostModel(path=PATH)
    accuracy = model.train_model()
    assert accuracy >= 0.9
