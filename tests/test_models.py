"""Test Models Module"""

import pytest
from models.models import KNNModel, RandomForestModel

PATH = "data/raw/TARP.csv"


@pytest.fixture
def knn_model():
    """Fixture to prepare KNN Model."""
    return KNNModel(PATH, n_neighbors=5)


@pytest.fixture
def random_forest_model():
    """Fixture to prepare Random Forest Model."""
    return RandomForestModel(PATH, n_estimators=10, random_state=42)


def test_knn_training(knn_model):
    """
    Test Training of KNN Model.

    This test checks if the KNN model can be trained successfully.
    """
    accuracy = knn_model.train_model()
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_knn_model_params(knn_model, capsys):
    """
    Test Show Params Method for KNN Model.

    This test checks if the show_params method of the KNN model displays parameters correctly.
    """
    knn_model.show_params()
    captured = capsys.readouterr()
    assert "KNN Model Parameters:" in captured.out


def test_knn_model_save(knn_model, tmp_path):
    """
    Test Save Model Method for KNN Model.

    This test checks if the KNN model can be saved successfully.
    """
    filename = tmp_path / "knn_model.pkl"
    knn_model.save_model(filename)
    assert filename.is_file()


def test_random_forest_model_training(random_forest_model):
    """
    Test Training of Random Forest Model.

    This test checks if the Random Forest model can be trained successfully.
    """
    accuracy = random_forest_model.train_model()
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_random_forest_model_params(random_forest_model, capsys):
    """
    Test Show Params Method for Random Forest Model.

    """
    random_forest_model.show_params()
    captured = capsys.readouterr()
    assert "Random Forest Model Parameters:" in captured.out


def test_random_forest_model_save(random_forest_model, tmp_path):
    """
    Test Save Model Method for Random Forest Model.

    This test checks if the Random Forest model can be saved successfully.
    """
    filename = tmp_path / "random_forest_model.pkl"
    random_forest_model.save_model(filename)
    assert filename.is_file()
