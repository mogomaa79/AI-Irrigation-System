import pytest
from models.models import KNNModel, RandomForestModel

path = "data/raw/TARP.csv"


@pytest.fixture
def knn_model():
    return KNNModel(path, n_neighbors=5)


@pytest.fixture
def random_forest_model():
    return RandomForestModel(path, n_estimators=10, random_state=42)


def test_knn_model_training(knn_model):
    accuracy = knn_model.train_model()
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_knn_model_params(knn_model, capsys):
    knn_model.show_params()
    captured = capsys.readouterr()
    assert "KNN Model Parameters:" in captured.out


def test_knn_model_save(knn_model, tmp_path):
    filename = tmp_path / "knn_model.pkl"
    knn_model.save_model(filename)
    assert filename.is_file()


def test_random_forest_model_training(random_forest_model):
    accuracy = random_forest_model.train_model()
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_random_forest_model_params(random_forest_model, capsys):
    random_forest_model.show_params()
    captured = capsys.readouterr()
    assert "Random Forest Model Parameters:" in captured.out


def test_random_forest_model_save(random_forest_model, tmp_path):
    filename = tmp_path / "random_forest_model.pkl"
    random_forest_model.save_model(filename)
    assert filename.is_file()
