import pandas as pd
import numpy as np
from dataset.dataset import DataLoader
from sklearn.impute import SimpleImputer
import pytest

path = "data/raw/TARP.csv"


@pytest.fixture
def data_loader():
    return {"unimputed": DataLoader(path), "imputed": DataLoader(path, SimpleImputer())}


@pytest.mark.parametrize("state", ["unimputed", "imputed"])
def test_load_data(data_loader, state):
    data = data_loader[state].load_data()

    assert isinstance(data, pd.DataFrame)
    assert not data.isnull().values.any()


@pytest.mark.parametrize("state", ["unimputed", "imputed"])
def test_prepare_data(data_loader, state):
    X_train, X_test, y_train, y_test = data_loader[state].prepare_data()

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
