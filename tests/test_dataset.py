"""Test for Dataset Module"""

import pandas as pd
import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from dataset.dataset import DataLoader

PATH = "data/raw/TARP.csv"


@pytest.fixture
def data_loader():
    """
    Fixture to create DataLoader instances for unimputed and imputed data.

    Returns:
        dict: Dictionary with keys 'unimputed' and 'imputed' containing DataLoader instances.
    """
    return {"unimputed": DataLoader(PATH), "imputed": DataLoader(PATH, SimpleImputer())}


@pytest.mark.parametrize("state", ["unimputed", "imputed"])
def test_load_data(data_loader, state):
    """
    Test the load_data method of DataLoader.

    Args:
        data_loader (dict): DataLoader instances.
        state (str): State of the data (unimputed or imputed).
    """
    data = data_loader[state].load_data()

    assert isinstance(data, pd.DataFrame)
    assert not data.isnull().values.any()


@pytest.mark.parametrize("state", ["unimputed", "imputed"])
def test_prepare_data(data_loader, state):
    """
    Test the prepare_data method of DataLoader.

    Args:
        data_loader (dict): DataLoader instances.
        state (str): State of the data (unimputed or imputed).
    """
    x_train, x_test, _, _ = data_loader[state].prepare_data()

    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
