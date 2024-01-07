"""Test module for making sure model package is working"""

from dataset.dataset import DataLoader
from models.models import KNNModel, RandomForestModel


def test_build():
    """Make sure pytest is working."""
    assert True
    assert DataLoader
    assert KNNModel
    assert RandomForestModel
