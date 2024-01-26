"""Test module for making sure model package is working"""

from siri.dataset import DataLoader
from siri.models import KNNModel, RandomForestModel


def test_build():
    """Make sure pytest is working."""
    assert True
    assert DataLoader
    assert KNNModel
    assert RandomForestModel
