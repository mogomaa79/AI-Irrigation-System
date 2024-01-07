"""Test module for making sure model package is working"""

import dataset
import models
from models import KNNModel, RandomForestModel


def test_build():
    """Make sure pytest is working."""
    assert True
    assert dataset
    assert models
    assert KNNModel
    assert RandomForestModel
