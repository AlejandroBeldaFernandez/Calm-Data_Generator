import river
import pytest


def test_river_import():
    print(f"River version: {river.__version__}")
    assert river.__version__ is not None
