import random

import numpy as np
import pytest


@pytest.fixture(scope="function", autouse=True)
def set_random_state():
    """Auto-use fixture to ensure deterministic results of unit tests."""
    np.random.seed(2022)
    random.seed(2022)
