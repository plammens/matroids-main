import random

import pytest


random.seed(2022)
__RANDOM_STATE = random.getstate()


@pytest.fixture(scope="function", autouse=True)
def set_random_state():
    """Auto-use fixture to ensure deterministic results of unit tests."""
    random.setstate(__RANDOM_STATE)
    yield
    random.setstate(__RANDOM_STATE)
