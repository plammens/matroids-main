import logging
import random
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")
