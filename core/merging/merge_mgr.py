import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
import random

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from merging.merge_algo import MergeAlgorithm

random.seed(Config.seed)
np.random.seed(Config.seed)

class MergeManager(object):
    """
        Function Merging Manager
    """
    def __init__(self, config=Config) -> None:
        self._config = config
        self._merge_algo = MergeAlgorithm(self._config)