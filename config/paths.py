# /config/paths.py

import os
from config import config

def get_experiment_root():
    """
    E.g. 'results/digits_SVM' or 'results/mnist_MLP'
    """
    return os.path.join("results", f"{config.dataset_name}_{config.model_name}")
