import os
import yaml
import logging
import random
import numpy as np
import torch


def load_config(path="src/config.yaml"):
    """
    Load YAML configuration file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_directory(path):
    """
    Create directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def setup_logging(log_file="logs/app.log"):
    """
    Configure logging for the project.
    """
    create_directory(os.path.dirname(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger()
