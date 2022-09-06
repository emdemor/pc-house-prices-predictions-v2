import os
import yaml
import logging
from base.commons import load_yaml


def optimize():
    """Optimize model hyperparameters"""

    filepaths = load_yaml(os.getenv("FILEPATHS"))
    logging.info("Optimize model hyperparameters")

    pass