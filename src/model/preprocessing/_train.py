import os
import yaml
import logging
from base.commons import load_yaml


def train():
    """Preprocess features"""

    filepaths = load_yaml(os.getenv("FILEPATHS"))
    logging.info("Train preprocessor")

    pass