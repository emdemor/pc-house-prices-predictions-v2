import os
import yaml
import logging
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine
from basix.parquet import write as to_parquet

from base.commons import load_yaml


def load():
    """load dataset"""

    filepaths = load_yaml(os.getenv("FILEPATHS"))

    logging.info("load dataset")

    pass
