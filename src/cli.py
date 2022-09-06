# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from yaml import load
import os
from model import data, features, preprocessor, regressor

from base import logger


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    data.load()
    features.build()
    preprocessor.train()
    regressor.train()


if __name__ == "__main__":

    load_dotenv(find_dotenv())

    logger.set()

    main()
