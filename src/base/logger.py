import os
import logging
import yaml
import pandas as pd
from datetime import datetime
from base.commons import load_yaml


def set():
    """Set a logging instance

    Returns
    -------
    logging
        Instance of logging
    """

    filepaths = load_yaml(os.getenv("FILEPATHS"))

    filename = os.path.join(filepaths["logs_directory_path"], "history.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(filename), logging.StreamHandler()],
    )

    return logging.getLogger()

def log_series(series: pd.Series, message: str) -> None:
    """Log a pandas series with a message as title.

    Parameters
    ----------
    series : pd.Series
        Pandas series to be logged
    message : str
        Log title
    """    
    msg = message + ":\n"
    msg += str(series)
    msg = msg.replace('\n', '\n\t') 
    logging.info(msg)