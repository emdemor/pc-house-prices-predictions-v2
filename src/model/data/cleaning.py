import os
import pandas as pd
import numpy as np
from unidecode import unidecode

from src.base.commons import load_yaml


def sanitize(X: pd.DataFrame) -> pd.DataFrame:
    """Apply some basic transformations, as castings
    constant imputations and outlier remotions

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with transformed features
    """

    X = X.assign(
        search_date=pd.to_datetime(X["search_date"]),
        latitude=np.where(X["latitude"].isna(), X["neighbor_latitude"], X["latitude"]),
        longitude=np.where(
            X["longitude"].isna(), X["neighbor_longitude"], X["longitude"]
        ),
    )

    return X


def raw_sanitize(X):
    # --

    X = (
        X.pipe(_sanitize_search_date)
        .pipe(_sanitize_neighborhood)
        .pipe(_sanitize_longitude)
    )

    return X


def _sanitize_search_date(X):
    X = X.assign(**{"search_date": pd.to_datetime(X["search_date"].astype("M8[ms]"))})
    return X


def _sanitize_neighborhood(X):

    variables = load_yaml(os.getenv("VARIABLES"))

    neighbor_replaces = variables["neighbor_replaces"]

    X = X.assign(
        **{
            "neighborhood": X["neighborhood"]
            .apply(lambda x: unidecode(x) if x is not None else x)
            .str.strip()
            .str.replace(" ", "_")
            .str.lower()
        }
    )

    for k in neighbor_replaces:

        X = X.assign(
            **{"neighborhood": X["neighborhood"].str.replace(k, neighbor_replaces[k])}
        )

    X = X.assign(**{"neighborhood": X["neighborhood"].astype("category")})

    return X


def _sanitize_longitude(X):
    longitude_treated = np.where(X["longitude"] > -30, np.nan, X["longitude"])
    X = X.assign(**{"longitude": longitude_treated})
    return X


def _cast_neighborhood(X):
    X = X.assign(neighborhood=X["neighborhood"].astype(str))
    return X
