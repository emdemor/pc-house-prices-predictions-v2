import os
import yaml
import logging
import pandas as pd
from base.commons import load_yaml
from model.data import sanitize


REFERENCE_DATE = pd.to_datetime("2021-01-01")


def build_features(X, **kwargs):
    """ """
    X = (
        X.pipe(_decompose_date_ymd, date_column=kwargs["date_column"])
        .pipe(_add_day_of_week, date_column=kwargs["date_column"])
        .pipe(_add_passing_days)
    )

    return X


# def select(X, features):
#     """ """
#     X = X[features]
#     return X


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Applies the steps of feature engineering

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with the built features
    """

    variables = load_yaml(os.getenv("VARIABLES"))

    date_column = variables["date_column"]

    X = X.pipe(sanitize).pipe(build_features, date_column=date_column)

    return X


def _decompose_date_ymd(dataframe: pd.DataFrame, date_column: str):
    """Decomposes date column into three columns of year, month and day.
    Parameters
    ----------
    dataframe : pd.DataFrame
        Pandas dataframe containing a date column.
    date_column : str
        A column containing date in datetime format.
    Returns
    -------
    pd.DataFrame
        Returns the dataframe with date decomposed into three columns.
    """

    dataframe = dataframe.assign(
        year=dataframe[date_column].dt.year,
        month=dataframe[date_column].dt.month,
        day=dataframe[date_column].dt.day,
    )

    return dataframe


def _add_day_of_week(dataframe: pd.DataFrame, date_column: str):
    """Adds a column to a dataframe containing the day of week
    correspondent to the date.
    Parameters
    ----------
    dataframe : pd.DataFrame
        Pandas dataframe containing a date column.
    date_column : str
        A column containing date in datetime format.
    Returns
    -------
    pd.DataFrame
        Returns the dataframe containing a column indicating the day of week.
    """
    week_day_order = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun",
    }

    dataframe = dataframe.assign(
        day_of_week=dataframe[date_column].dt.dayofweek.replace(week_day_order)
    )

    return dataframe


def _add_passing_days(dataframe: pd.DataFrame):

    passing_days = (
        pd.to_datetime(dataframe["search_date"].astype("M8[ms]")) - REFERENCE_DATE
    ).dt.days

    dataframe = dataframe.assign(passing_days=passing_days)
    return dataframe
