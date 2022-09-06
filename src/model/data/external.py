import os
import pandas as pd
from base.commons import load_yaml


def add_external_data(data: pd.DataFrame) -> pd.DataFrame:

    filepaths = load_yaml(os.getenv("FILEPATHS"))

    data = (
        data.pipe(_add_neighbor_region, filepaths=filepaths)
        .pipe(_add_region_population, filepaths=filepaths)
        .pipe(_add_min_income_pct, filepaths=filepaths)
        .pipe(_add_avg_income, filepaths=filepaths)
        .pipe(_add_literacy_rate, filepaths=filepaths)
    )

    return data


def _add_neighbor_region(data: pd.DataFrame, filepaths: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_neighbor = pd.read_csv(filepaths["data_external_neighbor_path"])
    df_neighbor = df_neighbor.loc[~df_neighbor["neighborhood"].duplicated()]
    data = data.merge(df_neighbor, on="neighborhood", how="left")

    return data


def _add_region_population(data: pd.DataFrame, filepaths: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_pop = pd.read_csv(filepaths["data_external_region_pop_path"])
    df_pop = df_pop.loc[~df_pop["neighbor_region"].duplicated()]
    data = data.merge(df_pop, on="neighbor_region", how="left")

    return data


def _add_min_income_pct(data: pd.DataFrame, filepaths: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_minc = pd.read_csv(filepaths["data_external_region_min_income_pct_path"])
    df_minc = df_minc.loc[~df_minc["neighbor_region"].duplicated()]
    data = data.merge(df_minc, on="neighbor_region", how="left")

    return data


def _add_avg_income(data: pd.DataFrame, filepaths: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_inc = pd.read_csv(filepaths["data_external_region_avg_income_path"])
    df_inc = df_inc.loc[~df_inc["neighbor_region"].duplicated()]
    data = data.merge(df_inc, on="neighbor_region", how="left")

    return data


def _add_literacy_rate(data: pd.DataFrame, filepaths: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_lit = pd.read_csv(filepaths["data_external_region_literacy_path"])
    df_lit = df_lit.loc[~df_lit["neighbor_region"].duplicated()]
    data = data.merge(df_lit, on="neighbor_region", how="left")

    return data
