import os
import yaml
import logging
import pandas as pd
import numpy as np
from unidecode import unidecode


from dotenv import load_dotenv
from sqlalchemy import create_engine
from basix.parquet import write as to_parquet

from base.commons import load_yaml
from src.model.data.cleaning import raw_sanitize


# EXTRACTION_QUERY = """\
# WITH tab AS (
#     SELECT
#         id
#         --, search_id
#         , search_date
#         --, id_zap
#         , type
#         , n_parking_spaces
#         , n_bathrooms
#         , n_bedrooms
#         , area
#         --, n_floors
#         --, units_on_floor
#         , n_suites
#         --, state
#         --, city
#         , neighborhood
#         --, street
#         , CASE WHEN ABS(longitude) > 0 THEN longitude ELSE NULL END as longitude
#         , CASE WHEN ABS(latitude) > 0 THEN latitude ELSE NULL END as latitude
#         , condo_fee
#         , iptu
#         --, resale
#         --, buildings
#         --, plan_only
#         , price
#     FROM pocos_de_caldas.imoveis i
#     WHERE True
#       AND price IS NOT NULL
#       AND price > 0
# ) SELECT * FROM tab t
# """

EXTRACTION_QUERY = """\
WITH tab AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY id_zap ORDER BY search_date DESC) linha
    FROM pocos_de_caldas.imoveis
)
SELECT
    id
    , search_date
    , type
    , n_parking_spaces
    , n_bathrooms
    , n_bedrooms
    , area
    , n_suites
    , neighborhood
    , CASE WHEN ABS(longitude) > 0 THEN longitude ELSE NULL END as longitude
    , CASE WHEN ABS(latitude) > 0 THEN latitude ELSE NULL END as latitude
    , condo_fee
    , iptu
    , price     
FROM tab
WHERE True
    AND price IS NOT NULL
    AND price > 0
    AND linha = 1
"""


def extract_scrapped_relational_data() -> None:
    """Extract relational data from PostgreSQl database and
    persists in a local parquet file.
    """

    assert load_dotenv()

    # --
    logging.info("Importing variables")
    filepaths = load_yaml(os.getenv("FILEPATHS"))
    variables = load_yaml(os.getenv("VARIABLES"))

    # --
    logging.info("Setting postgresql connection")
    engine = create_engine(
        "postgresql://{user}:{password}@{host}:{port}/{database}".format(
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
        )
    )

    # --
    logging.info("Reading data from PostgreSQl")
    df_basic = pd.read_sql(EXTRACTION_QUERY, engine)

    # --
    df_basic = df_basic.pipe(raw_sanitize)
    # df_basic = (
    #     df_basic.pipe(_sanitize_search_date)
    #     .pipe(_sanitize_neighborhood)
    #     .pipe(_sanitize_longitude)
    # )

    # df_basic["search_date"] = df_basic["search_date"].dt.date

    # --
    logging.info("Persisting relational raw data")
    to_parquet(
        df_basic,
        filepaths["raw_data_relational_scrapped"],
        overwrite=True,
        partition_cols=["search_date"],
    )


# def raw_sanitize(X):
#     # --
#     X = (
#         X.pipe(_sanitize_search_date)
#         .pipe(_sanitize_neighborhood)
#         .pipe(_sanitize_longitude)
#     )

#     return X


# def _sanitize_search_date(X):
#     X = X.assign(**{"search_date": pd.to_datetime(X["search_date"].astype("M8[ms]"))})
#     return X


# def _sanitize_neighborhood(X):

#     variables = load_yaml(os.getenv("VARIABLES"))

#     neighbor_replaces = variables["neighbor_replaces"]

#     X = X.assign(
#         **{
#             "neighborhood": X["neighborhood"]
#             .apply(lambda x: unidecode(x) if x is not None else x)
#             .str.strip()
#             .str.replace(" ", "_")
#             .str.lower()
#         }
#     )

#     for k in neighbor_replaces:

#         X = X.assign(
#             **{"neighborhood": X["neighborhood"].str.replace(k, neighbor_replaces[k])}
#         )

#     X = X.assign(**{"neighborhood": X["neighborhood"].astype("category")})

#     return X


# def _sanitize_longitude(X):
#     longitude_treated = np.where(X["longitude"] > -30, np.nan, X["longitude"])
#     X = X.assign(**{"longitude": longitude_treated})
#     return X
