from base import logger


def generate_eda_report(X):

    logger.log_series(X.isna().mean(), message="Missing value rates")
