import logging
from itertools import product
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_validate


def supervised_test_rf(X_train, X_test, y_train, y_test, random_state=42):

    regressor = TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=random_state),
        func=lambda x: x,
        inverse_func=lambda x: x,
    )

    cv_results = cross_validate(
        regressor, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )

    cv_scores = -np.array(cv_results["test_score"])

    logging.info(
        "cv_mae_score = {:.0f} ± {:.0f}".format(np.mean(cv_scores), np.std(cv_scores))
    )

    regressor.fit(X_train, y_train)

    logging.info(
        "MAE (train) = {:.3f}".format(
            mean_absolute_error(y_train, regressor.predict(X_train))
        )
    )
    logging.info(
        "MAE (test) = {:.3f}".format(
            mean_absolute_error(y_test, regressor.predict(X_test))
        )
    )
    logging.info(
        "R2 (train) = {:.3f}".format(r2_score(y_train, regressor.predict(X_train)))
    )
    logging.info(
        "R2 (test) = {:.3f}".format(r2_score(y_test, regressor.predict(X_test)))
    )
