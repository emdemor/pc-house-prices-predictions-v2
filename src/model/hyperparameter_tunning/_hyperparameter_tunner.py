import logging
import sklearn
import os
import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError
from basix.files import make_directory
from base.commons import load_yaml

from ._custom_scorers import CUSTOM_SCORERS
from ._searchers import *
from ..estimators._estimators import MODEL_MAPS


class HyperparameterTunner:
    def __init__(self):

        self.search_results = None
        self.fitted = False
        self.model = None
        self.estimator = None
        self.parametric_space = None
        self.search_strategy = None
        self.search_samples = None
        self.search_n_iter = None
        self.search_scoring = None
        self.search_cv = None
        self.search_verbose = None
        self.best_estimator_ = None
        self.best_index_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.searcher = self.__get_hyperparameter_searcher()

    def __str__(self):
        return "HyperparameterTunner()"

    def __repr__(self):
        return "HyperparameterTunner()"

    def fit(self, X, y):

        try:
            self.searcher.fit(X, y)
            self.best_estimator_ = self.searcher.best_estimator_
            self.best_index_ = self.searcher.best_index_
            self.best_params_ = self.searcher.best_params_
            self.best_score_ = self.searcher.best_score_
            self.fitted = True

        except Exception as err:
            self.fitted = False
            logging.error(err)
            raise err

        self.get_search_results()

    def get_search_results(self):

        if not self.fitted:
            raise NotFittedError(self.__str__() + " not fitted.")

        else:
            temp = pd.DataFrame(self.searcher.cv_results_)

            self.search_results = temp.assign(
                rmse=np.sqrt(-temp["mean_test_score"])
            ).sort_values("rank_test_score")

    def __get_hyperparameter_searcher(self):

        # -- get global variables -------------------
        variables = load_yaml(os.getenv("VARIABLES"))

        # -- get regressor model --------------------
        self.model = variables["estimator"]

        # -- import model config files --------------
        model_config = load_yaml(f"config/models/{self.model}.yaml")

        # -- create direcotory if it does not exists -------------------
        make_directory(
            os.path.join(*model_config["parametric_space_path"].split("/")[:-1])
        )

        # -- instantiate model -------------------
        self.estimator = self.__interpret_model(model_config)

        # -- interpret parametric space --------------------
        self.parametric_space = model_config["parametric_space"] = {
            param: eval(model_config["parametric_space"][param])
            for param in model_config["parametric_space"]
        }

        # -- get static params --------------------
        static_parameters = model_config["static_parameters"]

        # -- choose hyperparameter strategy -------------------
        self.search_strategy = model_config["hyperparameter_tunning"]["strategy"]
        self.search_samples = model_config["hyperparameter_tunning"]["random_samples"]
        self.search_n_iter = model_config["hyperparameter_tunning"]["n_iter"]
        self.search_scoring = model_config["hyperparameter_tunning"]["scoring"]
        self.search_cv = model_config["hyperparameter_tunning"]["cv"]
        self.search_verbose = model_config["hyperparameter_tunning"]["verbose"]

        if self.search_strategy == "RandomizedSearch":
            searcher = RandomizedSearchCV
            space = {
                k: self.parametric_space[k].rvs(self.search_samples)
                for k in self.parametric_space
            }

        elif self.search_strategy == "BayesSearchCV":
            searcher = BayesSearchCV
            space = self.parametric_space

        # -- If necessary, define here new search_strategies. (e.g. GridSearchCV)
        # ...

        # -- Run hyperparametric search ---------------------
        search_results = searcher(
            self.estimator,
            space,
            n_iter=self.search_n_iter,
            scoring=self.__defining_scoring(self.search_scoring),
            cv=self.search_cv,
            verbose=self.search_verbose,
        )
        return search_results

    def __defining_scoring(self, scoring: str):

        if scoring in sklearn.metrics.SCORERS.keys():
            return scoring

        # -- custom scorers
        elif scoring in CUSTOM_SCORERS:
            return CUSTOM_SCORERS[scoring]

        else:
            raise ValueError(
                f"It was not possible to interpret scorer `{scoring}`. "
                "The allowed values are that on `sklearn.metrics.SCORERS.keys()` "
                "or a custom scorer you can define on module `model.optimizer`."
            )

    def __interpret_model(self, model_config):

        model_label = model_config["model"]

        # -- check if the model description is mapped into a model -------------------
        if model_label not in MODEL_MAPS:
            error_mesage = f"The model {model_label} could not be interpreted."
            logging.error(error_mesage)
            raise ValueError(error_mesage)

        # -- instantiate model class --------------------
        model = MODEL_MAPS[model_label]()

        # -- get default parametes -------------------
        params = model.get_params()

        # -- get static hyperparameters set by user ---------------------
        # static hyperparameters are hyperparameters fixed by the user that
        # will not be considered on optimization process
        if "static_parameters" in model_config:
            static_parameters = model_config["static_parameters"]
        else:
            static_parameters = {}

        params.update(static_parameters)

        model.set_params(**params)

        return model
