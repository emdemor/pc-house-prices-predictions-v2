import os
import logging
import pandas as pd

from base.commons import load_yaml, load_json, dump_json, dump_pickle
from basix.files import make_directory


def export_artifacts(model_artifacts, preprocessor, estimator, tunner):

    filepaths = load_yaml(os.getenv("FILEPATHS"))

    variables = load_yaml(os.getenv("VARIABLES"))

    artifact_filename = variables["artifact_filename"]

    preprocessor_filename = variables["preprocessor_filename"]

    estimator_filename = variables["estimator_filename"]

    model_artifact_directorypath = os.path.join(
        filepaths["model_artifacts_path"], variables["estimator"]
    )

    model_artifact_filepath = os.path.join(
        model_artifact_directorypath, artifact_filename
    )

    preprocessor_filepath = os.path.join(
        model_artifact_directorypath, preprocessor_filename
    )

    estimator_filepath = os.path.join(model_artifact_directorypath, estimator_filename)

    # -- create directory if it does not exists -------------------
    try:
        make_directory(model_artifact_directorypath)
    except Exception as err:
        logging.error(err.__str__())
        raise err

    # -- exports a model parameters --------------
    try:
        dump_json(model_artifacts, model_artifact_filepath, indent=2)
    except Exception as err:
        logging.error(err.__str__())
        raise err

    # -- exports serializable for preprocessor pipeline -----------------
    try:
        dump_pickle(preprocessor, preprocessor_filepath)
    except Exception as err:
        logging.error(err.__str__())
        raise err

    # -- exports serializable for estimator -----------------
    try:
        dump_pickle(estimator, estimator_filepath)
    except Exception as err:
        logging.error(err.__str__())
        raise err

    # -- updates metadata -----------------
    try:
        update_metadata_store(tunner)
    except Exception as err:
        logging.error(err.__str__())
        raise err


def update_model_artifacts(preprocessor, estimator, tunner, force_overwrite=False):

    filepaths = load_yaml(os.getenv("FILEPATHS"))

    variables = load_yaml(os.getenv("VARIABLES"))

    artifact_filename = variables["artifact_filename"]

    action_for_different_metrics = variables["action_for_different_metrics"]

    model_artifact_directorypath = os.path.join(
        filepaths["model_artifacts_path"], variables["estimator"]
    )

    model_artifact_filepath = os.path.join(
        model_artifact_directorypath, artifact_filename
    )

    model_artifacts = {
        "model": variables["estimator"],
        "metric": {
            "name": tunner.search_scoring,
            "value": tunner.best_score_,
        },
        "hyper_params": tunner.best_params_,
    }

    # -- if artifact file doesn`t exists
    if not os.path.exists(model_artifact_filepath):
        logging.info(
            f"The artifact file {model_artifact_filepath} do not exists. "
            f"So, lets save this model."
        )
        export_artifacts(model_artifacts, preprocessor, estimator, tunner)

    else:
        # -- read artifact file -------------
        old_config = load_json(model_artifact_filepath)

        # -- check if the metrics of new and the old models are the same ------
        if old_config["metric"]["name"] != tunner.search_scoring:

            message = (
                f"\n\tThe old model was evaluted by the metric: {old_config['metric']['name']}. "
                f"\n\tThe new model was evaluted by the metric: {tunner.search_scoring}. "
            )

            if action_for_different_metrics == "keep_old":
                logging.warning(message + f"Keep the old model.")
                force_overwrite = False

            elif action_for_different_metrics == "keep_new":
                logging.warning(message + f"Keep the new model.")
                force_overwrite = True

            else:
                raise AssertionError(
                    message
                    + f"Since the metrics are different, there is no way to compare them. "
                    f"If you want to keep the old model, change the variable `action_for_different_metrics` "
                    f"to 'keep_old' and 'keep_new' to keep the new model. "
                )

        # -- check if the current metric is better than the old one -----------
        # -- if there exists improvement, overwrite the old artifact file -------
        if (old_config["metric"]["value"] < tunner.best_score_) or force_overwrite:
            logging.info(
                f"The new model has a better performance than the old."
                f'\n\tOld model: {old_config["metric"]}'
                f'\n\tNew model: {model_artifacts["metric"]}'
                "\nSaving the model."
            )
            export_artifacts(model_artifacts, preprocessor, estimator, tunner)

        else:
            logging.warning(
                f"The new model did not perform better than the old.\n"
                f'\tOld model: {old_config["metric"]}\n'
                f'\tNew model: {model_artifacts["metric"]}\n'
                "The current model artifacts was not saved."
            )


def update_metadata_store(tunner):

    filepaths = load_yaml(os.getenv("FILEPATHS"))

    variables = load_yaml(os.getenv("VARIABLES"))

    metadata_filepath = os.path.join(
        filepaths["model_artifacts_path"],
        variables["metadata_store_filename"],
    )

    if os.path.exists(metadata_filepath):
        metadata_store = pd.read_json(metadata_filepath)
        last_model = metadata_store.iloc[-1].to_dict()
        version = last_model["version"]
        subversion = last_model["subversion"]
        identifier = last_model["identifier"] + 1
    else:
        metadata_store = pd.DataFrame()
        version = 0
        subversion = 0
        identifier = 1

    model_metadata = [
        {
            "version": version,
            "subversion": subversion,
            "identifier": identifier,
            "estimator": variables["estimator"],
            "metric_name": tunner.search_scoring,
            "metric_score": tunner.best_score_,
        }
    ]

    updated_model_metadata = pd.concat(
        [
            metadata_store,
            pd.DataFrame(model_metadata),
        ]
    ).reset_index(drop=True)

    dump_json(
        updated_model_metadata.to_dict(orient="records"),
        metadata_filepath,
        indent=2,
    )
