import numpy as np
import json
import yaml
import dill as pickle


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dump_json(obj, filepath, *args, **kwargs):
    """Dump json

    Parameters
    ----------
    obj : _type_
        _description_
    filepath : _type_
        _description_
    """

    with open(filepath, "w") as file:
        json.dump(obj, file, cls=NpEncoder, *args, **kwargs)


def load_json(filepath, *args, **kwargs):
    """_summary_

    Parameters
    ----------
    filepath : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    with open(filepath, "r") as file:
        data = json.load(file, *args, **kwargs)

    return data


def dump_yaml(obj, filepath, *args, **kwargs):
    """_summary_

    Parameters
    ----------
    obj : _type_
        _description_
    filepath : _type_
        _description_
    """

    with open(filepath, "w") as file:
        yaml.dump(obj, file, *args, **kwargs)


def load_yaml(filename, *args, **kwargs):
    """_summary_

    Parameters
    ----------
    filename : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    with open(filename, "r") as file:
        data = yaml.safe_load(file, *args, **kwargs)

    return data


def dump_pickle(obj, filepath, *args, **kwargs):
    """_summary_

    Parameters
    ----------
    obj : _type_
        _description_
    filepath : _type_
        _description_
    """

    with open(filepath, "wb") as file:
        pickle.dump(obj, file, *args, **kwargs)


def load_pickle(filename, *args, **kwargs):
    """_summary_

    Parameters
    ----------
    filename : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    with open(filename, "rb") as file:
        data = pickle.load(file, *args, **kwargs)

    return data