import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
)

def custom_metric(*args, **kwargs):
    """Defining a custom metric as a didatic example"""
    return scipy.stats.hmean(
        [
            mean_absolute_error(*args, **kwargs),
            np.sqrt(np.abs(mean_squared_error(*args, **kwargs))),
        ]
    )


CUSTOM_SCORERS = {
    "customized_score": make_scorer(custom_metric, greater_is_better=False),
}
