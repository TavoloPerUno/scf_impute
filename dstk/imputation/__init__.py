from xgboost.sklearn import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
from dstk.imputation.ml_imputation import MLImputer
from dstk.imputation.encoders import MasterExploder, StringFeatureEncoder
from dstk.imputation.utils import mask_missing


def DefaultImputer(missing_string_marker='UNKNOWN', missing_features=None, random_state=10):
    xgbClassifier = XGBClassifier(seed=random_state)
    xgbRegressor = XGBRegressor(seed=random_state)
    return MLImputer(
        base_classifier=xgbClassifier,
        base_regressor=xgbRegressor,
        feature_encoder=StringFeatureEncoder(missing_marker=missing_string_marker),
        missing_features=missing_features)


def sample_dataset():
    N = -1
    NaN = np.NaN

    datax = pd.DataFrame(dict(
        a=[1, 1, 1, 1, 0, 0, 0, 1],
        b=[N, 0, 1, 0, N, 0, 1, 0],
        c=[1, 0, 0, N, N, 1, 0, 0],
        d=np.array([NaN, NaN, 1.0, NaN, NaN, 2.14, 0.0, NaN]),
        e=[3, N, N, 3, N, 3, 3, N]
    ))
    return datax


def wet_dataset():
    data = pd.DataFrame({
        'rain': [0, 0, 1, 1, 1, -1, 0, -1],
        'sprinkler': [0, 1, 1, 0, 1, 0, 1, -1],
        'wet_sidewalk': [0, 1, 1, 1, 1, 1, -1, 0],
        'some_numeric': [1.1, np.NaN, 0.2, -0.4, 0.1, 0.2, 0.0, 3.9],
        'some_string': ['B', 'A', 'A', 'A', 'A', 'A', 'A', 'UNKNOWN']
    })
    return data
