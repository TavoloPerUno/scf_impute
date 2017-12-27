from dstk.imputation import DefaultImputer
import os
import pandas as pd
import numpy as np
import re
from predictive_imputer import predictive_imputer
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

def rforest_impute(dct_data, dct_param):
    df_raw_data = dct_data['df_raw_data']
    imputer = predictive_imputer.PredictiveImputer(f_model='RandomForest')
    filled_in = imputer.fit(df_raw_data).transform(df_raw_data.copy())
    df_filled_in = pd.DataFrame(data=filled_in, columns=df_raw_data.columns, index=df_raw_data.index)
    return df_filled_in

def xgboost_impute(dct_data, dct_param):
    df_raw_data = dct_data['df_raw_data']

    imputer = DefaultImputer(missing_string_marker='nan')  # treat 'UNKNOWN' as missing value
    filled_in = imputer.fit(df_raw_data).transform(df_raw_data)

    return filled_in

def knn_impute(dct_data, dct_param, k):
    df_raw_data = dct_data['df_raw_data']
    df_raw_data.replace('nan', np.nan)
    filled_in = KNN(k=k).complete(df_raw_data)

    df_filled_in = pd.DataFrame(data=filled_in, columns=df_raw_data.columns, index=df_raw_data.index)

    return df_filled_in


