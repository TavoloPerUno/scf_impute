from dstk.imputation import DefaultImputer
import os
import pandas as pd
import numpy as np
import re

def xgboost_impute(dct_data, dct_param):
    df_raw_data = dct_data['df_raw_data']

    imputer = DefaultImputer(missing_string_marker='nan')  # treat 'UNKNOWN' as missing value
    filled_in = imputer.fit(df_raw_data).transform(df_raw_data)

    return filled_in