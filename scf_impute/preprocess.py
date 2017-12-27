import os
import pandas as pd
import numpy as np
import re

def pythonize_colnames(df):
    df.columns = list(map(lambda each:re.sub('[^0-9a-zA-Z]+', '_', each).lower(), df.columns))

def prepare(dct_data, dct_param):

    df_raw_data = dct_data['df_raw_data']
    del df_raw_data['Unnamed: 0']
    df_raw_data = df_raw_data.loc[df_raw_data['yy1'] != 0,]
    df_raw_data.set_index('yy1', inplace=True)

    df_xvariables = dct_data['df_xvariables']

    pythonize_colnames(df_xvariables)
    df_xvariables['na_code'] = df_xvariables['na_code'].fillna(0)

    df_raw_data = df_raw_data[list(filter(lambda x: x.startswith('x') and
                                                    x not in list(df_xvariables[df_xvariables['nominal_character_c_or_numeric_n'].isnull()].x),
                                          df_raw_data.columns))]

    lst_char_cols = [col for col in list(df_xvariables[df_xvariables['nominal_character_c_or_numeric_n'] == 'C'].x)
                     if col in df_raw_data.columns]
    lst_year_cols = [col for col in list(df_xvariables[df_xvariables['is_year'] == 1].x)
                     if col in df_raw_data.columns]
    lst_num_cols = [col for col in list(df_xvariables[df_xvariables.nominal_character_c_or_numeric_n.isin(['N', 'P', 'I'])].x)
                    if col in df_raw_data.columns]

    for yr_col in lst_year_cols:
        df_raw_data[yr_col] = df_raw_data[yr_col].fillna(df_xvariables.loc[df_xvariables['x']==yr_col,'na_code'].values[0])

    df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].astype(str)

    cols = list(df_raw_data)
    nunique = df_raw_data.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df_raw_data.drop(cols_to_drop, axis=1, inplace=True)

    dct_data['df_raw_data'] = df_raw_data
    dct_data['df_xvariables'] = df_xvariables
    dct_data['lst_char_cols'] = lst_char_cols
    dct_data['lst_num_cols'] = lst_num_cols
    dct_data['lst_year_cols'] = lst_year_cols
    dct_data['lst_skipped_cols'] = list(cols_to_drop)

    return dct_data