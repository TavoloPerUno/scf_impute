import os
import re
import sklearn.model_selection
import pandas as pd
import numpy as np
import random
import itertools

def key_val_products(dicts):
    return ([item for sublist in [list(itertools.product([k], dicts[k])) for k in list(dicts.keys())] for item in sublist])

def pythonize_colnames(df):
    df.columns = list(map(lambda each:re.sub('[^0-9a-zA-Z]+', '_', each).lower(), df.columns))


def track_holdout(dct_data, dct_param):
    df_raw_data = dct_data['df_raw_data']


    train_id, test_id = sklearn.model_selection.train_test_split(df_raw_data.index, train_size=0.8, random_state=10)



    # 1. Read a row
    # 2. Pick 10 or less numeric values that exist
    # 3. Record the row number and associated 10 or less columns
    # 4. Make the 10 or less numeric values missing
    # 5. Next row

    dct_removed = {}

    for i in test_id:
        # Select a row
        row = df_raw_data.loc[i,]

        # Find columns with data
        existing_columns = row.index[row.notnull()]

        existing_columns = pd.Index([col for col in existing_columns if col in dct_data['lst_char_cols'] + dct_data['lst_num_cols']])

        # Miniumum of length of existing columns or 10
        num_valuesdropped = min(len(existing_columns), 10)

        # Create list of indices and then randomly select values that will be dropped for analysis
        existing_columns_indices = list(range(len(existing_columns)))
        random.seed(10 + i)
        columns_dropped = existing_columns[random.sample(existing_columns_indices, num_valuesdropped)]

        dct_removed[i] = columns_dropped

    dct_data['df_full_cleaned_data'] = df_raw_data
    for row_id in list(dct_removed.keys()):
        df_raw_data.ix[row_id, dct_removed[row_id]] = np.nan

    holdout_idx = key_val_products(dct_removed)

    dct_data['df_raw_data'] = df_raw_data

    dct_data['dct_removed'] = dct_removed
    dct_data['holdout_idx'] = holdout_idx


    return dct_data

def prepare(dct_data, dct_param):

    df_raw_data = dct_data['df_raw_data']
    if 'Unnamed: 0' in df_raw_data.columns:
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

    df_col_structure = pd.DataFrame({'char_col': ','.join(lst_char_cols),
                                     'num_col': ','.join(lst_num_cols),
                                     'year_col': ','.join(lst_year_cols),
                                     'skip_col': ','.join(list(cols_to_drop))}, index=[0])

    df_col_structure.to_csv(os.path.join(dct_param['data'], 'col_structure.csv'), index=False)

    dct_data = track_holdout(dct_data, dct_param)

    lst_char_cols = [col for col in lst_char_cols if col in df_raw_data]

    df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].astype(str)

    return dct_data