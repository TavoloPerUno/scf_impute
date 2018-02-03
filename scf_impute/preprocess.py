import os
import sklearn.model_selection
import pandas as pd
import numpy as np
import random
from scf_impute import  util
import math
import pickle

def track_holdout(dct_data, dct_param):
    df_raw_data = dct_data['df_raw_data']

    dct_data['df_full_cleaned_data'] = df_raw_data.copy()

    if dct_param['nrun'] != 100:

        if dct_param['withhold']:

            df_missing_indicators = dct_data['df_missing_indicators']
            df_missing_probabilities = dct_data['df_missing_probabilities']

            lst_cols_to_withhold = dct_data['lst_char_cols'] + dct_data['lst_num_cols']
            lst_cols_to_withhold = [col for col in lst_cols_to_withhold
                                    if col not in dct_data['lst_skipped_cols']
                                    and col in df_raw_data.columns
                                    and (df_missing_indicators[col]==1).any()]

            dct_data['lst_cols_to_withhold'] = []
            dct_data['dct_withhold_sets'] = dict()

            for col in lst_cols_to_withhold:
                lst_persisting_ind = df_missing_indicators.loc[df_missing_indicators[col]==0].index.tolist()

                cutoff_probability = df_missing_probabilities.loc[df_missing_indicators[col] == 1][col].quantile(0.5)

                lst_withhold_candidates = df_missing_probabilities.loc[
                    (df_missing_indicators[col] == 0) & (df_missing_probabilities[col] >= cutoff_probability)].index.tolist()
                if len(lst_persisting_ind) > len(lst_withhold_candidates):

                    lst_withhold_sets = util.chunkify(lst_withhold_candidates, 5)

                    dct_col_sets = {col: []}

                    for lst_set in lst_withhold_sets:

                        if df_raw_data.loc[~df_raw_data.index.isin(lst_set), col].nunique(dropna=True) > 1:
                            dct_col_sets[col].append(lst_set)

                    if len(dct_col_sets[col]) > 0:
                        dct_data['lst_cols_to_withhold'].append(col)
                        dct_data['dct_withhold_sets'].update(dct_col_sets)
                        print("Withholding sets for %s are" % (col))
                        print(dct_col_sets[col])

            for i in range(1,6):
                df_removed = pd.DataFrame(dict((col,",".join(str(idx) for idx in dct_data['dct_withhold_sets'][col][i - 1])
                                                    if col in dct_data['dct_withhold_sets']
                                                       and len(dct_data['dct_withhold_sets'][col]) >= i
                                                    else '') for col in df_raw_data.columns), index=[0])

                dct_data['df_removed_' + str(i)] = df_removed

        else:

            for col in dct_data['lst_cols_to_withhold']:
                if len(dct_data['dct_withhold_sets'][col]) >= dct_param['nrun']:
                    df_raw_data.loc[dct_data['dct_withhold_sets'][col][dct_param['nrun'] - 1], col] = np.nan


        #     # 1. Read a row
        #     # 2. Pick 10 or less numeric values that exist
        #     # 3. Record the row number and associated 10 or less columns
        #     # 4. Make the 10 or less numeric values missing
        #     # 5. Next row
        #
        #     dct_removed = {}
        #     tot_rows = len(test_id)
        #
        #
        #     dct_removed_reverse = {}
        #
        #     for i in test_id:
        #         # Select a row
        #         row = df_raw_data.loc[i,]
        #
        #         # Find columns with data
        #         existing_nums = list(row[dct_data['lst_num_cols']].index[
        #                                     row[dct_data['lst_num_cols']].notnull()])
        #
        #         existing_chars = list(row[dct_data['lst_char_cols']].index[
        #                                     row[dct_data['lst_char_cols']].notnull()])
        #
        #
        #
        #         dct_unique_others = dict(df_raw_data.loc[df_raw_data.index != i, existing_chars].nunique(dropna=False))
        #         dct_unique = dict(df_raw_data.loc[:, existing_chars].nunique(dropna=False))
        #
        #         existing_chars = [val[0] for val in dct_unique_others.items() & dct_unique.items()]
        #
        #         existing_columns = existing_nums + existing_chars
        #
        #
        #         # Miniumum of length of existing columns or 10
        #         num_valuesdropped = max(0, min(len(existing_columns) - 5, 10))
        #         random.seed(10 + i)
        #         random.shuffle(existing_columns)
        #         kept = pd.Index(existing_columns)
        #         # Create list of indices and then randomly select values that will be dropped for analysis
        #         kept_indices = list(range(len(kept)))
        #         random.seed(10 + i)
        #         columns_dropped = kept[random.sample(kept_indices, num_valuesdropped)]
        #
        #         dct_removed[i] = columns_dropped
        #
        #         df_raw_data.loc[i, dct_removed[i]] = np.nan
        #
        #         for col in dct_removed[i]:
        #             dct_removed_reverse[col] = ','.join(filter(None, (dct_removed_reverse.setdefault(col, ''), str(i))))
        #
        #         print("Finished withholding %s of %s rows" % (str(len(list(dct_removed.keys()))), str(tot_rows)))
        #
        #     holdout_idx = util.key_val_products(dct_removed)
        #
        #     # dct_removed = util.reverse_map(dct_removed)
        #
        #     # for k in dct_removed.keys():
        #     #     dct_removed[k] = ",".join(map(str, dct_removed[k]))
        #
        #     dct_data['df_removed'] = pd.DataFrame(dct_removed_reverse, index=[0])
        #
        #     dct_data['holdout_idx'] = holdout_idx
        #
        # else:
        #     for col in dct_data['df_removed'].columns:
        #         rows = [int(x) for x in dct_data['df_removed'].loc[0, col].split(",")]
        #         df_raw_data.loc[rows, col] = np.nan



    nunique = df_raw_data.apply(pd.Series.nunique)
    empty_cols = list(nunique[nunique == 1].index)

    empty_cols.extend(df_raw_data.columns[df_raw_data.isnull().all()].tolist())
    df_raw_data = df_raw_data.dropna(axis=1, how='all')

    if len(empty_cols) > 0:
        df_raw_data.drop(empty_cols, axis=1, inplace=True)
    dct_data['empty_cols'] = empty_cols

    # for k in dct_removed:
    #     dct_removed[k] = [col for col in dct_removed[k] if col not in empty_cols]

    # df_raw_data[empty_cols] = dct_data['df_full_cleaned_data'][empty_cols]

    dct_data['df_raw_data'] = df_raw_data

    return dct_data

def prepare(dct_data, dct_param):

    df_raw_data = dct_data['df_raw_data']

    if dct_param['nrun'] == 100:
        if 'Unnamed: 0' in df_raw_data.columns:
            del df_raw_data['Unnamed: 0']
        df_raw_data = df_raw_data.loc[df_raw_data['yy1'] != 0,]
        df_raw_data.set_index('yy1', inplace=True)

    else:
        df_raw_data = dct_data[dct_param['method'] + '_imputed_100'].copy()

    df_xvariables = dct_data['df_xvariables']

    util.pythonize_colnames(df_xvariables)

    df_xvariables['na_code'] = df_xvariables['na_code'].fillna(0)

    df_raw_data = df_raw_data[list(filter(lambda x: x.startswith('x') and
                                                    x not in list(df_xvariables[df_xvariables['nominal_character_c_or_numeric_n'].isnull()].x),
                                          df_raw_data.columns))]

    lst_char_cols = [col for col in list(df_xvariables[df_xvariables['nominal_character_c_or_numeric_n'] == 'C'].x)
                     if col in df_raw_data.columns]
    lst_year_cols = [col for col in list(df_xvariables[df_xvariables['is_year'] == 1].x)
                     if col in df_raw_data.columns]
    lst_num_cols = [col for col in list(df_xvariables[df_xvariables.nominal_character_c_or_numeric_n.isin(['N', 'P', 'I'])].x)
                    if col in df_raw_data.columns and col not in lst_year_cols]


    nunique = df_raw_data.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique < 2].index

    skip_cols = list(cols_to_drop)
    skip_cols.extend(df_raw_data.columns[df_raw_data.isnull().all()].tolist())
    df_raw_data.drop(cols_to_drop, axis=1, inplace=True)
    df_raw_data = df_raw_data.dropna(axis=1, how='all')
    dct_data['df_raw_data'] = df_raw_data
    dct_data['df_xvariables'] = df_xvariables

    if dct_param['nrun'] == 100:
        dct_data['lst_char_cols'] = lst_char_cols
        dct_data['lst_num_cols'] = lst_num_cols
        dct_data['lst_year_cols'] = lst_year_cols
        dct_data['lst_skipped_cols'] = skip_cols


        dct_data['df_col_structure'] = pd.DataFrame({'char_col': ','.join(lst_char_cols),
                                         'num_col': ','.join(lst_num_cols),
                                         'year_col': ','.join(lst_year_cols),
                                         'skip_col': ','.join(list(cols_to_drop))}, index=[0])

    lst_char_cols = [col for col in lst_char_cols if col in df_raw_data]

    df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].fillna(-9223372036854775808)
    df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].astype(int)
    df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].astype(str)
    df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].replace({'-9223372036854775808': np.nan})


    lst_year_cols = [col for col in lst_year_cols if col in df_raw_data.columns]

    for yr_col in lst_year_cols:
        na_code = df_xvariables.loc[df_xvariables['x'] == yr_col, 'na_code'].values[0]
        df_raw_data[yr_col] = df_raw_data[yr_col].fillna(na_code)

    df_raw_data[lst_year_cols] = df_raw_data[lst_year_cols].astype(int)

    dct_data = track_holdout(dct_data, dct_param)


    return dct_data