import pandas as pd
import numpy as np
from scf_impute import master
from scf_impute import impute
from sklearn.metrics import mean_squared_error, accuracy_score

def prepare_for_scores(dct_data, dct_param, df_imputed, nrun, with_impute=True):

    if with_impute:
        df_raw_data = dct_data['df_raw_data'].copy()
        dct_data['df_raw_data'] = df_imputed
        dct_param['nrun'] = nrun
        df_imputed = impute.xgboost_impute(dct_data, dct_param)
        dct_data['df_raw_data'] = df_raw_data

    df_imputed = master.prepare_for_upload(dct_data, df_imputed)


    df_imputed.set_index(dct_data['df_full_cleaned_data'].index, inplace=True)

    return df_imputed


def get_mse(df_full_data, df_imputed, lst_num_cols, df_removed):
    df_full_scaled, df_col_mu_std = scale(df_full_data, lst_num_cols)
    df_imputed_scaled = scale_imputed(df_imputed, df_col_mu_std, lst_num_cols)

    y = pd.DataFrame(columns=('imputed', 'actual', 'row', 'col'))

    for col in set(lst_num_cols).intersection(set(df_removed.columns)):
        y = y.append(pd.DataFrame({'imputed': df_imputed_scaled.loc[
            np.fromstring(str(df_removed[col].values[0]), dtype=int, sep=','), col].values,
                                   'actual': df_full_scaled.loc[
                                       np.fromstring(str(df_removed[col].values[0]), dtype=int, sep=','), col].values,
                                   'row': np.fromstring(str(df_removed[col].values[0]), dtype=int, sep=','),
                                   'col': [col] * np.fromstring(str(df_removed[col].values[0]), dtype=int,
                                                                sep=',').size}))

    return y, mean_squared_error(y['actual'], y['imputed'])


def get_accuracy(df_full_data, df_imputed, lst_char_cols, df_removed):
    y = pd.DataFrame(columns=['actual', 'imputed', 'col'])
    for col in set(lst_char_cols).intersection(set(df_removed.columns)):
        actual = df_full_data.loc[np.fromstring(str(df_removed[col].values[0]), dtype=int, sep=','), col].values
        imputed = df_imputed.loc[np.fromstring(str(df_removed[col].values[0]), dtype=int, sep=','), col].values

        y = y.append(pd.DataFrame({'actual': actual,
                                   'imputed': imputed,
                                   'col': [col] * actual.shape[0]}))

    return y, accuracy_score(y['actual'], y['imputed'])


def scale(df_raw_data, lst_num_cols):
    df_col_mu_std = pd.DataFrame(columns=['mean', 'std'])
    for col in lst_num_cols:
        if col in df_raw_data.columns:
            mu = df_raw_data[col].mean(skipna=True)
            std = df_raw_data[col].std(skipna=True)
            df_col_mu_std = df_col_mu_std.append(pd.DataFrame({'mean': mu,
                                                               'std': std},
                                                              index=[col]))
            df_raw_data[col] = (df_raw_data[col] - mu) / std

    return df_raw_data, df_col_mu_std


def scale_imputed(df_raw_data, df_col_mu_std, lst_num_cols):
    for col in lst_num_cols:
        if col in df_col_mu_std.index:
            mu = df_col_mu_std.loc[col, 'mean']
            std = df_col_mu_std.loc[col, 'std']
            df_raw_data[col] = (df_raw_data[col] - mu) / std

    return df_raw_data