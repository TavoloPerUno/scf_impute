from dstk.imputation import DefaultImputer
import os
import pandas as pd
import numpy as np
import re
import random
from glrm.glrm import GLRM
from glrm.loss import QuadraticLoss, HingeLoss
from glrm.reg import QuadraticReg
from glrm.convergence import Convergence
from sklearn.preprocessing import MaxAbsScaler
#from predictive_imputer import predictive_imputer
from scf_impute.knn_imputer import Knn_Imputer
from scipy import stats

#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

# def rforest_impute(dct_data, dct_param):
#     df_raw_data = dct_data['df_raw_data']
#     imputer = predictive_imputer.PredictiveImputer(f_model='RandomForest')
#     filled_in = imputer.fit(df_raw_data).transform(df_raw_data.copy())
#     df_filled_in = pd.DataFrame(data=filled_in, columns=df_raw_data.columns, index=df_raw_data.index)
#     return df_filled_in

def xgboost_impute(dct_data, dct_param):
    df_raw_data = dct_data['df_raw_data']
    df_raw_data, df_col_mu_std = scale(df_raw_data, dct_data['lst_num_cols'])


    lst_char_cols = [col for col in dct_data['lst_char_cols'] if col in df_raw_data.columns and col not in dct_data['lst_skipped_cols']]
    lst_num_cols = [col for col in dct_data['lst_num_cols'] if col in df_raw_data.columns and col not in dct_data['lst_skipped_cols']]

    df_raw_data[lst_num_cols] = df_raw_data[lst_num_cols].astype(float)

    lst_cols_to_impute = lst_char_cols + lst_num_cols

    for col in lst_cols_to_impute:
        if not df_raw_data[col].isnull().any():
            lst_cols_to_impute.remove(col)

    random.seed(dct_param['nrun'] * 100)

    random.shuffle(lst_cols_to_impute)

    unique_count = df_raw_data[lst_cols_to_impute].nunique().values
    idx_split = [val[0] for val in np.argwhere(unique_count <= 2).tolist()]

    for char_col in lst_char_cols:
        if char_col in df_raw_data.columns:
            df_raw_data[char_col] = df_raw_data[char_col].fillna('nan')





    lst_parts = [lst_cols_to_impute[i:j] for i, j in zip([0] + idx_split, idx_split + [None])]

    for cols in lst_parts:

        imputer = DefaultImputer(missing_string_marker='nan', random_state=dct_param['nrun'] * 100, missing_features=cols)  # treat 'UNKNOWN' as missing value
        df_raw_data = imputer.fit(df_raw_data).transform(df_raw_data)
        print("(%s of %s)" % (str(lst_cols_to_impute.index(cols[- 1])), str(len(lst_cols_to_impute))))

    df_raw_data = descale(df_raw_data, dct_data['df_col_mu_std'], dct_data['lst_num_cols'])
    return df_raw_data

def glrm_impute(dct_data, dct_param):

    df_raw_data = dct_data['df_raw_data']
    # df_raw_data, df_col_mu_std = scale(df_raw_data, dct_data['lst_num_cols'])
    k = df_raw_data.shape[0]
    lst_char_cols = dct_data['lst_char_cols']
    lst_year_cols = dct_data['lst_year_cols']
    np_char = df_raw_data[[col for col in lst_char_cols if col in df_raw_data.columns]].values
    np_num = df_raw_data[[col for col in df_raw_data.columns if col in  dct_data['lst_year_cols'] +  dct_data['lst_num_cols']]].values
    lst_missing_num = np.argwhere(np.isnan(np_num)).tolist()
    lst_missing_char = np.argwhere(pd.isnull(np_char)).tolist()



    dat_list = [np_num, np_char]
    loss_list = [QuadraticLoss, HingeLoss]
    regX, regY = QuadraticReg(0.1), QuadraticReg(0.1)
    missing_list = [lst_missing_num, lst_missing_char]

    c = Convergence(TOL=1e-3, max_iters=1000)
    model = GLRM(dat_list, loss_list, regX, regY, k, missing_list, converge=c)
    model.fit()
    X, Y = model.factors()
    A_hat = model.predict()  # a horizontally concatenated matrix, not a list
    x = 0

def knn_impute(dct_data, dct_param, k):
    df_raw_data = dct_data['df_raw_data']
    df_raw_data.replace('nan', np.nan)
    df_raw_data, df_col_mu_std = scale(df_raw_data, dct_data['lst_num_cols'])
    impute = Knn_Imputer()

    lst_char_cols = [col for col in dct_data['lst_char_cols'] if col in df_raw_data.columns and col not in dct_data['lst_skipped_cols']]
    lst_num_cols = [col for col in dct_data['lst_num_cols'] if col in df_raw_data.columns and col not in dct_data['lst_skipped_cols']]

    dct_col_mean_mode = get_col_mean_mode(df_raw_data, lst_num_cols, lst_char_cols)

    lst_cols_to_impute = lst_char_cols + lst_num_cols

    for col in lst_cols_to_impute:
        if not df_raw_data[col].isnull().any():
            lst_cols_to_impute.remove(col)

    random.seed(dct_param['nrun'] * 100)

    random.shuffle(lst_cols_to_impute)
    np_mean_mode = np.asarray([dct_col_mean_mode[col] if col in dct_col_mean_mode else 0 for col in df_raw_data.columns])

    for col in lst_cols_to_impute:
        is_categorical = True
        if col in lst_num_cols:
            is_categorical = False
        print("Fitting column %s, (%s of %s)" % (col, lst_cols_to_impute.index(col) + 1, len(lst_cols_to_impute)))
        np_imputed = impute.knn(X=df_raw_data, column=col, k=k, is_categorical=is_categorical, np_mean_mode=np_mean_mode)
        df_raw_data[col] = pd.DataFrame(data=np_imputed, columns=df_raw_data.columns, index=df_raw_data.index)[col]



    # for col in dct_data['lst_num_cols']:
    #     print("Fitting column %s, (%s of %s)" % (col, len(lst_char_cols) + lst_num_cols.index(col) + 1, len(lst_char_cols + lst_num_cols)))
    #     np_imputed = impute.knn(X=df_raw_data, column=col, k=k, np_mean_mode=np_mean_mode)
    #     df_raw_data[col] = pd.DataFrame(data=np_imputed, columns=df_raw_data.columns, index=df_raw_data.index)[col]

    df_raw_data = descale(df_raw_data, df_col_mu_std, dct_data['lst_num_cols'])
    # filled_in = KNN(k=k).complete(df_raw_data)
    #
    # df_filled_in = pd.DataFrame(data=filled_in, columns=df_raw_data.columns, index=df_raw_data.index)

    return df_raw_data

def descale(df_raw_data, df_col_mu_std, lst_num_cols):
    for col in lst_num_cols:
        if col in df_raw_data.columns:
            mu = df_col_mu_std.loc[col, 'mean']
            std = df_col_mu_std.loc[col, 'std']
            df_raw_data[col] = df_raw_data[col]*std + mu

    return df_raw_data

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

def get_col_mean_mode(df_raw_data, lst_num_cols, lst_char_cols):
    dct_mean_mode = dict(zip(lst_num_cols, list(np.nanmean(df_raw_data[lst_num_cols].as_matrix(), 0))))
    dct_mean_mode.update(dict(zip(lst_char_cols, list(df_raw_data[lst_char_cols].mode(0).iloc[0]))))

    return dct_mean_mode

