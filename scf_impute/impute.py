from dstk.imputation import DefaultImputer
import os
import pandas as pd
import numpy as np
import re
from glrm.glrm import GLRM
from glrm.loss import QuadraticLoss, HingeLoss
from glrm.reg import QuadraticReg
from glrm.convergence import Convergence
from sklearn.preprocessing import MaxAbsScaler
from predictive_imputer import predictive_imputer
# from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

def rforest_impute(dct_data, dct_param):
    df_raw_data = dct_data['df_raw_data']
    imputer = predictive_imputer.PredictiveImputer(f_model='RandomForest')
    filled_in = imputer.fit(df_raw_data).transform(df_raw_data.copy())
    df_filled_in = pd.DataFrame(data=filled_in, columns=df_raw_data.columns, index=df_raw_data.index)
    return df_filled_in

def xgboost_impute(dct_data, dct_param):
    df_raw_data = dct_data['df_raw_data']
    for char_col in dct_data['lst_char_cols']:
        if char_col in df_raw_data.columns:
            df_raw_data[char_col] = df_raw_data[char_col].fillna('nan')

    imputer = DefaultImputer(missing_string_marker='nan')  # treat 'UNKNOWN' as missing value
    filled_in = imputer.fit(df_raw_data).transform(df_raw_data)
    return filled_in

def glrm_impute(dct_data, dct_param):

    df_raw_data = dct_data['df_raw_data']
    k = df_raw_data.shape[0]
    lst_char_cols = dct_data['lst_char_cols']
    lst_year_cols = dct_data['lst_year_cols']
    np_char = df_raw_data[[col for col in lst_char_cols if col in df_raw_data.columns]].values
    np_num = df_raw_data[[col for col in df_raw_data.columns if col not in lst_char_cols]].values
    lst_missing_num = np.argwhere(np.isnan(np_num)).tolist()
    lst_missing_char = np.argwhere(np_char == 'nan').tolist()



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

# def knn_impute(dct_data, dct_param, k):
#     df_raw_data = dct_data['df_raw_data']
#     df_raw_data.replace('nan', np.nan)
#     filled_in = KNN(k=k).complete(df_raw_data)
#
#     df_filled_in = pd.DataFrame(data=filled_in, columns=df_raw_data.columns, index=df_raw_data.index)
#
#     return df_filled_in


