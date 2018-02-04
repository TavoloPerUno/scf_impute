import os
import pandas as pd
import argparse
import sys
sys.path.append('../')
import pickle
from scf_impute import preprocess
from scf_impute import impute
from scf_impute import analysis_variables
import numpy as np

parser = argparse.ArgumentParser(description='Impute SCF Data')

# Required positional argument
parser.add_argument('method', type=str,
                    help='method')
parser.add_argument('--iter', type=int,
                    help='n_run', default=1, dest='nrun')

parser.add_argument('--redowithholding', type=int,
                    help='Should withholding be done again?', default=0, dest='withhold')

dct_param = {'data': os.path.join('..', 'data'),
             'missing_val': 'nan'}


def prepare_for_upload(dct_data, df_raw_data):
    lst_char_cols = [col for col in dct_data['lst_char_cols'] if col if col in df_raw_data.columns and not df_raw_data[col].isnull().any()]
    lst_year_cols = [col for col in dct_data['lst_year_cols'] if col if col in df_raw_data.columns and not df_raw_data[col].isnull().any()]

    try:
        df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].fillna(-9223372036854775808)
        df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].astype(int)
        df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].astype(str)
        df_raw_data[lst_char_cols] = df_raw_data[lst_char_cols].replace({'-9223372036854775808': np.nan})


        df_raw_data[lst_year_cols] = df_raw_data[lst_year_cols].astype(int)

    except:
        print("Categorical columns still have NAs")

    df_raw_data[dct_data['empty_cols']] = dct_data['df_full_cleaned_data'][dct_data['empty_cols']]

    return df_raw_data

def download_data():

    dct_data = dict()
    dct_data['df_orig_data'] = pd.read_csv(os.path.join(dct_param['data'], 'raw_2013.csv'))
    # dct_data['df_knn_imputed'] = pd.read_csv(os.path.join(dct_param['data'], 'knn_imputed.csv'))
    dct_data['df_raw_data'] = pd.read_csv(os.path.join(dct_param['data'], 'raw_2013.csv'))
    dct_data['df_xvariables'] = pd.read_excel(os.path.join(dct_param['data'], 'xvariables-final.xlsx'))
    dct_data['df_missing_probabilities'] = pd.read_csv(os.path.join(dct_param['data'], 'missing_probabilities.csv'), index_col=0)
    dct_data['df_missing_indicators'] = pd.read_csv(os.path.join(dct_param['data'], 'missing_indicator.csv'),
                                                       index_col=0)

    return dct_data


def main(argv):

    args = parser.parse_args()
    method = args.method
    nrun = args.nrun
    withhold = args.withhold

    dct_data = dict()
    dct_param['nrun'] = nrun
    dct_param['withhold'] = withhold
    dct_param['method'] = method
    if os.path.isfile(os.path.join(dct_param['data'], 'variables.pickle')):
        with open(os.path.join(dct_param['data'], 'variables.pickle'), 'rb') as handle:
            dct_data = pickle.load(handle)

    # dct_data['df_removed'].to_csv(os.path.join(dct_param['data'], 'withheld.csv'), index=False)
    # dct_data['df_full_cleaned_data'].to_csv(os.path.join(dct_param['data'], 'full_cleaned.csv'), index=True)

    # with open(os.path.join(dct_param['data'], method + '_imputed_' + str(nrun) + '.pickle'), 'wb') as handle:
    #     pickle.dump(dct_data, handle, protocol=2)

    # temp = dict()
    # if os.path.isfile(os.path.join(dct_param['data'], 'xgboost_impute.pickle')):
    #     with open(os.path.join(dct_param['data'], 'xgboost_impute.pickle'), 'rb') as handle:
    #         temp = pickle.load(handle)
    #
    # dct_data['xgboost_imputed'] = temp['xgboost_imputed']
    # # dct_data['xgboost_imputed'] = pd.DataFrame(data=dct_data['xgboost_imputed'], columns=temp['df_raw_data'].columns, index=temp['df_raw_data'].index)
    #
    #
    # # dct_data['knn_imputed'].to_csv(os.path.join(dct_param['data'], 'knn_imputed.csv'), index=False)
    #
    # # temp['df_raw_data'].to_csv(os.path.join(dct_param['data'], 'raw_cleaned.csv'), index=False)
    # # df_structure.to_csv(os.path.join(dct_param['data'], 'col_structure.csv'), index=False)
    #
    # with open(os.path.join(dct_param['data'], 'output.pickle'), 'rb') as handle:
    #     temp  = pickle.load(handle)
    # dct_data['knn_imputed'] = temp['knn_imputed']

    if True:#len(list(dct_data.keys())) < 1:
        dct_data.update(download_data())

        dct_data.update(preprocess.prepare(dct_data, dct_param))

        if nrun != 100:
            for i in range(1, 6):
                dct_data['df_removed_' + str(i)].to_csv(os.path.join(dct_param['data'], 'withheld_' + str(i) + '.csv'), index=False)

        # dct_data['df_removed'].to_csv(os.path.join(dct_param['data'], 'withheld.csv'), index=False)
        dct_data['df_full_cleaned_data'].to_csv(os.path.join(dct_param['data'], 'full_cleaned.csv'), index=True)
        dct_data['df_raw_data'].to_csv(os.path.join(dct_param['data'], 'withheld_cleaned.csv'), index=True)
        dct_data['df_col_structure'].to_csv(os.path.join(dct_param['data'], 'col_structure.csv'), index=False)

        with open(os.path.join(dct_param['data'], 'variables.pickle'), 'wb') as handle:
            pickle.dump(dct_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    df_imputed = ''
    if method == 'xgboost':
        df_imputed = impute.xgboost_impute(dct_data, dct_param)


    if method == 'knn':
        df_imputed = impute.knn_impute(dct_data, dct_param, 7)

    if method == 'rforest':
        df_imputed = impute.rforest_impute(dct_data, dct_param)

    if method == 'glrm':
        df_imputed = impute.glrm_impute(dct_data, dct_param)


    df_imputed = prepare_for_upload(dct_data, df_imputed)
    df_imputed.set_index(dct_data['df_full_cleaned_data'].index, inplace=True)
    dct_data[method + '_imputed_raw' + str(nrun)] = df_imputed
    with open(os.path.join(dct_param['data'], 'variables.pickle'), 'wb') as handle:
        pickle.dump(dct_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_imputed_with_analysis_variables = analysis_variables.fill_analysis_variables(dct_data, dct_param, df_imputed)



    dct_data[method + '_imputed_' + str(nrun)] = df_imputed_with_analysis_variables


    df_imputed_with_analysis_variables.to_csv(os.path.join(dct_param['data'], method + '_imputed_' + str(nrun) + '.csv'), index=True)

    # dct_data_new = dict()
    #
    # if os.path.isfile(os.path.join(dct_param['data'], 'variables.pickle')):
    #     with open(os.path.join(dct_param['data'], 'variables.pickle'), 'rb') as handle:
    #         dct_data_new = pickle.load(handle)
    #
    # # for key in dct_data_new:
    # #     if key in dct_data:
    # #         dct_data_new.pop(key, None)
    #
    # dct_data.update(dct_data_new)

    with open(os.path.join(dct_param['data'], 'variables.pickle'), 'wb') as handle:
        pickle.dump(dct_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main(sys.argv[1:])