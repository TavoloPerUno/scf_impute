import os
import pandas as pd
import argparse
import sys
sys.path.append('../')
import pickle
from scf_impute import preprocess
from scf_impute import impute
from scf_impute import analysis_variables

parser = argparse.ArgumentParser(description='Impute SCF Data')

# Required positional argument
parser.add_argument('method', type=str,
                    help='method')

dct_param = {'data': os.path.join('..', 'data'),
             'missing_val': 'nan'}

def download_data():

    dct_data = dict()
    dct_data['df_orig_data'] = pd.read_csv(os.path.join(dct_param['data'], 'raw_2013.csv'))
    dct_data['df_knn_imputed'] = pd.read_csv(os.path.join(dct_param['data'], 'knn_imputed.csv'))
    dct_data['df_raw_data'] = pd.read_csv(os.path.join(dct_param['data'], 'raw_2013.csv'))
    dct_data['df_xvariables'] = pd.read_excel(os.path.join(dct_param['data'], 'xvariables-final.xlsx'))

    return dct_data


def main(argv):

    args = parser.parse_args()
    method = args.method

    dct_data = dict()

    if os.path.isfile(os.path.join(dct_param['data'], 'results.pickle')):
        with open(os.path.join(dct_param['data'], 'results.pickle'), 'rb') as handle:
            dct_data = pickle.load(handle)

    temp = dict()
    if os.path.isfile(os.path.join(dct_param['data'], 'xgboost_impute.pickle')):
        with open(os.path.join(dct_param['data'], 'xgboost_impute.pickle'), 'rb') as handle:
            temp = pickle.load(handle)

    dct_data['xgboost_imputed'] = temp['xgboost_imputed']
    # dct_data['xgboost_imputed'] = pd.DataFrame(data=dct_data['xgboost_imputed'], columns=temp['df_raw_data'].columns, index=temp['df_raw_data'].index)


    # dct_data['knn_imputed'].to_csv(os.path.join(dct_param['data'], 'knn_imputed.csv'), index=False)

    # temp['df_raw_data'].to_csv(os.path.join(dct_param['data'], 'raw_cleaned.csv'), index=False)
    # df_structure.to_csv(os.path.join(dct_param['data'], 'col_structure.csv'), index=False)

    with open(os.path.join(dct_param['data'], 'output.pickle'), 'rb') as handle:
        temp  = pickle.load(handle)
    dct_data['knn_imputed'] = temp['knn_imputed']

    dct_data.update(download_data())

    dct_data.update(preprocess.prepare(dct_data, dct_param))
    with open(os.path.join(dct_param['data'], 'xgboost_knn.pickle'), 'wb') as handle:
        pickle.dump(dct_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_knn_data = analysis_variables.fill_analysis_variables(dct_data, dct_param, dct_data['knn_imputed'])
    df_xgboost_data = analysis_variables.fill_analysis_variables(dct_data, dct_param, dct_data['xgboost_imputed'])
    df_knn_data.to_csv(os.path.join(dct_param['data'], 'knn_imputed_analysis.csv'), index=False)
    df_xgboost_data.to_csv(os.path.join(dct_param['data'], 'xgboost_imputed_analysis.csv'), index=False)

    if method == 'xgboost':
        dct_data['xgboost_imputed'] = impute.xgboost_impute(dct_data, dct_param)

    if method == 'knn':
        dct_data['knn_imputed'] = impute.knn_impute(dct_data, dct_param, 7)

    if method == 'rforest':
        dct_data['rforest_imputed'] = impute.rforest_impute(dct_data, dct_param)

    if method == 'glrm':
        dct_data['glrm_imputed'] = impute.glrm_impute(dct_data, dct_param)

    with open(os.path.join(dct_param['data'], 'results.pickle'), 'wb') as handle:
        pickle.dump(dct_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main(sys.argv[1:])