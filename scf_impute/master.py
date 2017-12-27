import os
import pandas as pd
import argparse
import sys
sys.path.append('../')
import pickle
from scf_impute import preprocess
from scf_impute import impute

parser = argparse.ArgumentParser(description='Impute SCF Data')

# Required positional argument
parser.add_argument('method', type=str,
                    help='method')

dct_param = {'data': os.path.join('..', 'data'),
             'missing_val': 'nan'}

def download_data():

    dct_data = dict()
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

    # temp = dict()
    # if os.path.isfile(os.path.join(dct_param['data'], 'output.pickle')):
    #     with open(os.path.join(dct_param['data'], 'output.pickle'), 'rb') as handle:
    #         temp = pickle.load(handle)
    #
    #
    #
    # df_structure = pd.DataFrame({'char_col': ",".join(temp['lst_char_cols']),
    #                              'num_col': ",".join(temp['lst_num_cols']),
    #                              'year_col': ",".join(temp['lst_year_cols']),
    #                              'skip_col': ",".join(list(temp['lst_skipped_cols']))}, index=[0])
    #
    # temp['knn_imputed'].to_csv(os.path.join(dct_param['data'], 'knn_imputed.csv'), index=False)
    #
    # temp['df_raw_data'].to_csv(os.path.join(dct_param['data'], 'raw_cleaned.csv'), index=False)
    # df_structure.to_csv(os.path.join(dct_param['data'], 'col_structure.csv'), index=False)

    dct_data.update(download_data())

    dct_data.update(preprocess.prepare(dct_data, dct_param))

    if method == 'xgboost':
        dct_data['xgboost_imputed'] = impute.xgboost_impute(dct_data, dct_param)

    if method == 'knn':
        dct_data['knn_imputed'] = impute.knn_impute(dct_data, dct_param, 7)

    if method == 'rforest':
        dct_data['rforest_imputed'] = impute.rforest_impute(dct_data, dct_param)

    with open(os.path.join(dct_param['data'], 'results.pickle'), 'wb') as handle:
        pickle.dump(dct_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main(sys.argv[1:])