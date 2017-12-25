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

    dct_data = download_data()
    dct_data = preprocess.prepare(dct_data, dct_param)
    if method == 'xgboost':
        dct_data['xgboost_imputed'] = impute.xgboost_impute(dct_data, dct_param)

    with open(os.path.join(dct_param['data'], 'output.pickle'), 'wb') as handle:
        pickle.dump(dct_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main(sys.argv[1:])