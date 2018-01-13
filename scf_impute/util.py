import itertools
import re

def reverse_map(old_dct):
    new_dct = {}
    for key, value in old_dct.items():
        for string in value:
            new_dct.setdefault(string, []).append(key)
    return new_dct

def key_val_products(dicts):
    return ([item for sublist in [list(itertools.product([k], dicts[k])) for k in list(dicts.keys())] for item in sublist])

def pythonize_colnames(df):
    df.columns = list(map(lambda each:re.sub('[^0-9a-zA-Z]+', '_', each).lower(), df.columns))