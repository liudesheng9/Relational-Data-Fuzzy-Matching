import pandas as pd
import os
from .tfidf_vectorizer import *
from .global_const import *

def data_loader(file_path):
    '''
    a safer method for loading data
    '''
    return pd.read_csv(file_path)

def data_saver(data, file_path):
    '''
    a safer method for saving data
    '''
    data.to_csv(file_path, index=False)

def create_global_path_list():
    '''
    create a global path list
    '''
    for path in global_path_list:
        os.makedirs(path, exist_ok=True)
