import pandas as pd
import os
from .mat_manipulate import *
from .global_const import *

def data_loader(file_path: str) -> pd.DataFrame:
    '''
    a safer method for loading data
    '''
    return pd.read_csv(file_path)

def data_saver(data: pd.DataFrame, file_path: str):
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

def clean_string_list(string_list: list[str]) -> list[str]:
    '''
    clean the string list
    '''

    return [str(element).strip().lower() for element in list(set(string_list))]

