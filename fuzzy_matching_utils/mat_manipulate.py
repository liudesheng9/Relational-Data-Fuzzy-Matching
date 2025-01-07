from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import multiprocessing as mp
from .global_const import *

def tf_idf_get_matrix(input_short: list[str], input_long: list[str]) -> int:
    def matrix_csr_save_iterator(matrix, block_size):
        num_rows = matrix.shape[0]
        i = 0
        for row_start in range(0, num_rows, block_size):
            path = matrix_path + r'/tf_idf_match_matrix_{}.npz'.format(i)
            row_end = min(row_start + block_size, num_rows)
            current_block = matrix[row_start:row_end, :]#deriveb the current block to sp mat
            sp.save_npz(path, current_block, True)
            i += 1
        return i

    small_matrix_path = matrix_path + r'/tf_idf_clean_matrix.npz'

    vectorizer = TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(2, 6))
    large_matrix = vectorizer.fit_transform(input_long)
    small_matrix = vectorizer.transform(input_short)
    print('vecorizing complete')
    block_size = 1000000
    sp.save_npz(small_matrix_path, small_matrix, True)
    all_file_num = matrix_csr_save_iterator(large_matrix, block_size)
    print('done')
    return all_file_num

def cossim_calculator(i):
    def sparse_cosine_similarity(matrix1, matrix2):
        def normalize_matrix_rows(matrix):
            squared = matrix.multiply(matrix)
            sqrt_sum_squared_rows = np.array(np.sqrt(squared.sum(axis=1)))[:, 0]
            row_indices, col_indices = matrix.nonzero()
            matrix.data /= sqrt_sum_squared_rows[row_indices]
            return matrix

        # Ensure matrix1 is in CSR format
        out1 = (matrix1.copy() if isinstance(matrix1, csr_matrix) else matrix1.tocsr())
        # Ensure matrix2 is in CSR format
        out2 = (matrix2.copy() if isinstance(matrix2, csr_matrix) else matrix2.tocsr())
        # Calculate the dot product of normalized matrices
        out1_normalized = normalize_matrix_rows(out1)
        out2_normalized = normalize_matrix_rows(out2)
        similarity_matrix = out1_normalized.dot(out2_normalized.T)
        return similarity_matrix

    try:
        print('matrix', i, 'begin')
        cali_matrix_path = matrix_path + r'/tf_idf_clean_matrix.npz'
        match_matrix_path = matrix_path + r'/tf_idf_match_matrix_{}.npz'.format(i)
        cali_csr = sp.load_npz(cali_matrix_path)
        match_csr = sp.load_npz(match_matrix_path)
        print(i, 'load matrix done')
        result_csr = sparse_cosine_similarity(match_csr, cali_csr)
        print(i, 'calculate done')
        path_output = matrix_path + r'/tf_idf_cossim_matrix_0.5_{}.npz'.format(i)
        result_csr.data[result_csr.data <= 0.5] = 0
        result_csr.eliminate_zeros()
        sp.save_npz(path_output, result_csr, True)

        print('matrix', i, 'done')
    except Exception as e:
        print('matrix', i, 'error', str(e))
for i in range(41,85):
    cossim_calculator(i)

def run_cossim_calculator(m):
    with mp.Pool(2) as p:
        p.starmap_async(cossim_calculator, [(i,) for i in range(m)])

def build_col_list_core(matrix, col):
    try:
        if (col % 100 == 0) & (col != 0):
            print(col)
        col_value = matrix[:, col].toarray().flatten()
        nonzero_indices = np.nonzero(col_value)[0]
        nonzero_values = col_value[nonzero_indices]
        result_array = np.column_stack((nonzero_indices, nonzero_values))
        del col_value
        return result_array
    except Exception as e:
        print('col', col, 'error', str(e))


def find_top_n_core(col_values, n, i):
    try:
        print('col', i, 'begin')
        sorted_indices = np.argsort(col_values[:, 1])
        sorted_matrix = col_values[sorted_indices]
        new_matrix = sorted_matrix[-n:]
        top_indices = new_matrix[:, 0].tolist()  # 获取最大的前n个值的索引
        top_values = new_matrix[:, 1].tolist()  # 获取对应的值
        print('col', i, 'done')
        return (top_indices, top_values)
    except Exception as e:
        print('col', i, 'error', str(e))


def get_max_n():
    def find_top_n_indices_sparse(matrix, n):
        result = []
        values_result = []
        with mp.Pool(40) as p_col:
            col_value_list = p_col.starmap_async(build_col_list_core,
                                                 [(matrix, col) for col in range(matrix.shape[1])]).get()
        print('col value done')
        with mp.Pool(20) as p:
            list_result = p.starmap_async(find_top_n_core,
                                          [(col_values, n, i) for i, col_values in enumerate(col_value_list)]).get()
        for element in list_result:
            result.append(element[0])
            values_result.append(element[1])
        return result, values_result

    inpath = r'/mnt/century/Franchise/minnesota_match/matrix_tfidf_2/tf_idf_cossim_matrix_0.5_all.npz'
    matrix = sp.load_npz(inpath)
    n = 40
    result_find = find_top_n_indices_sparse(matrix, n)
    indices = result_find[0]
    similiarity = result_find[1]
    print('build csv begin')
    sample_db_path = r'/mnt/century/Franchise/Sampletry/db_companyname_list_bigger'
    subject_path = r'/mnt/century/Franchise/minnesota_match/minnesota_fnadded_simplified_bigger.csv'
    output_path = mid_path + r'/tf_idf_matrix_cos_try.csv'
    subject_df = pd.read_csv(subject_path, index_col=0)
    clean_list = list(subject_df['company name'])
    db_year_df = pd.read_csv(sample_db_path, index_col=0)
    db_year_list = list(db_year_df['companyname db'])
    matched_list = [str(element) for element in db_year_list]
    cali_name_list = []
    similarity_list = []
    db_name_list = []
    for i in range(len(indices)):
        similiarity_i = similiarity[i]
        index_i = indices[i]
        similarity_list.extend(similiarity_i)
        cali_name_list.extend([clean_list[i]] * len(index_i))
        for m in index_i:
            db_name_list.append(matched_list[int(m)])
    df_dict = {'indi company name': cali_name_list,
               'db company name': db_name_list,
               'similarity': similarity_list}
    df_output = pd.DataFrame(df_dict)
    print('build csv done')
    df_output.to_csv(output_path)
get_max_n()

