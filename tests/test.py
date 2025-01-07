import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fuzzy_matching_utils

#create dir
fuzzy_matching_utils.create_global_path_list()

#read data
df_short = fuzzy_matching_utils.data_loader('./short.csv')
df_long = fuzzy_matching_utils.data_loader('./long.csv')

#clean data
short_list = fuzzy_matching_utils.clean_string_list(df_short['comany_name'].tolist())
long_list = fuzzy_matching_utils.clean_string_list(df_long['comany_name'].tolist())

#calculate tf-idf matrix
all_long_mat_num = fuzzy_matching_utils.tf_idf_get_matrix(short_list, long_list)

#calculate cosine similarity
fuzzy_matching_utils.run_cossim_calculator(all_long_mat_num)
