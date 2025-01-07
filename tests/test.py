import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fuzzy_matching_utils

#create dir
fuzzy_matching_utils.create_global_path_list()