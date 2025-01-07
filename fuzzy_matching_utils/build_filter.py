import pandas as pd
import copy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity
import re
from .global_const import *
def fuzzy_score_calculator():
    def complete_partial(company_name,db_name):
        db_name_list=re.split(r'[,\s]+', db_name)
        company_name_list = re.split(r'[,\s]+', company_name)
        df_name_set=set([t for t in db_name_list if t!=''])
        company_name_set= set([t for t in company_name_list if t != ''])
        if (df_name_set.issubset(company_name_set))|(company_name_set.issubset(df_name_set)):
            return 1
        else:
            common_elements_count=0
            for element in company_name_list:
                if element in db_name_list:
                    common_elements_count += 1
            return common_elements_count/len(company_name_list)
    suffix_list=['LTD','LLC','INC','CO','CORP']
    def clean_suffix(s):
        s_list=s.split(' ')
        s_list=[t for t in s_list if t not in suffix_list]
        s=' '.join(s_list)
        return s
    def flit_result_match():
        csv_path = mid_path + r'/tf_idf_matrix_cos_try_scored.csv'
        output_path = final_path + r'/tf_idf_matrix_cos_try_flitered.csv'
        fuzzy_match_df = pd.read_csv(csv_path, index_col=0)
        df_output = pd.DataFrame()
        for group, df_grouped in fuzzy_match_df.groupby(short_col_name):
            df_sorted_sim = df_grouped.sort_values(by='similarity', ascending=False)
            top_5_or_high_similarity = df_sorted_sim.head(5)
            df_sorted_fuz1 = df_grouped.sort_values(by='ratio score', ascending=False)
            top_5_or_high_fuzzyscore1 = df_sorted_fuz1.head(5)
            df_sorted_fuz2 = df_grouped.sort_values(by='partial ratio score', ascending=False)
            top_5_or_high_fuzzyscore2 = df_sorted_fuz2.head(5)
            df_sorted_fuz3 = df_grouped.sort_values(by='ratio score nonsuf', ascending=False)
            top_5_or_high_fuzzyscore3 = df_sorted_fuz3.head(5)
            df_sorted_fuz4 = df_grouped.sort_values(by='partial ratio score nonsuf', ascending=False)
            top_5_or_high_fuzzyscore4 = df_sorted_fuz4.head(5)
            high_similarity = df_sorted_sim[df_sorted_sim['similarity'] > 0.95]  # 超过0.95的
            result = pd.concat([top_5_or_high_similarity, high_similarity, top_5_or_high_fuzzyscore1,top_5_or_high_fuzzyscore2,top_5_or_high_fuzzyscore3,top_5_or_high_fuzzyscore4]).drop_duplicates(
                subset=long_col_name)
            result = result.sort_values(by='similarity', ascending=False)

            result['dedi1']=result['true partial']*result['partial ratio score nonsuf']
            result1=result[result['dedi1']==100]
            result1=result1.drop('dedi1', axis=1)

            result2=result[result['dedi1']!=100]
            result2_sorted_sim=result2.sort_values(by='similarity', ascending=False)
            result2_head3_sim=result2_sorted_sim.head(3)
            result2_sorted_fuz1=result2.sort_values(by='true partial', ascending=False)
            result2_head3_fuz1=result2_sorted_fuz1.head(1)
            result2_sorted_fuz2 = result2.sort_values(by='partial ratio score nonsuf', ascending=False)
            result2_head3_fuz2 = result2_sorted_fuz2.head(3)
            result2=pd.concat([result2_head3_sim, result2_head3_fuz1,result2_head3_fuz2]).drop_duplicates(
                subset=long_col_name)
            result2 = result2.drop('dedi1', axis=1)

            result=pd.concat([result2, result1]).drop_duplicates(
                subset=long_col_name)
            result = result.sort_values(by='similarity', ascending=False)
            df_output = pd.concat([df_output, result])

        df_output.to_csv(output_path)

    csv_path = mid_path + r'/tf_idf_matrix_cos_match_table.csv'
    out_path = mid_path + r'/tf_idf_matrix_cos_try_scored.csv'

    short_col_name = src_data_name
    long_col_name = match_data_name

    csv_df = pd.read_csv(csv_path, index_col=0)
    similarity_list = list(csv_df['similarity'])
    db_name_list = list(csv_df[long_col_name])
    cali_name_list = list(csv_df[short_col_name])
    #build non suffix db
    db_name_nonsuffix_list=[clean_suffix(i) for i in db_name_list]
    partial_ratio_nonsuf=[]
    partial_ratio_suf = []
    ratio_nonsuf = []
    ratio_suf = []
    true_partial=[]
    for i in range(len(db_name_list)):
        ratio_suf.append(fuzz.ratio(cali_name_list[i].lower(), db_name_list[i].lower()))
        ratio_nonsuf.append(fuzz.ratio(cali_name_list[i].lower(), db_name_nonsuffix_list[i].lower()))
        partial_ratio_suf.append(fuzz.partial_ratio(cali_name_list[i].lower(), db_name_list[i].lower()))
        partial_ratio_nonsuf.append(fuzz.partial_ratio(cali_name_list[i].lower(), db_name_nonsuffix_list[i].lower()))

        true_partial.append(complete_partial(cali_name_list[i].lower(),db_name_nonsuffix_list[i].lower()))
    df_dict = {short_col_name: cali_name_list,
               long_col_name: db_name_list,
               'similarity': similarity_list,
               'ratio score': ratio_suf,
               'partial ratio score':partial_ratio_suf,
               'db company name nonsuf':db_name_nonsuffix_list,
               'ratio score nonsuf': ratio_nonsuf,
                'partial ratio score nonsuf':partial_ratio_nonsuf,
               'true partial':true_partial
    }
    df_output = pd.DataFrame(df_dict)
    df_output.to_csv(out_path)
    flit_result_match()