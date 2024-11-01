# -*- coding: utf-8 -*-
import count_score
import utils

"""  pick certain attrs that decoy files should have， take size rank as example """
def generate_decoy_files(main_model, weight_model, user_vec_dict):

    # pick certain attrs that decoy files should have， take size rank as example
    picked_size_rank_attr = count_score.find_certain_attr(main_model, user_vec_dict, weight_model.attr_weight_list)

    picked_attr_list = []
    picked_attr_list.extend(picked_size_rank_attr)
    picked_attr_list = list(set(picked_attr_list))

    item_dict = utils.load_dict('../data/ransomware/item_dict.pkl')
    attr_dict = utils.load_dict('../data/ransomware/attr_dict.pkl')
    picked_attr_index_list = []
    for attr in picked_attr_list:
        for key, value in attr_dict.items():
            if attr == key:
                picked_attr_index_list.append(int(value))

    
    decoy_index_list = []
    for i_key, i_value in item_dict.items():
        for i in picked_attr_index_list:
            
            # certain method to get decoy files
            
            decoy_index_list.append(int(i_value['title']))

    decoy_file_name_list = []
    feature_dict = utils.load_dict('../data/ransomware/attr_dict.pkl')
    for index in decoy_index_list:
        for file_key, file_value in feature_dict.items():
            if file_value == int(index):
                file_name = file_key
                splited_file_name = file_name.split("\\")
                file_name = "\\".join(splited_file_name[-2:])
                decoy_file_name_list.append(file_name)

    final_decoy_file_list = []

    return final_decoy_file_list

