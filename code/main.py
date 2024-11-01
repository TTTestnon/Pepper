# -*- coding: utf-8 -*-

import argparse
import os
from train import train
import torch
import weight_method
import utils
from data_loader import Dataset
from torch_geometric.data import DataLoader
import generate_decoy

""" Using evaluation metrics to test the effectiveness of decoy files """
def get_hit_rate(decoy_file_name_list, mode):

    if mode == 'test':
        pkl = 'ransomware_file_reaction_dict_for_test_dict.pkl'
    ransomware_file_dict = utils.load_dict(f'../data/{pkl}')

    result_dict = {}

    # certain method to get detection rate ... ...

    hit_count_list = []  # average file loss list
    miss_list = []  # undetected ransomware list
    for key, value in result_dict.items():
        try:
            loss_file_count = value['loss file count']
            hit_count_list.append(loss_file_count)
        except Exception as e:
            miss_list.append(key)

    average_loss_count = sum(hit_count_list) / len(hit_count_list)

    return result_dict, j, average_loss_count, miss_list




if __name__ == '__main__':

    # Select the mode you want to test,
    # for example, 'test' means comparing with interaction records from the real test set
    mode = 'test'

    args_dict = {
        '--dataset': {'type': str, 'default': 'ransomware', 'help': 'which dataset to use'},
        '--rating_file': {'type': str, 'default': f'interaction_ratings.csv', 'help': 'reaction record file'},
        '--dim': {'type': int, 'default': 64, 'help': 'dimension of entity and relation embeddings'},
        '--l2_weight': {'type': float, 'default': 1e-5, 'help': 'weight of the l2 regularization term'},
        '--lr': {'type': float, 'default': 0.001, 'help': 'learning rate'},
        '--batch_size': {'type': int, 'default': 128, 'help': 'batch size'},
        '--n_epoch': {'type': int, 'default': 50, 'help': 'the number of epochs'},
        '--hidden_layer': {'type': int, 'default': 256, 'help': 'neural hidden layer'},
        '--num_user_features': {'type': int, 'default': 2, 'help': 'the number of user attributes'},
        '--random_seed': {'type': int, 'default': 2024, 'help': 'size of common item be counted'}
    }
    args = prepair_training_args(args_dict)

    datainfo = prepair_dataloader(args)

    # using GNN to learn the interaction between ransomware and user files in the training set
    model_path = '..\\checkpoint\\trained_model.pkl'
    if not os.path.exists(model_path):
        train(args, datainfo, model_path)
    main_model = torch.load(model_path)
    utils.print_formatted_current_time(f'have loaded the trained model form {model_path}')

    # using a linear model to learn the weights of each attribute
    train_user_vec_dict, train_file_vec_dict, user_vec_dict = get_train_vector(main_model)
    model_path = '..\\checkpoint\\trained_weight_model.pkl'
    if not os.path.exists(model_path):
        weight_method.train_weight(user_vec_dict, train_user_vec_dict, train_file_vec_dict, model_path)
    weight_model = torch.load(model_path)
    utils.print_formatted_current_time(f'have loaded trained weight model from {model_path}')

    # generating decoy files
    decoy_file_name_list = generate_decoy.generate_decoy_files(main_model, weight_model, user_vec_dict)
    utils.print_formatted_current_time(f' {len(decoy_file_name_list)} decoy files have been selected')

    # Using evaluation metrics to test the effectiveness of bait files
    result_dict, j, average_loss_count, miss_list  = get_hit_rate(decoy_file_name_list, mode)

    # Presenting downstream tasks of test results ... ...
