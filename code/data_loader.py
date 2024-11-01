# -*- coding: utf-8 -*-
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import os


""" prepare initial embeddings and construct graph """
class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, rating_file, sep, args, transform=None, pre_transform=None):

        self.path = root
        self.dataset = dataset
        self.rating_file = rating_file
        self.sep = sep
        self.store_backup = True
        self.args = args
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.stat_info = torch.load(self.processed_paths[1])
        self.data_num = self.stat_info['data_num']
        self.attr_num = self.stat_info['attr_num']
        # other parameters ...


    def process(self):

        self.userfile  = self.raw_file_names[0]
        self.itemfile  = self.raw_file_names[1]
        self.attrfile = self.raw_file_names[2]
        self.ratingfile  = self.raw_file_names[3]
        graphs, stat_info = self.read_data()

        if not os.path.exists(f"{self.path}processed/{self.dataset}"):
            os.mkdir(f"{self.path}processed/{self.dataset}")
            print(f"data has saved to {self.path}processed/{self.dataset}")
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(stat_info, self.processed_paths[1])

    def read_data(self):

        # load initial embeddings for train and construct graph for trained embeddings
        # ... ...

