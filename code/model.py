# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
import numpy as np

""" Define Internal Interaction Network """
class inner_GNN(MessagePassing):
    def __init__(self, dim, hidden_layer):

        super(inner_GNN, self).__init__(aggr='mean')
        self.lin1 = nn.Linear(dim, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_weight=None):

        x = x.squeeze()
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):

        pairwise_analysis = x_i * x_j
        pairwise_analysis = self.lin1(pairwise_analysis)
        pairwise_analysis = self.act(pairwise_analysis)
        pairwise_analysis = self.lin2(pairwise_analysis)
        pairwise_analysis = self.drop(pairwise_analysis)
        if edge_weight != None:
            interaction_analysis = pairwise_analysis * edge_weight.view(-1,1)
        else:
            interaction_analysis = pairwise_analysis
        return interaction_analysis


    def update(self, aggr_out):
        
        return aggr_out


""" Define Cross-Interaction Network """
class cross_GNN(MessagePassing):

    def __init__(self, dim, hidden_layer):

        super(cross_GNN, self).__init__(aggr='mean')

    def forward(self, x, edge_index, edge_weight=None):

         x = x.squeeze()
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):

        pairwise_analysis = x_i * x_j
        if edge_weight != None:
            interaction_analysis = pairwise_analysis * edge_weight.view(-1,1)
        else:
            interaction_analysis = pairwise_analysis
        return interaction_analysis

    def update(self, aggr_out):

        return aggr_out


class Pepper(nn.Module):
    """
    Pepper main model
    """
    def __init__(self, args, n_features, device):

       super(Pepper, self).__init__()
        self.n_features = n_features
        self.dim = args.dim
        self.hidden_layer = args.hidden_layer
        self.device = device
        self.batch_size = args.batch_size
        self.num_user_features = args.num_user_features
        self.feature_embedding = nn.Embedding(self.n_features + 1, self.dim)
        self.node_dict = {}
        self.user_dict = {}
        self.item_dict = {}
        self.inner_gnn = inner_GNN(self.dim, self.hidden_layer)
        self.outer_gnn = cross_GNN(self.dim, self.hidden_layer)
        self.node_weight = nn.Embedding(self.n_features + 1, 1)
        self.node_weight.weight.data.normal_(0.0, 0.01)
        self.update_f = nn.GRU(input_size=self.dim, hidden_size=self.dim, dropout=0.5)

    def forward(self, data, is_training=True):

        torch.set_printoptions(threshold=sys.maxsize)
        node_id = data.x.to(self.device)
        batch = data.batch
        node_w = torch.squeeze(self.node_weight(node_id))
        sum_weight = global_add_pool(node_w, batch)
        node_emb = self.feature_embedding(node_id)
        inner_edge_index = data.edge_index
        outer_edge_index = torch.transpose(data.edge_attr, 0, 1)
        outer_edge_index = self.outer_offset(batch, self.num_user_features, outer_edge_index)
        inner_node_message = self.inner_gnn(node_emb, inner_edge_index)
        outer_node_message = self.outer_gnn(node_emb, outer_edge_index)

        if len(outer_node_message.size()) < len(node_emb.size()):
            outer_node_message = outer_node_message.unsqueeze(1)
            inner_node_message = inner_node_message.unsqueeze(1)
        updated_node_input = torch.cat((node_emb, inner_node_message, outer_node_message), 1)
        updated_node_input = torch.transpose(updated_node_input, 0, 1)
        gru_h0 = torch.normal(0, 0.01, (1, node_emb.size(0), self.dim)).to(self.device)
        gru_output, hn = self.update_f(updated_node_input, gru_h0)
        updated_node = gru_output[-1]
        new_batch = self.split_batch(batch, self.num_user_features)
        updated_graph = torch.squeeze(global_mean_pool(updated_node, new_batch))
        user_graphs, item_graphs = torch.split(updated_graph, int(updated_graph.size(0) / 2))

        user_id_list = node_id[::(self.num_user_features + self.dim)]
        user_id_list = user_id_list.tolist()
        item_id_list = node_id[self.num_user_features::(self.num_user_features + self.dim)]
        item_id_list = item_id_list.tolist()
        j = 0
        while j < len(user_graphs):
            self.user_dict[f'{user_id_list[j][0]}'] = user_graphs[j].tolist()
            self.item_dict[f'{item_id_list[j][0]}'] = item_graphs[j].tolist()
            j += 1

        node_id_list = node_id.tolist()
        updated_node_list = updated_node.tolist()
        i = 0
        for node in node_id_list:
            self.node_dict[f'{node[0]}'] = updated_node_list[i]
            i += 1

        y = torch.unsqueeze(torch.sum(user_graphs * item_graphs, 1) + sum_weight, 1)
        y = torch.sigmoid(y)

        return y

    def split_batch(self, batch, user_node_num):

        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))
        cum_num_list = [cum_num + i for i in range(user_node_num)]
        multi_hot = torch.cat(cum_num_list)
        test = torch.sum(F.one_hot(multi_hot, ones.size(0)), dim=0) * (torch.max(batch) + 1)

        return batch + test

    def outer_offset(self, batch, user_node_num, outer_edge_index):

        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        inter_per_graph = (nodes_per_graph - user_node_num) * user_node_num * 2
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))
        offset_list = torch.repeat_interleave(cum_num, inter_per_graph, dim=0).repeat(2, 1)
        outer_edge_index_offset = outer_edge_index + offset_list
        return outer_edge_index_offset
