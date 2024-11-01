# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import utils
import numpy as np
import torch.optim as optim

""" define a linear model to learn the weights of different attr classes """
class SingleLayerNetwork(nn.Module):
    def __init__(self, input_size, output_size):

        super(SingleLayerNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.auc_list = []
        self.weight = 0
        self.attr_weight_list = []
        
    def forward(self, X, user_vec_dict):

        file_matrix = []
        for x in X:
            file_attr_list = self.linear(x)
            new_file = torch.mean(file_attr_list, dim=0, keepdim=True)
            file_matrix.append(new_file)
        file_vec_list = torch.stack(file_matrix)
        file_vec_list = torch.squeeze(file_vec_list, dim=1)
        user_vec_list = list(user_vec_dict.values())
        user_vec_list = torch.tensor(user_vec_list)
        pred = torch.mm(user_vec_list, file_vec_list.t())
        pred_matrix = torch.sigmoid(pred)

        return pred_matrix


def train_weight(user_vec_dict, train_user_vec_dict, train_file_vec_dict, model_path):
    """
    Pepper weight training model
    """

    file_vec_list = []
    for file in list(train_file_vec_dict.values()):
        file_vec_list.append(file)
    X = torch.tensor(file_vec_list)
    Y = utils.construct_interaction_matrix(train_user_vec_dict, train_file_vec_dict)
    Y = torch.tensor(Y)

    input_size = 64
    output_size = 64
    model = SingleLayerNetwork(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    epochs = 600
    for epoch in range(epochs):
        pred_matrix = model(X, user_vec_dict)
        loss = criterion(pred_matrix, Y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        auc = calculate_accuracy(pred_matrix, Y)
        model.auc_list.append(auc)

        if epoch % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, AUC: {auc:.6f}')

    model.weight = model.linear.weight.detach().numpy()
    eigenvalues = np.linalg.eigvals(model.weight)
    eigenvalues = [arr.tolist() for arr in eigenvalues]
    eigenvalues = [float(complex_num.real) for complex_num in eigenvalues]
    model.attr_weight_list = torch.sigmoid(torch.tensor(eigenvalues))

    if not os.path.exists(model_path):
        torch.save(model, model_path)


def calculate_accuracy(predicted_matrix, true_interaction_matrix, threshold=0.5):

    # calculate the accuracy of the linear model
    # ... ...
