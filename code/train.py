# -*- coding: utf-8 -*-

import os
from utils import print_formatted_current_time
import numpy as np
import torch
from model import Pepper
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
from sklearn.metrics import recall_score
import pandas as pd
from datetime import datetime


def train(args, data_info, model_path):

    # learning process,
    # including:
    # prepare train data, define loss function, bp process, evaluate prediction results and save model
    # ... ...
    # ... ...

def evaluate(model, data_loader, device):

    # evaluate the prediction result ... ...


