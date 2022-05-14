import scipy.io as sio
import torch
from torch_geometric.data import Data
import numpy as np
import random

# 加载mat文件
def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    return a.todense(), x, y

# 构建pyg的Data类
def Net2Graph(file):
    A, X, Y = load_network(file)
    A = torch.tensor(np.where(A == 1), dtype = torch.long)
    X = torch.tensor(X, dtype = torch.float)
    Y = torch.tensor(np.argmax(Y,axis=1))
    graph = Data(x=X, edge_index=A, y=Y)

    # 划分训练集、验证集、测试集
    idx = np.arange(graph.num_nodes)
    random.shuffle(idx)

    train_size = int(graph.num_nodes * 0.7)
    val_size = int(graph.num_nodes * 0.1)
    test_size = int(graph.num_nodes * 0.2)

    train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

    train_mask[idx[:train_size]] = True
    val_mask[idx[train_size: train_size + val_size]] = True
    test_mask[idx[train_size + val_size:]] = True

    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask

    return graph