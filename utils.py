import numpy as np
import pickle
import torch.optim as optim
from torch.optim import lr_scheduler

def create_number_dict(n):
    """
    创建一个字典，其中键为字符串形式的数字，值为对应的整数
    
    参数:
    n (int): 正整数，表示字典的大小
    
    返回:
    dict: 格式为 {'1': 1, '2': 2, ..., 'n': n} 的字典
    """
    return {str(float(i)): i for i in range(1, n+1)}

import torch

def convert_to_list(tensor):
    if tensor.ndim != 2 or tensor.shape[0] != 2:
        raise ValueError("输入必须是一个 2×N 的张量")
    return tensor.transpose(0, 1).tolist()


def adjacency_to_edge_matrix(adj):
    """
    将邻接矩阵转换为边矩阵（不包含自环）
    
    参数:
        adj (torch.Tensor): M×M的邻接矩阵（支持有向/无向，支持权重/非权重）
    
    返回:
        torch.Tensor: 2×M'的边矩阵，其中M'是非自环边的数量
    """
    # 获取非零元素的索引（包括自环）
    all_edges = torch.nonzero(adj).t()
    
    # 过滤掉自环边 (i == j)
    non_self_loops = all_edges[:, all_edges[0] != all_edges[1]]
    
    return non_self_loops

import random
def add_noise_edge(Aedge, Aadj, edge_noise):
    Aadjn = np.zeros_like(Aadj)
    for u, v in zip(Aedge[0], Aedge[1]):
        if u==v:
            Aadjn[u][v] = 1
        if u<v:
            if random.random() > edge_noise:
                Aadjn[u][v], Aadjn[v][u] = 1, 1
            else:
                while 1:
                    u, v = random.randint(0, Aadj.shape[0]-1), random.randint(0, Aadj.shape[0]-1)
                    if u != v and Aadj[u][v] == 0:
                        Aadjn[u][v], Aadjn[v][u] = 1, 1
                        break
    return Aadjn


